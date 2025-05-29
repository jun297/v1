import math
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)

from .configuration_v1 import V1Config


def init_identity(layer, scale: float = 1):
    if isinstance(layer, nn.Linear):
        with torch.no_grad():
            # Ensure weight matrix is square
            rows, cols = layer.weight.shape
            identity_matrix = (
                torch.eye(rows, cols) * scale
            )  # Creates an identity matrix
            layer.weight.copy_(
                identity_matrix
            )  # Copy identity matrix into layer weights
            if hasattr(layer, "bias"):
                layer.bias.fill_(0)  # Set bias to zero (or another value if needed)


@dataclass
class V1CausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    z_loss: torch.Tensor = None
    gen_loss: torch.Tensor = None
    copy_loss: torch.Tensor = None


class V1ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = V1Config

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.model = Qwen2_5_VLModel(config)
        self.copy_init_scale = 1 / math.sqrt(self.config.hidden_size)

        # self.tokenizer_vocab_size = (
        #     config.tokenizer_vocab_size
        # )  # Qwen2.5-VL: different from embedding_size==vocab_size. 151665 vs. 152064
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        if self.config.do_copy:
            if self.config.tie_copy_heads:
                self._copy_head = nn.Linear(config.hidden_size, config.copy_hidden_size)
            else:
                self._copy_q_head = nn.Linear(
                    config.hidden_size, config.copy_hidden_size
                )
                self._copy_k_head = nn.Linear(
                    config.hidden_size, config.copy_hidden_size
                )
            if self.config.use_gate:
                self.gate = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def after_loading(self):
        if self.config.do_copy:
            self.init_heads()
            if self.config.use_gate:
                self.lm_head.weight.data = self.lm_head.weight.data * 2
                self.gate.weight.data.fill_(0)

    @property
    def copy_q_head(self):
        return self._copy_head if self.config.tie_copy_heads else self._copy_q_head

    @property
    def copy_k_head(self):
        return self._copy_head if self.config.tie_copy_heads else self._copy_k_head

    def init_heads(self):
        if hasattr(self, "_copy_head"):
            init_identity(self._copy_head, self.copy_init_scale)
        if hasattr(self, "_copy_k_head"):
            init_identity(self._copy_k_head, self.copy_init_scale)
        if hasattr(self, "_copy_q_head"):
            init_identity(self._copy_q_head, self.copy_init_scale)

    def copy_representations(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
        copy_values: Optional[torch.FloatTensor] = None,
    ):
        if copy_values is None:
            mask = input_ids == self.config.image_token_id
            copy_values, _ = self.extract_image_tokens(inputs_embeds, mask)  # initial
        assert copy_values is not None
        copy_values = copy_values.to(inputs_embeds.device)
        input_ids = input_ids.to(inputs_embeds.device)

        input_ids = input_ids.clone()
        input_ids = input_ids - self.config.copy_token_start
        copy_mask = input_ids >= 0
        input_ids[~copy_mask] = 0

        assert copy_values is not None
        extracted = copy_values.gather(
            1, input_ids[..., None].repeat(1, 1, copy_values.shape[-1])
        )
        copy_mask = copy_mask.to(extracted.dtype)[..., None]
        return copy_mask * extracted + (1 - copy_mask) * inputs_embeds

    def extract_image_tokens(self, features: torch.FloatTensor, mask: torch.Tensor):
        out_feat, out_mask = extract_image_tokens_right_pad(features, mask)
        return out_feat, out_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_ids = input_ids.clone()
        input_ids_with_ptrs = input_ids.clone()
        input_ids[input_ids >= self.config.copy_token_start] = (
            self.config.region_token_id
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                raise NotImplementedError("video inputs are not supported yet.")
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if self.config.do_copy:
            copy_keys, copy_keys_mask = None, None
            copy_values, copy_values_mask = None, None

            has_cache = bool(past_key_values)
            if has_cache:
                copy_keys, copy_values = past_key_values[len(past_key_values) - 2]
                copy_keys_mask, copy_values_mask = past_key_values[
                    len(past_key_values) - 1
                ]
                # we add channel dim to the mask for consistency in tensor shape in cache
                copy_keys_mask = copy_keys_mask[..., 0]
                copy_values_mask = copy_values_mask[..., 0]

            inputs_embeds = self.copy_representations(
                inputs_embeds, input_ids_with_ptrs, copy_values
            )

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (
            attention_mask is None or attention_mask.ndim == 2
        ):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        gen_logits = self.lm_head(hidden_states)

        if self.config.do_copy:
            assert (
                self.config.copy_extraction_layer == -1
            ), f"copy_extraction_layer should be -1: {self.config.copy_extraction_layer}"
            copy_hidden_states = hidden_states
            copy_q_states = copy_hidden_states
            if self.config.normalize_copy_states:
                copy_q_states = F.normalize(copy_q_states, 2, -1)
            copy_q_states = self.copy_q_head(copy_q_states)

            present_key_values = outputs.past_key_values

            if not has_cache:
                mask = input_ids == self.config.image_token_id
                copy_k_states = (
                    inputs_embeds
                    if self.config.use_embeddings_as_keys
                    else copy_hidden_states
                )
                if self.config.normalize_copy_states:
                    copy_k_states = F.normalize(copy_k_states, 2, -1)
                copy_k_states, copy_k_mask = self.extract_image_tokens(
                    self.copy_k_head(copy_k_states), mask
                )
                copy_v_states, copy_v_mask = self.extract_image_tokens(
                    inputs_embeds.detach(), mask
                )

                # we add channel dim to the mask for consistency in tensor shape in cache
                copy_memories = [
                    (copy_k_states.detach(), copy_v_states.detach()),
                    (copy_k_mask[..., None], copy_v_mask[..., None]),
                ]

                if use_cache:
                    # only update at the first iteration
                    start = len(present_key_values)
                    for i, mem in enumerate(copy_memories):
                        present_key_values.update(*mem, start + i)
            else:
                copy_k_states = copy_keys
                copy_k_mask = copy_keys_mask

            assert copy_k_states is not None
            assert copy_k_mask is not None
            assert (
                copy_k_states.shape[1] > 0
            ), f"zero image tokens on batch elements: {copy_k_mask.sum(dim=1)}"

            copy_logits = (copy_q_states @ copy_k_states.transpose(-1, -2)).to(
                gen_logits.device
            ) * self.copy_init_scale

            if hasattr(self, "gate"):
                gate = torch.sigmoid(self.gate(hidden_states))
                gen_logits = gen_logits * (1 - gate)
                copy_logits = copy_logits * gate

            copy_logits = copy_logits.masked_fill(
                ~copy_k_mask[:, None, :].to(copy_logits.device),
                torch.finfo(copy_logits.dtype).min,
            )
            logits = torch.cat(
                [gen_logits[..., : self.config.copy_token_start], copy_logits], dim=-1
            )
        else:
            logits = gen_logits
            loss = None
            z_loss = None
            gen_loss = None
            if labels is not None:
                gen_logits = gen_logits.float()
                shift_gen_logits = gen_logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                gen_loss_fct = CrossEntropyLoss(reduction="none")
                gen_logits_flat = shift_gen_logits.view(-1, shift_gen_logits.shape[-1])
                gen_labels_flat = shift_labels.view(-1)

                gen_loss_all = gen_loss_fct(gen_logits_flat, gen_labels_flat)
                gen_loss = gen_loss_all.mean()

                loss = gen_loss

                if self.config.z_loss_weight > 0:
                    valid_mask = shift_labels >= 0
                    # top-k approx z_loss for better memory usage
                    top_logits, _ = torch.topk(
                        shift_gen_logits, k=self.config.z_loss_top_k, dim=-1
                    )
                    lse = torch.logsumexp(top_logits, dim=-1)
                    z_loss = lse[valid_mask].pow(2).mean() * self.config.z_loss_weight

                    # z_loss = (
                    #     torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
                    #     * self.config.z_loss_weight
                    # )
                    loss = loss + z_loss
                    z_loss = z_loss.detach()

            return V1CausalLMOutputWithPast(
                loss=loss,
                z_loss=z_loss,
                gen_loss=gen_loss,
                copy_loss=None,
                logits=logits,
                # copy_logits=copy_logits,
                # gen_logits=gen_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )

        loss = None
        z_loss = None
        gen_loss = None
        copy_loss = None
        if labels is not None:
            if self.config.separate_copy_loss:
                # Shift labels and logits for next-token prediction
                shift_gen_logits = gen_logits[:, :-1, :].contiguous().float()
                shift_copy_logits = copy_logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                shift_logits = shift_copy_logits

                # Build masks
                gen_mask = shift_labels < self.config.copy_token_start
                copy_mask = shift_labels >= self.config.copy_token_start

                # Generation loss
                if gen_mask.any():
                    gen_loss_fct = CrossEntropyLoss(reduction="none")

                    G = shift_gen_logits.shape[-1]
                    gen_logits_flat = shift_gen_logits.view(-1, G)
                    gen_labels_flat = shift_labels.view(-1)
                    gen_mask_flat = gen_mask.view(-1)
                    # mask logits
                    gen_logits_flat_masked = gen_logits_flat[gen_mask_flat]
                    gen_labels_flat_masked = gen_labels_flat[gen_mask_flat]

                    gen_loss_all = gen_loss_fct(
                        gen_logits_flat_masked, gen_labels_flat_masked
                    )
                    gen_loss = gen_loss_all.mean()

                # Copy loss (adjust label indices to match copy_logits range)
                if copy_mask.any():
                    copy_loss_fct = CrossEntropyLoss(reduction="none")
                    C = shift_copy_logits.shape[-1]
                    copy_logits_flat = shift_copy_logits.view(-1, C)
                    copy_labels_flat = (
                        shift_labels.view(-1) - self.config.copy_token_start
                    )
                    copy_mask_flat = copy_mask.view(-1)
                    copy_logits_flat_masked = copy_logits_flat[copy_mask_flat]
                    copy_labels_flat_masked = copy_labels_flat[copy_mask_flat]
                    copy_loss_all = copy_loss_fct(
                        copy_logits_flat_masked, copy_labels_flat_masked
                    )
                    copy_loss = copy_loss_all.mean()
            else:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
                total_vocab_size = logits.shape[-1]  # gen + copy
                shift_logits = shift_logits.view(-1, total_vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                gen_loss = loss_fct(shift_logits, shift_labels)

            loss = 0.0
            if gen_loss is not None:
                loss += gen_loss
            if copy_loss is not None:
                loss += copy_loss

            if self.config.z_loss_weight > 0:
                valid_mask = shift_labels >= 0
                # top-k approx z_loss for better memory usage
                top_logits, _ = torch.topk(
                    shift_logits, k=self.config.z_loss_top_k, dim=-1
                )
                lse = torch.logsumexp(top_logits, dim=-1)
                z_loss = lse[valid_mask].pow(2).mean() * self.config.z_loss_weight

                # z_loss = (
                #     torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
                #     * self.config.z_loss_weight
                # )
                loss = loss + z_loss
                z_loss = z_loss.detach()

            if gen_loss is not None:
                gen_loss = gen_loss.detach()
            if copy_loss is not None:
                copy_loss = copy_loss.detach()

        if self.config.use_cfg:
            # expand as max_size for logit processors
            extended_vocab_size = self.config.vocab_size + self.config.copy_token_num
            B, L, V = logits.shape
            pads = torch.full(
                (B, L, extended_vocab_size - V),
                torch.finfo(gen_logits.dtype).min,
                device=logits.device,
            ).to(logits.dtype)
            logits = torch.cat([logits, pads], dim=-1)
            # logits = logits.clamp_min(-1e4)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        logits = logits.float()
        return V1CausalLMOutputWithPast(
            loss=loss,
            z_loss=z_loss,
            gen_loss=gen_loss,
            copy_loss=copy_loss,
            logits=logits,
            # copy_logits=copy_logits,
            # gen_logits=gen_logits,
            past_key_values=present_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


def extract_image_tokens_right_pad(features: torch.FloatTensor, mask: torch.Tensor):
    X, M = features, mask.long()  # bool is not supported for sort in CUDA
    B, L, _ = X.shape
    device = X.device
    M = M.to(device)

    # Compute number of valid elements per batch
    valid_counts = M.sum(dim=1)  # Shape: [B]
    # Replace `.item()` with `max()` and `clamp_min()` for Torch Dynamo compatibility
    R = valid_counts.max().clamp_min(1)  # Ensures at least 1 for tensor compatibility
    # Create index tensors for selection
    sorted_indices = M.argsort(dim=1, descending=True)  # Move True values to front
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

    # Gather sorted X based on mask sorting
    X_sorted = X[batch_indices, sorted_indices]  # Shape: [B, L, C]
    X_selected = X_sorted[:, :R, :]  # Select the top valid elements per batch

    # Create new mask M2 using `torch.arange`
    M2 = torch.arange(L, device=device).expand(B, L) < valid_counts.unsqueeze(1)
    M2 = M2[:, :R]  # Trim to selected size

    # Set out-of-bound values to zero
    X_selected = torch.where(M2.unsqueeze(-1), X_selected, torch.zeros_like(X_selected))

    return X_selected, M2


__all__ = ["V1ForConditionalGeneration"]
