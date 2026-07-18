import os
import json
import inspect

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from functools import partial

import tyro
import torch

from trl import SFTConfig, SFTTrainer
import wandb
from transformers import TrainerCallback, set_seed

from dataset import V1GDataset, postprocess_fn
from v1 import V1ForConditionalGeneration, get_processor, collate_fn


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    resume_from_checkpoint: Optional[str] = None

    data: str = "data/v1g_train.json"
    eval_data: str = "data/v1g_eval_samples.json"
    image_dir: Optional[str] = "data/images"
    max_image_size: int = 672
    fix_image_size: Optional[int] = None  # Set to 448 for 448x448 fixed size

    save_total_limit: int = 2

    cache_dir: str = "checkpoints"
    local_rank: int = 0
    distributed: bool = False
    max_seq_length: Optional[int] = 8192
    batch_size: int = 2
    grad_acc: int = 4
    max_steps: int = -1
    seed: int = 729

    z_loss_weight: float = 1e-5  # 1e-5
    # use_gate: bool = False
    label_smoothing: float = 0.0
    save_eval_steps: int = 200
    separate_copy_loss: bool = False
    save_steps: int = 200
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    skip_final_save: bool = False

    debug: bool = False
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: str = "v1"
    wandb_run_id: Optional[str] = None
    run_name: Optional[str] = None  # override output dir name (use instead of auto-derived exp_name)
    attn_implementation: Optional[str] = None  # override default attn impl (flash_attention_2 otherwise)


def build_exp_name(args):
    model_key = Path(args.model).name.replace(".", "_")
    exp_name = f"v1g_{model_key}"
    if args.fix_image_size is not None:
        exp_name += f"_fix_image_size_{args.fix_image_size}"
    return exp_name


args = tyro.cli(Config)
set_seed(args.seed)

if args.z_loss_weight > 0 and args.separate_copy_loss:
    print(
        "z_loss_weight > 0 when separate_copy_loss True occurs error, known issue in v1/modeling_v1.py"
    )

# build sample data
eval_path = Path(args.eval_data)
if not os.path.exists(eval_path):
    print(f"building dummy eval data at: {eval_path}")
    with open(args.data) as f:
        data = json.load(f)
    with open(eval_path, "w") as f:
        json.dump(data[:8], f)
    print(f"built dummy eval data at: {eval_path}")

    del data

exp_name = args.run_name if args.run_name else build_exp_name(args)

out_dir = Path(args.cache_dir) / exp_name
out_dir.parent.mkdir(exist_ok=True, parents=True)

# Modified model loading code to use checkpoint if provided
print(f"Resuming training from: {args.resume_from_checkpoint}")
model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.model

attn_implementation = (
    args.attn_implementation if args.attn_implementation else "flash_attention_2"
)
model = V1ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    # use_cache=False,  # gradient checkpointing
    z_loss_weight=args.z_loss_weight,
    # use_gate=args.use_gate,
    label_smoothing=args.label_smoothing,
    separate_copy_loss=args.separate_copy_loss,
    do_copy=True,
)
model.after_loading()

if args.local_rank == 0:
    print(f"model config: {model.config}")

processor = get_processor(
    model_path,
    prepend_raw_region_to_text=True,
    separate_copy_loss=args.separate_copy_loss,
)
_postprocess_fn = partial(postprocess_fn, processor=processor)
collate_kwargs = {"processor": processor}
if "max_seq_length" in inspect.signature(collate_fn).parameters:
    collate_kwargs["max_seq_length"] = args.max_seq_length
_collate_fn = partial(collate_fn, **collate_kwargs)

train_dataset = V1GDataset(
    args.data,
    image_dir=args.image_dir,
    max_image_size=args.max_image_size,
    postprocess_fn=_postprocess_fn,
    debug=args.debug,
    fix_image_size=args.fix_image_size,
)
val_dataset = V1GDataset(
    args.eval_data,
    image_dir=args.image_dir,
    max_image_size=args.max_image_size,
    postprocess_fn=_postprocess_fn,
    debug=args.debug,
    fix_image_size=args.fix_image_size,
)


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor=None):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if self.processor:
            self.processor.save_pretrained(checkpoint_dir)
        if wandb.run is not None:
            wandb.log({"checkpoint_dir": checkpoint_dir})


class ResumeControlCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        state.logging_steps = args.logging_steps
        state.eval_steps = args.eval_steps
        state.save_steps = args.save_steps
        if args.max_steps > 0:
            state.max_steps = args.max_steps
        return control


class LoggerTrainer(SFTTrainer):
    def __init__(self, model, *args, **kwargs):
        self._full_model_config = model.config
        # print(f"model_config: {self._full_model_config}")
        super().__init__(model, *args, **kwargs)
        # self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        if torch.isnan(loss) or torch.isinf(loss):
            self.accelerator.print("⚠️  NaN loss detected — skipping batch.")
            return None
        if outputs.z_loss is not None:
            self._metrics[mode]["z_loss"] = [
                self.accelerator.gather_for_metrics(outputs.z_loss.mean()).mean().item()
            ]
        if outputs.gen_loss is not None:
            self._metrics[mode]["gen_loss"] = [
                self.accelerator.gather_for_metrics(outputs.gen_loss.mean())
                .mean()
                .item()
            ]
        if outputs.copy_loss is not None:
            self._metrics[mode]["copy_loss"] = [
                self.accelerator.gather_for_metrics(outputs.copy_loss.mean())
                .mean()
                .item()
            ]
        batch_size = inputs["input_ids"].shape[0]
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(inputs["attention_mask"].sum())
                    .sum()
                    .item()
                )
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(
                    inputs["position_ids"].size(1), device=inputs["position_ids"].device
                )
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
                )
            else:
                raise ValueError(
                    "Expected 'attention_mask' or 'position_ids' in inputs."
                )

            copy_mask = inputs["labels"] >= self._full_model_config.copy_token_start
            num_local_copy_tokens = copy_mask.float().sum(-1).mean()

            num_copy_tokens_in_batch = (
                self.accelerator.gather_for_metrics(num_local_copy_tokens).sum().item()
            )

            self._total_train_tokens += num_tokens_in_batch
            self._metrics[mode]["sample_num_tokens"] = [
                num_tokens_in_batch / batch_size
            ]
            self._metrics[mode]["copy_num_tokens"] = [
                num_copy_tokens_in_batch / batch_size
            ]
        self._metrics[mode]["total_num_tokens"] = [self._total_train_tokens]

        if getattr(outputs, "gen_logits_max", None) is not None:
            self._metrics[mode]["gen_logits_max"] = [
                self.accelerator.gather_for_metrics(outputs.gen_logits_max)
                .mean()
                .item()
            ]
            self._metrics[mode]["gen_logits_mean"] = [
                self.accelerator.gather_for_metrics(outputs.gen_logits_mean)
                .mean()
                .item()
            ]
            self._metrics[mode]["gen_logits_min"] = [
                self.accelerator.gather_for_metrics(outputs.gen_logits_min)
                .mean()
                .item()
            ]
            self._metrics[mode]["gen_logits_std"] = [
                self.accelerator.gather_for_metrics(outputs.gen_logits_std)
                .mean()
                .item()
            ]
        if getattr(outputs, "copy_logits_max", None) is not None:
            self._metrics[mode]["copy_logits_max"] = [
                self.accelerator.gather_for_metrics(outputs.copy_logits_max)
                .mean()
                .item()
            ]
            self._metrics[mode]["copy_logits_min"] = [
                self.accelerator.gather_for_metrics(outputs.copy_logits_min)
                .mean()
                .item()
            ]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if (
            "labels" in inputs
            and not self.args.use_liger_kernel
            and outputs.logits is not None
        ):
            logits = outputs.logits.detach()

            copy_logits = logits[..., self._full_model_config.copy_token_start :]
            gen_logits = logits[..., : self._full_model_config.copy_token_start]

            self._metrics[mode]["copy_logits_max"] = [
                self.accelerator.gather_for_metrics(copy_logits.max(-1).values.mean())
                .mean()
                .item()
            ]
            # self._metrics[mode]["copy_logits_mean"] = [
            #     self.accelerator.gather_for_metrics(copy_logits.mean(-1).mean())
            #     .mean()
            #     .item()
            # ]
            self._metrics[mode]["copy_logits_min"] = [
                self.accelerator.gather_for_metrics(copy_logits.min(-1).values.mean())
                .mean()
                .item()
            ]
            # self._metrics[mode]["copy_logits_std"] = [
            #     self.accelerator.gather_for_metrics(copy_logits.std(-1).mean())
            #     .mean()
            #     .item()
            # ]
            self._metrics[mode]["gen_logits_max"] = [
                self.accelerator.gather_for_metrics(gen_logits.max(-1).values.mean())
                .mean()
                .item()
            ]
            self._metrics[mode]["gen_logits_mean"] = [
                self.accelerator.gather_for_metrics(gen_logits.mean(-1).mean())
                .mean()
                .item()
            ]
            self._metrics[mode]["gen_logits_min"] = [
                self.accelerator.gather_for_metrics(gen_logits.min(-1).values.mean())
                .mean()
                .item()
            ]
            self._metrics[mode]["gen_logits_std"] = [
                self.accelerator.gather_for_metrics(gen_logits.std(-1).mean())
                .mean()
                .item()
            ]

            shift_logits = logits[..., :-1, :]
            shift_labels = inputs["labels"][..., 1:]

            shift_copy_mask = (
                inputs["labels"][..., 1:] >= self._full_model_config.copy_token_start
            )
            shift_gen_mask = ~shift_copy_mask

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            correct_copy_predictions = (
                (predictions == shift_labels) & mask & shift_copy_mask
            )
            total_copy_tokens = (mask & shift_copy_mask).sum()
            correct_copy_tokens = correct_copy_predictions.sum()

            correct_gen_predictions = (
                (predictions == shift_labels) & mask & shift_gen_mask
            )
            total_gen_tokens = (mask & shift_gen_mask).sum()
            correct_gen_tokens = correct_gen_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            correct_copy_tokens = self.accelerator.gather_for_metrics(
                correct_copy_tokens
            )
            total_copy_tokens = self.accelerator.gather_for_metrics(total_copy_tokens)
            correct_gen_tokens = self.accelerator.gather_for_metrics(correct_gen_tokens)
            total_gen_tokens = self.accelerator.gather_for_metrics(total_gen_tokens)
            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (
                (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            )
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

            total_copy_sum = total_copy_tokens.sum()
            copy_accuracy = (
                (correct_copy_tokens.sum() / total_copy_sum).item()
                if total_copy_sum > 0
                else 0.0
            )
            self._metrics[mode]["mean_copy_accuracy"].append(copy_accuracy)

            total_gen_sum = total_gen_tokens.sum()
            gen_accuracy = (
                (correct_gen_tokens.sum() / total_gen_sum).item()
                if total_gen_sum > 0
                else 0.0
            )
            self._metrics[mode]["mean_gen_accuracy"].append(gen_accuracy)

        return (loss, outputs) if return_outputs else loss


kwargs = {}
if args.distributed:
    kwargs = dict(
        deepspeed="./zero3.json",
        local_rank=args.local_rank,
    )

if not args.debug:
    kwargs["report_to"] = "wandb" if args.use_wandb else "none"
    kwargs["save_steps"] = args.save_steps
    kwargs["eval_steps"] = args.save_steps
else:
    kwargs["report_to"] = "none"
    kwargs["save_steps"] = args.save_eval_steps
    kwargs["eval_steps"] = args.save_eval_steps

training_args = SFTConfig(
    output_dir=str(out_dir),  # Directory to save the model
    seed=args.seed,  # propagate CLI seed to trainer + data sampler
    data_seed=args.seed,  # ensures SeedableRandomSampler uses CLI seed
    num_train_epochs=5,  # Number of training epochs
    per_device_train_batch_size=args.batch_size,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=args.grad_acc,  # Steps to accumulate gradients
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=3e-5,  # Learning rate for training
    # lr_scheduler_type="constant",  # Type of learning rate scheduler
    lr_scheduler_type="linear",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=1,  # Steps interval for logging
    # save_steps=100,  # Steps interval for saving
    save_total_limit=args.save_total_limit,  # save little
    load_best_model_at_end=args.eval_strategy != "no",
    eval_strategy=args.eval_strategy,  # Strategy for evaluation
    save_strategy=args.save_strategy,  # Strategy for saving the model
    # save_only_model=True,
    save_only_model=False,
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    # greater_is_better=False,  # Whether higher metric values are better
    # load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.5,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    max_steps=args.max_steps,
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    # report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # Options for gradient checkpointing
    # group_by_length=True,
    length_column_name="input_ids",
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    max_seq_length=args.max_seq_length,
    **kwargs,
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset

if args.local_rank == 0 and not args.debug and args.use_wandb:
    wandb_kwargs = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": exp_name,
        "config": {
            "command_line_args": vars(args),
            "training_config": training_args.to_dict(),
            "model_name": args.model,
            "dataset": "v1g",
        },
    }

    # If wandb_run_id is provided, resume that specific run
    if args.wandb_run_id:
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = (
            "allow"  # Resume if exists, otherwise create new with this ID
        )
        print(f"Resuming wandb run: {args.wandb_run_id}")

    wandb.init(**wandb_kwargs)
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")
    # Log detailed configuration after init
    wandb.config.update(
        {
            "learning_rate": training_args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": training_args.num_train_epochs,
            "max_image_size": args.max_image_size,
        }
    )

# trainer = SFTTrainer(
trainer = LoggerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=_collate_fn,
    processing_class=processor.tokenizer,
    callbacks=[
        ResumeControlCallback(),
        SaveProcessorCallback(processor=processor),
    ],
)

# Add resume_from_checkpoint parameter to train() if checkpoint is provided
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()

if not args.skip_final_save:
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
