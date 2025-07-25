from typing import Tuple, List, Optional, Union
import re
import math
from collections import defaultdict

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import (
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize,
    Qwen2VLImageProcessor,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessorKwargs,
    Qwen2_5_VLProcessor,
)

"""
Qwen2.5-VL does not use AnyRes to my relief.
Things to take into account:
- smart_resize
- temporal dimension
    - grid_t = patches.shape[0] // self.temporal_patch_size
- grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
- merge_size (2)


Usage:

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"


processor = Qwen2_5_VLPointerProcessor.from_pretrained(model_name)
processor.image_processor = Qwen2VLImagePointerProcessor.from_pretrained(model_name)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://example---/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        'role': 'assistant',
        'content': [
            {
                'type': 'text', 'text': '<think>Theres a cat at <|region|>, a dog at <|region|>.</think>A calico cat hanging out with a golden retriever.'
            }
        ]
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
regions = [
    [0, 10, 100, 200],
    [300, 0, 600, 250]
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    regions=[regions]
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")


# Qwen2VLImageProcessor in a nutshell
'(tl tp) c (hlm hm hp) (wlm wm wp) -> (tl hlm wlm hm wm) (c tp hp wp)'
"""


BBOX = Tuple[int, int, int, int]


class PointerProcessor:
    @staticmethod
    def normalize_bbox(image_size: Tuple[int, int], bbox: BBOX):
        w, h = image_size
        bbox = [
            bbox[0] / w,
            bbox[1] / h,
            bbox[2] / w,
            bbox[3] / h,
        ]
        return "[{}]".format(", ".join([f"{v:.2f}" for v in bbox]))

    def get_mask(self, image_size: Tuple[int, int], indices: List[int]):
        width, height = image_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # grid_h = resized_height // self.patch_size // self.merge_size
        grid_w_m = resized_width // self.patch_size // self.merge_size

        mask = torch.zeros(resized_height, resized_width)
        for index in indices:
            index_h = index // grid_w_m
            index_w = index % grid_w_m
            bbox = (
                max(index_w * self.patch_size * self.merge_size, 0),
                max(index_h * self.patch_size * self.merge_size, 0),
                min((index_w + 1) * self.patch_size * self.merge_size, resized_width),
                min((index_h + 1) * self.patch_size * self.merge_size, resized_height),
            )
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 1
        # mask = mask.t()  # to width, height
        return mask, (resized_width, resized_height)

    def get_patch_pointers(
        self, image_size: Tuple[int, int], region: Union[BBOX, np.ndarray]
    ):
        if isinstance(region, np.ndarray):
            return self.get_mask_patch_pointers(image_size, region)
        else:
            return self.get_bbox_patch_pointers(image_size, region)

    def get_bbox_patch_pointers(self, image_size: Tuple[int, int], bbox: BBOX):
        factor = self.merge_size
        # factor = 1
        width, height = image_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        x0, y0, x1, y1 = bbox
        resized_bbox = [
            max(x0 / width * resized_width, 0),
            max(y0 / height * resized_height, 0),
            min(x1 / width * resized_width, resized_width),
            min(y1 / height * resized_height, resized_height),
        ]
        # patch_bbox = [v / self.patch_size / self.merge_size for v in resized_bbox]
        patch_bbox = [v / self.patch_size / factor for v in resized_bbox]
        x0, y0, x1, y1 = patch_bbox
        boundaries = [
            math.floor(x0),
            math.floor(y0),
            math.ceil(x1),
            math.ceil(y1),
        ]
        x0, y0, x1, y1 = boundaries

        # t, h, w
        grid_w = resized_width // self.patch_size
        grid_w_m = grid_w // factor
        rows, cols = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        grid_indices = np.column_stack((rows.ravel(), cols.ravel()))
        indices = grid_indices[:, 0] * grid_w_m + grid_indices[:, 1]
        base_ids = list(indices)
        ids = np.array(base_ids)
        return ids

    def get_mask_patch_pointers(self, image_size: Tuple[int, int], mask: np.ndarray):
        # mask size: w h
        width, height = image_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        grid_w_m = resized_width // self.patch_size // self.merge_size
        grid_h_m = resized_height // self.patch_size // self.merge_size

        m = torch.from_numpy(mask).float()
        m = F.interpolate(
            m[None, None], (grid_h_m, grid_w_m), mode="bilinear", antialias="bilinear"
        )[0, 0]

        grid_indices = m.nonzero(as_tuple=False)
        indices = grid_indices[:, 0] * grid_w_m + grid_indices[:, 1]
        ids = indices.numpy()
        return ids

    def renormalize(self, tensor):
        # crude - non-accurate implementation for the lazy
        mean = np.array(self.image_mean).mean()
        std = np.array(self.image_std).mean()
        return tensor * std + mean

class Qwen2VLImagePointerProcessor(Qwen2VLImageProcessor, PointerProcessor):
    pass


class V1Processor(Qwen2_5_VLProcessor):
    image_processor_class = "Qwen2VLImagePointerProcessor"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        prepend_raw_region_to_text: bool = True,
        separate_copy_loss: bool = False,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

        self.region_token = "<|region|>"
        self.copy_token_start = None
        self.prepend_raw_region_to_text = prepend_raw_region_to_text
        self.separate_copy_loss = separate_copy_loss
        self.copy_start_token = "<|box_start|>"
        self.copy_end_token = "<|box_end|>"

    def extract_masks(self, image_size: Tuple[int, int], text: str):
        # Match full detect(...) blocks and extract their content
        # detect_pattern = r"detect\([^)]+objects\s*=\s*\[(.*?)\]\)"
        detect_pattern = r'detect\(\s*query\s*=\s*"([^"]+)"\s*,\s*objects\s*=\s*\["((?:[^"\\]|\\.)*)"\]\s*\)'
        obj_region_pattern = r"<obj(\d+)><region>\[.*?\](.*?)</region>"
        copy_pattern = r"<\|copy_(\d+)\|>"

        # results = defaultdict(list)
        results = {}

        for detect_match in re.finditer(detect_pattern, text, re.DOTALL):
            query_str = detect_match.group(1)
            objects_content = detect_match.group(2)

            for obj_match in re.finditer(
                obj_region_pattern, objects_content, re.DOTALL
            ):
                obj_index = int(obj_match.group(1))
                region_content = obj_match.group(2)
                copy_ids = [int(m) for m in re.findall(copy_pattern, region_content)]
                obj_key = f"<obj{obj_index}>"
                results[obj_key] = (query_str, copy_ids)

        results = dict(results)

        masks = {}
        resized_image_size = image_size
        for k, (desc, region) in results.items():
            mask, resized_image_size = self.image_processor.get_mask(image_size, region)
            masks[k] = (desc, mask)
        return masks, resized_image_size

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        videos: VideoInput = None,
        regions: Optional[List[dict[str, Union[BBOX, np.ndarray]]]] = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            regions:
                either bboxes: List[dict[str, Tuple[int, int, int, int]]]
                or masks: List[dict[str, np.ndarray[width, height]]]
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        obj_ptrs = None
        if images is not None:
            image_inputs = self.image_processor(
                images=images, videos=None, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]

            for image in images:
                assert isinstance(
                    image, Image.Image
                ), "only supporting a single image per row for now"

            if regions is not None:
                obj_ptrs = [
                    {
                        name: (
                            self.image_processor.get_patch_pointers(image.size, region)
                            if region is not None
                            else np.array([])
                        )
                        for name, region in image_region.items()
                    }
                    for image, image_region in zip(images, regions)
                ]
        else:
            image_inputs = {}
            image_grid_thw = None

        assert videos is None, "video inputs are not supported yet"  # TODO
        if videos is not None:
            videos_inputs = self.image_processor(
                images=None, videos=videos, **output_kwargs["images_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [
                    self.image_processor.temporal_patch_size / fps
                ] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [
                    self.image_processor.temporal_patch_size / tmp for tmp in fps
                ]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

            if obj_ptrs is not None:
                assert regions is not None
                for i in range(len(text)):
                    image_ptrs = obj_ptrs[i]
                    image_region = regions[i]

                    for name, region in image_region.items():
                        region_ptr = image_ptrs[name]

                        assert name in text[i], f"object {name} not found in: {text[i]}"

                        ptrs_str = "".join([f"<|copy_{j}|>" for j in region_ptr])
                        region_str = self.image_processor.normalize_bbox(
                            image.size, region
                        )
                        if self.separate_copy_loss:
                            ptrs_str = (
                                self.copy_start_token + ptrs_str + self.copy_end_token
                            )
                        out_str = ("<region>" + ptrs_str + "</region>",)
                        if self.prepend_raw_region_to_text:
                            out_str = "<region>" + region_str + ptrs_str + "</region>"

                        text[i] = text[i].replace(name, out_str)

                    for name in image_region.keys():
                        assert name not in text[i]

        if video_grid_thw is not None:
            # TODO: support video inputs
            raise NotImplementedError("video inputs are not yet supported")
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<patch>"
                        + "<|placeholder|>"
                        * (video_grid_thw[index].prod() // merge_length)
                        + "</patch>",
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})


def get_processor(model_name: str, **kwargs):
    processor = V1Processor.from_pretrained(model_name, **kwargs)
    processor.image_processor = Qwen2VLImagePointerProcessor.from_pretrained(
        model_name, **kwargs
    )
    # max_position_tokens = processor.tokenizer.model_max_length
    # new_tokens = [f"<|copy_{i}|>" for i in range(max_position_tokens)]  # too slow
    processor.tokenizer.orig_vocab_size = len(processor.tokenizer)
    new_tokens = [f"<|copy_{i}|>" for i in range(30000)]
    processor.tokenizer.add_tokens(new_tokens)
    processor.copy_token_start = processor.tokenizer.convert_tokens_to_ids("<|copy_0|>")
    return processor


# Create a data collator to encode text and image pairs
def collate_fn(examples, processor):
    convs = [row["conversation"] for row in examples]
    regions = [row["region"] for row in examples]
    image_sizes = [row["image_size"] for row in examples]

    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in convs
    ]  # Prepare texts for processing
    image_inputs = [
        process_vision_info(conv)[0][0] for conv in convs
    ]  # Process the images to extract inputs
    image_inputs = [
        image.resize(image_size) for image, image_size in zip(image_inputs, image_sizes)
    ]

    # Tokenize the texts and process the images
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=None,
        regions=regions,
        padding=True,
        return_tensors="pt",
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = (
        -100
    )  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    image_tokens = [
        151652,
        151653,
        151655,
    ]  # Specific image token IDs for Qwen2VLProcessor

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

if __name__ == '__main__':
    import ipdb; ipdb.set_trace()