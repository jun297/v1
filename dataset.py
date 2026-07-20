from typing import Optional, Callable
import math
import json

import torch

IMAGE_PLACEHOLDER = "<image>\n"
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def postprocess_fn(example, region, image_size, processor):
    text = processor.apply_chat_template(
        example, tokenize=False, add_generation_prompt=False
    )
    return {
        "input_ids": processor.tokenizer.encode(
            text, return_tensors="pt"
        ),  # only used for batch grouping
        "conversation": example,
        "region": region,
        "image_size": image_size,
    }


class V1GDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        max_image_size: int = 672,
        postprocess_fn: Optional[Callable] = None,
        debug: bool = False,
        fix_image_size: Optional[int] = None,
    ):
        self.name: str = "v1g"
        self.postprocess_fn = postprocess_fn

        self.max_image_size = max_image_size
        self.fix_image_size = fix_image_size

        print(f"v1g dataset: loaded {len(data)} items")
        self.data = data

    def __len__(self):
        return len(self.data)

    def resize_image(self, image_size):
        width, height = image_size
        if width > self.max_image_size or height > self.max_image_size:
            scaling_factor = min(
                self.max_image_size / width, self.max_image_size / height
            )
            width = int(width * scaling_factor)
            height = int(height * scaling_factor)

            height, width = smart_resize(height, width)
        return width, height

    def resize_bbox(self, bbox, prev_size, new_size):
        x1, y1, x2, y2 = bbox
        return [
            math.floor(x1 * new_size[0] / prev_size[0]),
            math.floor(y1 * new_size[1] / prev_size[1]),
            math.ceil(x2 * new_size[0] / prev_size[0]),
            math.ceil(y2 * new_size[1] / prev_size[1]),
        ]

    def __getitem__(self, idx: int):
        row = self.data[idx]

        bbox_dt = json.loads(row["regions"])
        conversations = row["conversations"]
        if len(conversations) != 2:
            raise ValueError(f"row {row['id']}: expected exactly two conversation turns")
        human, gpt = conversations
        if human["from"] != "human" or gpt["from"] != "gpt":
            raise ValueError(f"row {row['id']}: expected human/gpt conversation roles")
        user_text = human["value"]
        if not user_text.startswith(IMAGE_PLACEHOLDER):
            raise ValueError(
                f"row {row['id']}: missing {IMAGE_PLACEHOLDER!r} prefix"
            )

        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["image"]},
                    {
                        "type": "text",
                        "text": user_text[len(IMAGE_PLACEHOLDER) :],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gpt["value"]}],
            },
        ]

        original_image_size = tuple(row["image_size"])

        if self.fix_image_size is not None:
            image_size = (self.fix_image_size, self.fix_image_size)
        else:
            image_size = self.resize_image(original_image_size)

        if image_size != original_image_size:
            bbox_dt = {
                k: self.resize_bbox(bbox, original_image_size, image_size)
                for k, bbox in bbox_dt.items()
            }

        if self.postprocess_fn is not None:
            return self.postprocess_fn(conv, bbox_dt, image_size)
        return conv, bbox_dt, image_size
