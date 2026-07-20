"""Materialize the v1g dataset from the Hugging Face Hub into the local layout
expected by train.py: data/images/<id>.png + data/v1g_train.json.

Requires ~90GB free disk (HF cache + extracted images). Run once before training:

    python prepare_data.py
"""

import json
from pathlib import Path

import tyro
from datasets import Image, load_dataset
from tqdm import tqdm

IMAGE_PLACEHOLDER = "<image>\n"


def main(
    repo_id: str = "kjunh/v1g",
    out_dir: str = "data",
    num_proc: int = 4,
):
    out = Path(out_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, split="train", num_proc=num_proc)
    ds = ds.cast_column("image", Image(decode=False))

    ann_path = out / "v1g_train.json"
    count = 0
    with open(ann_path, "w") as f:
        f.write("[")
        for row in tqdm(ds, desc="materializing v1g"):
            idx = row["id"]
            image_path = image_dir / f"{idx}.png"
            if not image_path.exists():
                image_bytes = row["image"]["bytes"]
                if image_bytes is None:
                    raise ValueError(f"row {idx}: image bytes are not embedded")
                image_path.write_bytes(image_bytes)

            conversations = row["conversations"]
            if len(conversations) != 2:
                raise ValueError(f"row {idx}: expected exactly two conversation turns")
            human, gpt = conversations
            if human["from"] != "human" or gpt["from"] != "gpt":
                raise ValueError(f"row {idx}: expected human/gpt conversation roles")
            user_text = human["value"]
            if not user_text.startswith(IMAGE_PLACEHOLDER):
                raise ValueError(f"row {idx}: missing {IMAGE_PLACEHOLDER!r} prefix")
            user_text = user_text[len(IMAGE_PLACEHOLDER) :]

            item = {
                "index": int(idx),
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"images/{idx}.png"},
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": gpt["value"]}],
                    },
                ],
                "regions": json.loads(row["regions"]),
                "image_size": row["image_size"],
                "num_tokens": row["num_tokens"],
            }
            if count:
                f.write(",")
            json.dump(item, f)
            count += 1
        f.write("]")
    print(f"wrote {count} items to {ann_path}, images in {image_dir}")


if __name__ == "__main__":
    tyro.cli(main)
