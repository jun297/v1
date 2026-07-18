"""Materialize the v1g dataset from the Hugging Face Hub into the local layout
expected by train.py: data/images/<id>.png + data/v1g_train.json.

Requires ~90GB free disk (HF cache + extracted images). Run once before training:

    python prepare_data.py
"""

import json
from pathlib import Path

import tyro
from datasets import load_dataset
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

    items = []
    for row in tqdm(ds, desc="materializing v1g"):
        idx = row["id"]
        image_path = image_dir / f"{idx}.png"
        if not image_path.exists():
            row["image"].save(image_path)

        human, gpt = row["conversations"][0], row["conversations"][1]
        assert human["from"] == "human" and gpt["from"] == "gpt"
        user_text = human["value"]
        if user_text.startswith(IMAGE_PLACEHOLDER):
            user_text = user_text[len(IMAGE_PLACEHOLDER) :]

        items.append(
            {
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
        )

    ann_path = out / "v1g_train.json"
    with open(ann_path, "w") as f:
        json.dump(items, f)
    print(f"wrote {len(items)} items to {ann_path}, images in {image_dir}")


if __name__ == "__main__":
    tyro.cli(main)
