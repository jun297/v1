from transformers import AutoProcessor
import torch
import os

from v1 import V1ForConditionalGeneration, get_processor
from qwen_vl_utils import process_vision_info

hf_model_path = "kjunh/v1-7B"

processor = get_processor(hf_model_path)

model = V1ForConditionalGeneration.from_pretrained(
    hf_model_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

system_message = """You are a helpful assistant."""
TEMPLATE_PROMPT = "{}\nPlease answer the question using a long-chain reasoning style and think step by step."

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://farm8.staticflickr.com/7028/6680892455_f255f88ccc_z.jpg",
            },
            {
                "type": "text",
                "text": TEMPLATE_PROMPT.format("How many bears are in the picture?"),
            },
        ],
    },
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image = process_vision_info(messages)[0][0]

inputs = processor(
    text=[text],
    images=[image],
    videos=None,
    padding=True,
    return_tensors="pt",
).to("cuda")

sampling_params = dict(
    do_sample=False,
    max_new_tokens=8192,
    use_cache=True,
    repetition_penalty=1.05
)

with torch.inference_mode():
    generated_ids = model.generate(**inputs, **sampling_params)

output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)