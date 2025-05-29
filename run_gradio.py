from dataclasses import dataclass
import torch
from typing import Optional
import tyro
from PIL import Image
import gradio as gr
from v1 import V1ForConditionalGeneration, get_processor
from qwen_vl_utils import smart_resize


@dataclass
class Config:
    model: str = (
        "kjunh/v1-7B"
    )
    port: int = 8888


args = tyro.cli(Config)

processor = get_processor(
    args.model,
    padding_side="left",
)

model = V1ForConditionalGeneration.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
model.eval()

SYSTEM_MESSAGE = """You are a helpful assistant."""
TEMPLATE_PROMPT = "{}\nPlease answer the question using a long-chain reasoning style and think step by step."

def resize(
    img: Image.Image, max_size: Optional[int] = None, min_size: Optional[int] = None
) -> Image.Image:
    width, height = img.size

    # Handle min_size constraint
    if min_size is not None and (width < min_size or height < min_size):
        if width < height:
            scale_factor = min_size / float(width)
        else:
            scale_factor = min_size / float(height)

        width = int(width * scale_factor)
        height = int(height * scale_factor)

    # Handle max_size constraint
    if max_size is not None and (width > max_size or height > max_size):
        if width > height:
            scale_factor = max_size / float(width)
        else:
            scale_factor = max_size / float(height)

        width = int(width * scale_factor)
        height = int(height * scale_factor)

    return img.resize((width, height))


def process_and_generate(image_input: Image.Image, query_text: str):
    if image_input is None:
        raise gr.Error("Please upload an image.")
    if not query_text or not query_text.strip():
        raise gr.Error("Please enter a query.")
    
    width, height = image_input.size
    img_processor_details = processor.image_processor
    
    patch_size = getattr(img_processor_details, 'patch_size', 14)
    merge_size = getattr(img_processor_details, 'merge_size', 1)
    min_pixels = getattr(img_processor_details, 'min_pixels', 224*224)
    max_pixels = getattr(img_processor_details, 'max_pixels', 1605632)

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    image = image_input.resize((resized_width, resized_height))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TEMPLATE_PROMPT.format(query_text)},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text_prompt],
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
    
    generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    
    display_text = generated_text.strip()

    ann_image = (image, [])

    masks, _ = processor.extract_masks(image.size, generated_text)
    
    processed_annotations = []
    for k, (desc, mask_tensor) in masks.items():
        if not mask_tensor.any():
            continue

        nz = torch.nonzero(mask_tensor.cpu(), as_tuple=False)
        
        if nz.numel() == 0: 
            continue
        
        y_coords = nz[:, 0]
        x_coords = nz[:, 1]

        x_min = int(x_coords.min())
        y_min = int(y_coords.min())
        x_max = int(x_coords.max())
        y_max = int(y_coords.max())
        
        if x_min > x_max or y_min > y_max:
            continue

        bounding_box = [x_min, y_min, x_max, y_max]
        processed_annotations.append((bounding_box, f"{k}: {desc}"))
    
    if processed_annotations:
        ann_image = (image, processed_annotations)

    return display_text, ann_image


with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("## v1 Demo")
    gr.Markdown("Upload an image, ask a question. The model will respond and try to highlight relevant image regions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="pil", image_mode="RGB", sources=["upload", "clipboard"])
            query_input = gr.Textbox(
                label="Your Query", 
                placeholder="e.g., Describe the object in the center. How many cars are there?",
                lines=3
            )
            submit_button = gr.Button("Submit", variant="primary")
        
        with gr.Column(scale=1):
            annotated_image_display = gr.AnnotatedImage(label="Image with Annotations", height=480) # Adjust height
            text_output_display = gr.Textbox(label="Model Response", lines=10, interactive=False)

    submit_button.click(
        process_and_generate,
        inputs=[image_input, query_input],
        outputs=[text_output_display, annotated_image_display],
        api_name="generate_and_annotate"
    )
    
    gr.Examples(
        examples=[
            ["https://farm8.staticflickr.com/7028/6680892455_f255f88ccc_z.jpg", "How many bears are in the picture?"],
            ["assets/example_clevr.png", "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all rubber balls. Subtract all yellow shiny things. How many objects are left?"],
            ["assets/example_chart.png", "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: What is the sum of all the values in the ruling group?"]
        ],
        inputs=[image_input, query_input],
        outputs=[text_output_display, annotated_image_display],
        fn=process_and_generate,
        cache_examples=False,
    )

print(f"Launching Gradio demo on http://0.0.0.0:{args.port}")
demo.launch(server_name="0.0.0.0", server_port=args.port)