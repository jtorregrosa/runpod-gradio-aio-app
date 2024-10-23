import os
import time
import random
import json
import base64
import torch
from diffusers import FluxPipeline
from io import BytesIO
from typing import List, Tuple, Optional
from zipfile import ZipFile
import gradio as gr
from PIL import Image, ImageDraw, UnidentifiedImageError
from huggingface_hub import login
from colorama import Fore, Style, init
init(autoreset=True)

def login_hf(hf_token: str):
    # Log in to Hugging Face using the token
    login(token=hf_token)

    print(f"{Fore.GREEN}Logged into Hugging Face successfully.{Style.RESET_ALL}")

def process_prompts(prompts: str) -> List[str]:
    """Process the multiline string of prompts into a list of individual prompts."""
    if not prompts.strip():
        raise ValueError("Prompts input cannot be empty.")
    return [prompt.strip() for prompt in prompts.strip().split('\n') if prompt.strip()]


def load_pipeline(model_id: str, lora_weights_id: Optional[str], lora_weight_name: Optional[str],
                  lora_scale: float, cpu_offload: bool) -> FluxPipeline:
    """Loads the FluxPipeline model and applies LoRA weights."""
    try:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model '{model_id}': {e}")

    if lora_weights_id:
        try:
            pipe.load_lora_weights(lora_weights_id, weight_name=lora_weight_name)
            pipe.fuse_lora(lora_scale=lora_scale)
        except Exception as e:
            raise RuntimeError(f"Failed to load or fuse LoRA weights: {e}")

    if cpu_offload:
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")   

    return pipe


def infer(
        pipe: FluxPipeline,
        prompt: str,
        width: int,
        height: int,
        images_per_prompt: int,
        guidance_scale: float,
        num_inference_steps: int,
        max_sequence_length: int,
        generator: torch.Generator,
) -> List[Image.Image]:
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if width < 256 or width > 2048 or height < 256 or height > 2048:
        raise ValueError("Width and height must be between 256 and 2048 pixels.")

    try:
        output = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
            num_images_per_prompt=images_per_prompt
        )
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    return output.images

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    except (base64.binascii.Error, UnidentifiedImageError) as e:
        raise ValueError(f"Failed to decode base64 string to image: {e}")


def create_update_lora_image_callback(providers):
    def update_lora_image(model_id: str):
        data = None
        for model in providers:
            if model.get("modelId") == model_id:
                # Return the sample property if it exists, otherwise return None
                data = model.get("sample", None)
                break

        if data:
            # Convert Base64 string to an image and return as PIL Image
            try:
                return base64_to_image(data)
            except ValueError as e:
                return None
        else:
            return None  # Return a default image if no matching LoRA weight is found

    return update_lora_image


with open('lora_providers.json', 'r') as json_file:
    try:
        lora_providers = json.load(json_file)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to load lora_providers.json: {e}")

model_ids = [""] + [item['modelId'] for item in lora_providers]

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Flux.1 LORA Image Generator</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Input Parameters", open=True):
                prompt_input = gr.Textbox(
                    label="Enter your prompts (one per line)",
                    lines=5,
                    placeholder="Enter one prompt per line",
                )
                generate_button = gr.Button("Generate Images")
            with gr.Accordion("Model Settings", open=False):
                model_selector = gr.Dropdown(
                    choices=["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"],
                    label="Select Base Model",
                    value="black-forest-labs/FLUX.1-dev",
                )
                lora_weights_selector = gr.Dropdown(
                    choices=model_ids,
                    label="LORA Weights",
                    value=None,
                )
                lora_image_viewer = gr.Image(label="LORA Sample", type="filepath")
            with gr.Accordion("Output Settings", open=False):
                width_input = gr.Number(
                    label="Output Image Width",
                    minimum=256,
                    maximum=2048,
                    value=1024,
                )
                height_input = gr.Number(
                    label="Output Image Height",
                    minimum=256,
                    maximum=2048,
                    value=1024,
                )
                images_per_prompt_input = gr.Number(
                    label="Images Per Prompt",
                    minimum=1,
                    value=1,
                )
            with gr.Accordion("Pipeline Settings", open=False):
                cpu_offload_checkbox=gr.Checkbox(label="Enable CPU Offload")
                lora_scale_slider = gr.Slider(
                    label="LORA Scale",
                    minimum=0.0,
                    maximum=3,
                    value=1,
                    step=0.1,
                )
                guidance_scale_slider = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.1,
                )
                num_inference_steps_slider = gr.Slider(
                    label="Number of Inference Steps",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                )
                max_sequence_length_slider = gr.Slider(
                    label="Max Sequence Length",
                    minimum=1,
                    maximum=1024,
                    value=512,
                    step=1,
                )
        with gr.Column(scale=1):
            image_gallery = gr.Gallery(
                label="Generated Images",
                columns=3,
                show_download_button=True,
                show_label=True,
                show_fullscreen_button=True,
            )
            progress_output = gr.Textbox(
                label="Progress",
                lines=1,
                interactive=False
            )


    # Define the interaction for generating images
    def generate_and_update_gallery(
            prompts: str,
            base_model: str,
            lora_weights_id,
            width: int,
            height: int,
            cpu_offload: bool,
            lora_scale: float,
            images_per_prompt: int,
            guidance_scale: float,
            num_inference_steps: int,
            max_sequence_length: int
    ):
        for arg_name, arg_value in locals().items():
            print(f"{arg_name} = {arg_value}")
        
        try:
            prompt_list = process_prompts(prompts)
        except ValueError as e:
            yield [], str(e)
            return

        images = []
        lora = next((item for item in lora_providers if item.get("modelId") == lora_weights_id), None)

        try:
            if lora is None:
                pipe = load_pipeline(base_model, None, None, lora_scale, cpu_offload)
            else:
                pipe = load_pipeline(base_model, lora["modelId"], lora["weight_name"], lora_scale, cpu_offload)
        except RuntimeError as e:
            yield [], str(e)
            return

        random_seed = int(time.time())
        generator = torch.Generator("cpu").manual_seed(random_seed)

        for idx, prompt in enumerate(prompt_list):
            try:
                imgs = infer(pipe, prompt, width, height, images_per_prompt, guidance_scale, num_inference_steps,
                             max_sequence_length, generator)
                images.extend(imgs)
                yield images, f"Generating image(s) for prompt {idx + 1}/{len(prompt_list)}"
            except RuntimeError as e:
                yield images, str(e)
                return


    generate_button.click(
        fn=generate_and_update_gallery,
        inputs=[
            prompt_input,
            model_selector,
            lora_weights_selector,
            width_input,
            height_input,
            cpu_offload_checkbox,
            lora_scale_slider,
            images_per_prompt_input,
            guidance_scale_slider,
            num_inference_steps_slider,
            max_sequence_length_slider,
        ],
        outputs=[image_gallery, progress_output],  # Update the gallery incrementally and show progress
    )

    update_lora_image_callback = create_update_lora_image_callback(lora_providers)
    lora_weights_selector.change(
        fn=update_lora_image_callback,
        inputs=[lora_weights_selector],
        outputs=[lora_image_viewer],  # Update the image viewer
    )

if __name__ == "__main__":
    login_hf(os.getenv('HF_TOKEN', ''))
    demo.launch(server_name='0.0.0.0')
