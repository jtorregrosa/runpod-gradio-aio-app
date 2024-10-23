import os
import logging
from typing import List

import gradio as gr
import time
import random
from PIL import Image
from UIBase.ui_base import UIBase
from Utils.utils import load_yaml
from .flux_image_generator import FluxImageGenerator
from huggingface_hub import login

class UIFlux(UIBase):
    def __init__(self):
        self.loras = []
        self.available_lora_ids = []
        self.defaults = {}
        self.logger = logging.getLogger(__name__)
        self.generator = FluxImageGenerator()

    def initialize(self) -> UIBase:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file_path = os.path.join(current_dir, 'config.yaml')
        
        config = load_yaml(yaml_file_path)
        
        self.loras = config['loras']
        self.defaults = config['defaults']['settings']
        self.available_lora_ids = [""] + [item['id'] for item in config['loras']]

        login(token=os.getenv('HF_TOKEN', ''))
        
        return self

    def process_prompt(self, prompt: str) -> List[str]:
        if not prompt.strip():
            raise ValueError("Prompts input cannot be empty.")
        return [prompt.strip() for prompt in prompt.strip().split('\n') if prompt.strip()]

    def submit(
            self,
            prompt,
            model,
            lora_weights,
            width,
            height,
            images_per_prompt,
            cpu_offload,
            lora_scale,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
    ):
        lora = next((item for item in self.loras if item.get('id') == lora_weights), None)
        
        self.generator.initialize(
            model, 
            lora.get('id', None) if lora else None, 
            lora.get('weight_name', None) if lora else None, 
            lora_scale, 
            cpu_offload);
        
        prompt_list = self.process_prompt(prompt)

        if lora:
            prompt_list = [lora.get('trigger', '') + item for item in prompt_list]
        
        progress = gr.Progress()

        def gradio_callback(idx, step, num_prompts, total_steps):
            current_progress = ((idx * total_steps) + step + 1) / (num_prompts * total_steps)
            progress(current_progress)

        for images in self.generator.generate(
            prompt_list,
            model,
            lora_weights,
            width,
            height,
            images_per_prompt,
            lora_scale,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
            gradio_callback
        ):
            yield images

    def clear(self):
        return

    def on_lora_change(self, lora_id):
        lora_scale_slider = gr.Slider(
            label="LORA Scale",
            minimum=0.0,
            maximum=3,
            value=self.defaults['pipeline']['lora_scale'],
            step=0.1,
            visible=False
        )

        lora_weights_description_input = gr.Textbox(
            label="LORA Description",
            value=None,
            visible=False,
        )

        if lora_id != '':
            lora = next((item for item in self.loras if item.get('id') == lora_id), None)
            description = lora.get('description', 'NONE')
            lora_scale_slider = gr.Slider(
                label="LORA Scale",
                minimum=0.0,
                maximum=3,
                value=self.defaults['pipeline']['lora_scale'],
                step=0.1,
                visible=True
            )
            
            lora_weights_description_input = gr.Textbox(
                label="LORA Description",
                value=description,
                visible=True,
            )

        return lora_scale_slider, lora_weights_description_input

    def interface(self):
        with gr.Blocks() as interface:
            title_markdown = gr.Markdown("<h1 style='text-align: center;'>Flux.1 LORA Image Generator</h1>")
            with gr.Row():
                with gr.Column(scale=1):
                    # Input Parameters Accordion
                    with gr.Accordion("Input Parameters", open=True) as input_parameters:
                        prompt_input = gr.Textbox(
                            label="Enter your prompts (one per line)",
                            lines=5,
                            placeholder="Enter one prompt per line",
                        )

                    # Model Settings Accordion
                    with gr.Accordion("Model Settings", open=False) as model_settings:
                        model_selector = gr.Dropdown(
                            choices=["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"],
                            label="Select Base Model",
                            value="black-forest-labs/FLUX.1-dev",
                        )
                        lora_weights_selector = gr.Dropdown(
                            choices=self.available_lora_ids,
                            label="LORA Weights",
                            value=None,
                        )
                        lora_weights_description_input = gr.Textbox(
                            label="LORA Description",
                            value=None,
                            visible=False,
                        )

                    # Output Settings Accordion
                    with gr.Accordion("Output Settings", open=False) as output_settings:
                        width_input = gr.Number(
                            label="Output Image Width",
                            minimum=256,
                            maximum=2048,
                            value=self.defaults['output']['width'],
                        )
                        height_input = gr.Number(
                            label="Output Image Height",
                            minimum=256,
                            maximum=2048,
                            value=self.defaults['output']['height'],
                        )
                        images_per_prompt_input = gr.Number(
                            label="Images Per Prompt",
                            minimum=1,
                            value=self.defaults['output']['images_per_prompt'],
                        )

                    # Pipeline Settings Accordion
                    with gr.Accordion("Pipeline Settings", open=False) as pipeline_settings:
                        cpu_offload_checkbox=gr.Checkbox(
                            label="Enable CPU Offload",
                            value=self.defaults['pipeline']['cpu_offload'],
                        )
                        lora_scale_slider = gr.Slider(
                            label="LORA Scale",
                            minimum=0.0,
                            maximum=3,
                            value=self.defaults['pipeline']['lora_scale'],
                            step=0.1,
                            visible=False,
                        )
                        guidance_scale_slider = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=20.0,
                            value=self.defaults['pipeline']['guidance_scale'],
                            step=0.1,
                        )
                        num_inference_steps_slider = gr.Slider(
                            label="Number of Inference Steps",
                            minimum=1,
                            maximum=100,
                            value=self.defaults['pipeline']['num_inference_steps'],
                            step=1,
                        )
                        max_sequence_length_slider = gr.Slider(
                            label="Max Sequence Length",
                            minimum=1,
                            maximum=1024,
                            value=self.defaults['pipeline']['max_sequence_length'],
                            step=1,
                        )
                    with gr.Row():
                        clear_btn = gr.Button(
                            value="Clear",
                            variant="secondary",
                        )
                        generate_btn = gr.Button(
                            value="Submit",
                            variant="primary",
                        )
                with gr.Column(scale=1):
                    image_gallery = gr.Gallery(
                        label="Generated Images",
                        format="png",
                        columns=3,
                        show_download_button=True,
                        show_label=True,
                        show_fullscreen_button=True,
                    )

            generate_btn.click(
                fn=self.submit,
                inputs=[
                    prompt_input,
                    model_selector,
                    lora_weights_selector,
                    width_input,
                    height_input,
                    images_per_prompt_input,
                    cpu_offload_checkbox,
                    lora_scale_slider,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    max_sequence_length_slider,
                ],
                outputs=[
                    image_gallery
                ],
                queue=True,  # Enable queuing to ensure proper streaming
                show_progress="minimal"
            )

            clear_btn.click(fn=self.clear)

            lora_weights_selector.change(
                fn=self.on_lora_change,
                inputs=[lora_weights_selector],
                outputs=[lora_scale_slider, lora_weights_description_input]
            )

        return interface