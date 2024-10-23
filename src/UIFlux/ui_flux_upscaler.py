import os
import logging
from typing import List

import gradio as gr
import time
import random
from PIL import Image
from UIBase.ui_base import UIBase
from Utils.utils import load_yaml
from .flux_image_upscaler import FluxImageUpscaler
from huggingface_hub import login

class UIFluxUpscaler(UIBase):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generator = FluxImageUpscaler()

    def initialize(self) -> UIBase:
        return self

    def submit(
            self,
            control_image,
            scale_factor,
            cpu_offload,
            controlnet_conditioning_scale,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
    ):
        self.generator.initialize(cpu_offload);
        progress = gr.Progress()

        def gradio_callback(step, total_steps):
            current_progress = (step + 1) / total_steps
            progress(current_progress)

        generated_image = self.generator.generate(
            control_image,
            scale_factor,
            controlnet_conditioning_scale,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
            gradio_callback
        )
        
        return generated_image

    def clear(self):
        return

    def interface(self):
        with gr.Blocks() as interface:
            title_markdown = gr.Markdown("<h1 style='text-align: center;'>Flux.1 Image Upscaler</h1>")
            with gr.Row():
                with gr.Column(scale=1):
                    # Input Parameters Accordion
                    with gr.Accordion("Input Parameters", open=True) as input_parameters:
                        control_image_selector = gr.Image(
                            label="Control Image",
                            type="pil",
                        )

                    # Output Settings Accordion
                    with gr.Accordion("Output Settings", open=False) as output_settings:
                        scale_factor_slider = gr.Slider(
                            label="Scale Factor",
                            minimum=1,
                            maximum=8,
                            value=2,
                            step=0.5,
                        )

                    # Pipeline Settings Accordion
                    with gr.Accordion("Pipeline Settings", open=False) as pipeline_settings:
                        cpu_offload_checkbox=gr.Checkbox(
                            label="Enable CPU Offload",
                            value=True,
                        )
                        controlnet_conditioning_scale_slider = gr.Slider(
                            label="Controlnet Conditioning Scale",
                            minimum=0.0,
                            maximum=3,
                            value=0.6,
                            step=0.1,
                            visible=True,
                        )
                        guidance_scale_slider = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=20.0,
                            value=3.5,
                            step=0.1,
                        )
                        num_inference_steps_slider = gr.Slider(
                            label="Number of Inference Steps",
                            minimum=1,
                            maximum=100,
                            value=28,
                            step=1,
                        )
                        max_sequence_length_slider = gr.Slider(
                            label="Max Sequence Length",
                            minimum=1,
                            maximum=1024,
                            value=512,
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
                    generated_image = gr.Image(
                        label="Generated Image",
                        format="png"
                    )

            generate_btn.click(
                fn=self.submit,
                inputs=[
                    control_image_selector,
                    scale_factor_slider,
                    cpu_offload_checkbox,
                    controlnet_conditioning_scale_slider,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    max_sequence_length_slider,
                ],
                outputs=[
                    generated_image
                ],
                queue=True,  # Enable queuing to ensure proper streaming
                show_progress="minimal"
            )

            clear_btn.click(fn=self.clear)

        return interface