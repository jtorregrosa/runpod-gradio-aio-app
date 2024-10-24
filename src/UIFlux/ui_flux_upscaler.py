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

logger = logging.getLogger(__name__)

class UIFluxUpscaler(UIBase):
    def __init__(self):
        super().__init__()
        self.generator = FluxImageUpscaler()
        self.logger = logger

    def initialize(self) -> UIBase:
        return self

    def submit(
        self,
        control_image: Image,
        scale_factor: float,
        cpu_offload: bool,
        controlnet_conditioning_scale: float,
        guidance_scale: float,
        num_inference_steps: int,
        max_sequence_length: int,
    ):
        try:
            self.logger.info("Submission started with scale factor: %s", scale_factor)

            # Initialize the generator with the given settings
            self.generator.initialize(cpu_offload)

            progress = gr.Progress()
            progress(0, "Preparing...")

            def gradio_callback(step, total_steps):
                current_progress = (step + 1) / total_steps
                progress(current_progress, f"Step {step+1}/{total_steps}")

            # Generate image
            start_time = time.time()
            generated_image = self.generator.generate(
                control_image,
                scale_factor,
                controlnet_conditioning_scale,
                guidance_scale,
                num_inference_steps,
                max_sequence_length,
                gradio_callback
            )
            end_time = time.time()

            self.logger.info("Image generated successfully in %.2f seconds.", end_time - start_time)
            return generated_image
        except Exception as e:
            self.logger.error("Error occurred during image generation: %s", str(e))
            raise

    def clear(self):
        # Add a log message for the clear function
        self.logger.info("Clear function called. Clearing all inputs and outputs.")
        # Implement any necessary cleanup code if required
        return

    def interface(self):
        try:
            self.logger.info("Setting up the Gradio interface.")
            with gr.Blocks() as interface:
                gr.Markdown("<h1 style='text-align: center;'>Flux.1 Image Upscaler</h1>")
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input Parameters Accordion
                        with gr.Accordion("Input Parameters", open=True):
                            control_image_selector = gr.Image(
                                label="Control Image",
                                type="pil",
                            )

                        # Output Settings Accordion
                        with gr.Accordion("Output Settings", open=False):
                            scale_factor_slider = gr.Slider(
                                label="Scale Factor",
                                minimum=1,
                                maximum=8,
                                value=2,
                                step=0.5,
                            )

                        # Pipeline Settings Accordion
                        with gr.Accordion("Pipeline Settings", open=False):
                            cpu_offload_checkbox = gr.Checkbox(
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

                # Button click handlers
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

            self.logger.info("Gradio interface setup complete.")
            return interface
        except Exception as e:
            self.logger.error("Error occurred while setting up the interface: %s", str(e))
            raise
