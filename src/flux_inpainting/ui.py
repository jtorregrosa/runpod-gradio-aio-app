import os
import logging
from typing import List, Tuple, Optional
from PIL import Image
from UIFlux.fake_image_generator import FakeImageGenerator
import gradio as gr
from UIBase.ui_base import UIBase
from Utils.utils import load_yaml, zip_images
from .flux_image_inpainting_generator import FluxImageInpaintingGenerator

logger = logging.getLogger(__name__)

class UIFluxInpainting(UIBase):
    def __init__(self):
        super().__init__()
        self.logger = logger
        self.logger.info("UIFlux Inpainting instance initialized.")

    def initialize(self) -> UIBase:
        return self

    def submit(
        self,
        input_image_editor,
        prompt: str,
        negative_prompt: str,
        model_cpu_offload: bool,
        sequential_cpu_offload: bool,
        vae_slicing: bool,
        vae_tiling: bool,
        guidance_scale: float,
        num_inference_steps: int,
        controlnet_conditioning_scale: float,
    ):
        try:
            self.generator = FluxImageInpaintingGenerator()

            self.generator.initialize(
                model_cpu_offload=model_cpu_offload,
                sequential_cpu_offload=sequential_cpu_offload,
                vae_slicing=vae_slicing,
                vae_tiling=vae_tiling,
            )

            progress = gr.Progress()
            progress(0, "Preparing...")
            def gradio_callback(step, total_steps):
                current_progress = (step + 1) / total_steps
                progress(current_progress, f"Step {step+1}/{total_steps}")

            control_image = input_image_editor['background']
            control_mask = input_image_editor['layers'][0]
            
            result = self.generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                control_mask=control_mask,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                callback=gradio_callback
            )
            
            return result, "Result"
                        
        except Exception as e:
            self.logger.error("An error occurred during image generation: %s", e, exc_info=True)
            raise RuntimeError(f"Image generation failed: {e}")

    def clear(self) -> Tuple:
        self.logger.info("Clearing all input fields to default values.")
        return (
            None, # input_image_editor_component,
            None, # prompt_input,
            "(lowres, low quality, worst quality)", # negative_prompt_input,
            False, # model_cpu_offload_checkbox,
            True, # sequential_cpu_offload_checkbox,
            False, # vae_slicing_checkbox,
            False, # vae_tiling_checkbox,
            3.5, # guidance_scale_slider,
            24, # num_inference_steps_slider,
            0.9, # controlnet_conditioning_scale_slider,
            None, # output_image_component,
        )
     
    def interface(self) -> gr.Blocks:
        self.logger.info("Constructing Gradio interface.")
        
        with gr.Blocks() as interface:
            gr.HTML('<h1 style="text-align: center; margin:1rem;">Image Inpainting Lab</h1>')
            gr.HTML('<p style="text-align: center; margin:1rem;"A Gradio-based application for single-image inpainting using the Flux.1 model, designed to produce high-quality, detailed images. It features an intuitive interface with precise control over inpainting parameters, providing a seamless editing experience</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        input_image_editor_component = gr.ImageEditor(
                            label='Image',
                            type='pil',
                            sources=["upload", "webcam"],
                            image_mode='RGB',
                            layers=False,
                            brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed")
                        )
                        
                        
                        prompt_input = gr.Textbox(
                            label="✍🏼 Prompt",
                            info="Enter your prompt to be applied to the image mask",
                            lines=5,
                            placeholder="Enter one prompt per line",
                        )
                        negative_prompt_input = gr.Textbox(
                            label="❌ Negative prompt",
                            info="Enter your negative prompt to be applied to the image mask",
                            lines=3,
                            placeholder="Enter global negative prompt",
                            value="(lowres, low quality, worst quality)",
                        )
                        
                        with gr.Row():
                            clear_btn = gr.Button(
                                value="Clear",
                                variant="secondary",
                                scale=1,
                            )
                            generate_btn = gr.Button(
                                value="Submit",
                                variant="primary",
                                scale=2,
                            )

                    with gr.Accordion("⚗️ Pipeline Settings", open=False):
                        guidance_scale_slider = gr.Slider(
                            label="Guidance Scale",
                            info="A scale factor for classifier-free guidance, which controls how much the model should adhere to the given prompt. Higher values lead to images more closely aligned with the prompt usually at the expense of lower image quality.",
                            minimum=0.0,
                            maximum=20.0,
                            value=3.5,
                            step=0.5,
                        )
                        num_inference_steps_slider = gr.Slider(
                            label="Number of Inference Steps",
                            info="The number of inference steps used for the generation process. More steps typically result in higher quality images, but also increase the computation time.",
                            minimum=1,
                            maximum=100,
                            value=24,
                            step=1,
                        )
                        controlnet_conditioning_scale_slider = gr.Slider(
                            label="ControlNet Conditioning Scale",
                            info="The outputs of the controlnet are multiplied by controlnet_conditioning_scale before they are added to the residual in the original unet.",
                            minimum=0,
                            maximum=1,
                            value=0.9,
                            step=0.01,
                        )

                    with gr.Accordion("🪶 Memory Saving Settings", open=False):
                        model_cpu_offload_checkbox = gr.Checkbox(
                            label="Model CPU Offload",
                            info="Offloads the model to the CPU to manage memory efficiently. This can be useful when GPU memory is limited, as portions of the model are offloaded to the CPU during the inference process.",
                            value=False,
                        )
                        sequential_cpu_offload_checkbox = gr.Checkbox(
                            label="Sequential CPU Offload",
                            info="Offloads model layers one at a time during inference. This helps in managing memory consumption when GPU resources are constrained",
                            value=False,
                        )
                        vae_slicing_checkbox = gr.Checkbox(
                            label="VAE Slicing",
                            info="VAE slicing reduces the memory required during image processing by slicing the VAE operations into smaller chunks.",
                            value=False,
                        )
                        vae_tiling_checkbox = gr.Checkbox(
                            label="VAE Tiling",
                            info="Allows processing of large images by dividing them into smaller tiles. This can be useful for handling high-resolution inputs when memory is a concern.",
                            value=False,
                        ) 

                with gr.Column(scale=1):
                    output_image_component = gr.Image(
                        type='pil',
                        image_mode='RGB',
                        label='Generated image',
                        format="png",
                    )
            
            def on_image_upload():
                return gr.ImageEditor(
                    height=None
                )
            
            input_image_editor_component.upload(
                fn=on_image_upload,
                inputs=[],
                outputs=[
                    input_image_editor_component,
                ],
            )
                    
            generate_btn.click(
                fn=self.submit,
                inputs=[
                    input_image_editor_component,
                    prompt_input,
                    negative_prompt_input,
                    model_cpu_offload_checkbox,
                    sequential_cpu_offload_checkbox,
                    vae_slicing_checkbox,
                    vae_tiling_checkbox,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    controlnet_conditioning_scale_slider,
                ],
                outputs=[
                    output_image_component
                ],
                queue=True,
                show_progress="minimal"
            )

            clear_btn.click(
                fn=self.clear,
                inputs=[],
                outputs=[
                    input_image_editor_component,
                    prompt_input,
                    negative_prompt_input,
                    model_cpu_offload_checkbox,
                    sequential_cpu_offload_checkbox,
                    vae_slicing_checkbox,
                    vae_tiling_checkbox,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    controlnet_conditioning_scale_slider,
                    output_image_component,
                ]
            )

        return interface
