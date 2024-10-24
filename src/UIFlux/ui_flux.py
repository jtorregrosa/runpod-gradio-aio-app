import os
import logging
from typing import List, Tuple, Optional

import gradio as gr
from UIBase.ui_base import UIBase
from Utils.utils import load_yaml
from .flux_image_generator import FluxImageGenerator

logger = logging.getLogger(__name__)

class UIFlux(UIBase):
    """
    UIFlux provides a Gradio interface for generating images using LoRA weights and other settings.
    """

    def __init__(self):
        super().__init__()
        self.loras: List[dict] = []
        self.available_lora_ids: List[str] = []
        self.defaults: dict = {}
        self.generator = FluxImageGenerator()
        self.logger = logger
        self.logger.info("UIFlux instance initialized.")

    def initialize(self) -> UIBase:
        """
        Initializes the UI configuration from a YAML file.

        Returns:
            UIBase: The initialized UI object.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_file_path = os.path.join(current_dir, 'config.yaml')
            config = load_yaml(yaml_file_path)

            # Load configuration
            self.loras = config.get('loras', [])
            self.defaults = config.get('defaults', {}).get('settings', {})
            self.available_lora_ids = [""] + [item.get('id', '') for item in self.loras]

            self.logger.info("Configuration loaded successfully from '%s'.", yaml_file_path)
        except Exception as e:
            self.logger.error("Failed to load configuration: %s", e, exc_info=True)
            raise RuntimeError("Failed to initialize the UI configuration.")
        return self

    def process_prompt(self, prompt: str) -> List[str]:
        """
        Processes the prompt input by splitting and stripping each line.

        Parameters:
            prompt (str): The raw prompt input from the user.

        Returns:
            List[str]: A list of cleaned prompts.
        
        Raises:
            ValueError: If the prompt input is empty.
        """
        if not prompt.strip():
            self.logger.error("Prompts input cannot be empty.")
            raise ValueError("Prompts input cannot be empty.")
        processed_prompts = [line.strip() for line in prompt.split('\n') if line.strip()]
        self.logger.info("Processed %d prompts.", len(processed_prompts))
        return processed_prompts

    def submit(
        self,
        prompt: str,
        model_id: str,
        lora_weights: Optional[str],
        width: int,
        height: int,
        images_per_prompt: int,
        cpu_offload: bool,
        lora_scale: float,
        guidance_scale: float,
        num_inference_steps: int,
        max_sequence_length: int,
    ):
        """
        Handles the submission to generate images based on the provided parameters.

        Parameters:
            Various UI input values to generate images.

        Yields:
            List[Tuple[Image.Image, str]]: Generated images with descriptions.
        """
        try:
            self.logger.info("Submission started for model: %s with prompt: %s", model_id, prompt[:50])

            # Find selected LoRA weights
            lora = next((item for item in self.loras if item.get('id') == lora_weights), None)

            # Initialize the generator
            self.generator.initialize(
                model_id=model_id,
                lora_weights_id=lora.get('id') if lora else None,
                lora_weight_name=lora.get('weight_name') if lora else None,
                lora_scale=lora_scale,
                cpu_offload=cpu_offload
            )

            self.logger.info("Generator initialized with model: %s and LoRA weights: %s", model_id, lora_weights)

            # Process prompt
            prompt_list = self.process_prompt(prompt)

            # Add specific LoRA trigger to the prompts if applicable
            if lora and lora.get('trigger'):
                trigger = lora.get('trigger', '')
                prompt_list = [f"{trigger} {item}".strip() for item in prompt_list]
                self.logger.info("Applied LoRA trigger to prompts.")

            # Set up progress callback
            progress = gr.Progress()
            progress(0, "Preparing...")
            def gradio_callback(idx: int, step: int, num_prompts: int, total_steps: int):
                current_progress = ((idx * total_steps) + step + 1) / (num_prompts * total_steps)
                progress(current_progress, f"[Prompt {idx+1}] Step {step+1}/{total_steps}")
                self.logger.debug("Progress updated: %0.2f%%", current_progress * 100)

            # Generate images and yield results to update the UI
            for images in self.generator.generate(
                prompt_list=prompt_list,
                width=width,
                height=height,
                images_per_prompt=images_per_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                callback=gradio_callback
            ):
                self.logger.info("Yielding generated images for current batch.")
                yield images

        except Exception as e:
            self.logger.error("An error occurred during image generation: %s", e, exc_info=True)
            raise RuntimeError(f"Image generation failed: {e}")

    def clear(self) -> Tuple:
        """
        Resets all input fields to their default values.

        Returns:
            Tuple: Default values for all input components.
        """
        self.logger.info("Clearing all input fields to default values.")
        return (
            "",  # Reset prompt input
            "black-forest-labs/FLUX.1-dev",  # Reset model selector to default value
            None,  # Reset LORA weights to None
            self.defaults.get('output', {}).get('width', 512),  # Width default
            self.defaults.get('output', {}).get('height', 512),  # Height default
            self.defaults.get('output', {}).get('images_per_prompt', 1),  # Images per prompt
            self.defaults.get('pipeline', {}).get('cpu_offload', False),  # CPU offload
            self.defaults.get('pipeline', {}).get('lora_scale', 1.0),  # LORA scale
            self.defaults.get('pipeline', {}).get('guidance_scale', 7.5),  # Guidance scale
            self.defaults.get('pipeline', {}).get('num_inference_steps', 50),  # Inference steps
            self.defaults.get('pipeline', {}).get('max_sequence_length', 512),  # Max sequence length
        )

    def interface(self) -> gr.Blocks:
        """
        Constructs the Gradio interface for the Flux Image Generator.

        Returns:
            gr.Blocks: The constructed Gradio Blocks interface object.
        """
        self.logger.info("Constructing Gradio interface.")
        with gr.Blocks() as interface:
            gr.Markdown("<h1 style='text-align: center;'>Flux.1 LORA Image Generator</h1>")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Input Parameters", open=True):
                        prompt_input = gr.Textbox(
                            label="Enter your prompts (one per line)",
                            lines=5,
                            placeholder="Enter one prompt per line",
                        )

                    with gr.Accordion("Model Settings", open=False):
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

                    with gr.Accordion("Output Settings", open=False):
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

                    with gr.Accordion("Pipeline Settings", open=False):
                        cpu_offload_checkbox = gr.Checkbox(
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

            # Button click actions for submission and clearing
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
                outputs=[image_gallery],
                queue=True,
                show_progress="minimal"
            )

            clear_btn.click(
                fn=self.clear,
                inputs=[],
                outputs=[
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
                ]
            )

        return interface
