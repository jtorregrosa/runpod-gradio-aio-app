import os
import logging
import pathlib
from typing import List, Tuple, Optional
from PIL import Image
from UIFlux.fake_image_generator import FakeImageGenerator
import gradio as gr
from UIBase.ui_base import UIBase
from Utils.utils import load_yaml, zip_images, generate_random_prompt
from .flux_image_generator import FluxImageGenerator
from .sd3_image_generator import SD3ImageGenerator

logger = logging.getLogger(__name__)

class UIFlux(UIBase):
    """
    UIFlux provides a Gradio interface for generating images using LoRA weights and other settings.
    """

    def __init__(self):
        super().__init__()
        self.models: List[dict] = []
        self.available_models: List[Tuple[str,str]] = []
        self.available_loras: List[Tuple[str,str]] = []
        self.defaults: dict = {}
        self.generator = None
        self.logger = logger
        self.images_list = []
        
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
            self.models = config.get('models', [])
            self.defaults = config.get('defaults', {}).get('settings', {})
            self.available_models = [[item.get('name'), item.get('id')] for item in self.models]
            self.available_loras = [["None", ""]] + [[item.get('name'), item.get('id')] for item in self.models[0].get('loras', [])]
            self.logger.info("Configuration loaded successfully from '%s'.", yaml_file_path)
        except Exception as e:
            self.logger.error("Failed to load configuration: %s", e, exc_info=True)
            raise gr.Error("Failed to initialize the UI configuration.")
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
            self.logger.error("Prompt input cannot be empty.")
            raise gr.Error("Prompt input cannot be empty.")
        processed_prompts = [line.strip() for line in prompt.split('\n') if line.strip()]
        self.logger.info("Processed %d prompts.", len(processed_prompts))
        return processed_prompts

    def submit(
        self,
        image_state,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        lora_weights_1: Optional[str],
        lora_adapter_1_weight: float,
        lora_weights_2: Optional[str],
        lora_adapter_2_weight: float,
        width: int,
        height: int,
        images_per_prompt: int,
        model_cpu_offload: bool,
        sequential_cpu_offload: bool,
        vae_slicing: bool,
        vae_tiling: bool,
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
            model = next((item for item in self.models if item.get('id') == model_id), None)
            lora1 = next((item for item in model.get('loras', []) if item.get('id') == lora_weights_1), None)
            lora2 = next((item for item in model.get('loras', []) if item.get('id') == lora_weights_2), None)
            
            match model.get('type'):
                case "sd3":
                    self.generator = SD3ImageGenerator()
                case "flux":
                    self.generator = FluxImageGenerator()
                case "dummy":
                    self.generator = FakeImageGenerator()

            # Initialize the generator
            self.generator.initialize(
                model_id=model_id,
                lora_weight_1_id=lora1.get('repo') if lora1 else None,
                lora_weight_1_name=lora1.get('weight_name') if lora1 else None,
                lora_adapter_1_weight=lora_adapter_1_weight,
                lora_weight_2_id=lora2.get('repo') if lora2 else None,
                lora_weight_2_name=lora2.get('weight_name') if lora2 else None,
                lora_adapter_2_weight=lora_adapter_1_weight,
                lora_scale=lora_scale,
                model_cpu_offload=model_cpu_offload,
                sequential_cpu_offload=sequential_cpu_offload,
                vae_slicing=vae_slicing,
                vae_tiling=vae_tiling,
            )

            self.logger.info("Generator initialized")

            # Process prompt
            prompt_list = self.process_prompt(prompt)

            # Add specific LoRA trigger to the prompts if applicable
            if lora1 and lora1.get('trigger'):
                trigger = lora1.get('trigger', '')
                prompt_list = [f"{trigger} {item}".strip() for item in prompt_list]
                self.logger.info("Applied LoRA 1 trigger to prompts.")

            if lora2 and lora2.get('trigger'):
                trigger = lora2.get('trigger', '')
                prompt_list = [f"{trigger} {item}".strip() for item in prompt_list]
                self.logger.info("Applied LoRA 2 trigger to prompts.")

            # Set up progress callback
            progress = gr.Progress()
            progress(0, "Preparing...")
            def gradio_callback(idx: int, step: int, num_prompts: int, total_steps: int):
                current_progress = ((idx * total_steps) + step + 1) / (num_prompts * total_steps)
                progress(current_progress, f"[Prompt {idx+1}] Step {step+1}/{total_steps}")
                self.logger.debug("Progress updated: %0.2f%%", current_progress * 100)

            match model.get('type'):
                case "sd3":
                    for images in self.generator.generate(
                        prompt_list=prompt_list,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        images_per_prompt=images_per_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        callback=gradio_callback
                    ):
                        yield images, images
                case "flux":
                    for images in self.generator.generate(
                        prompt_list=prompt_list,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        images_per_prompt=images_per_prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        callback=gradio_callback
                    ):
                        yield images, images
                case "dummy":
                    for images in self.generator.generate(
                        prompt_list=prompt_list,
                        width=width,
                        height=height,
                        images_per_prompt=images_per_prompt,
                        num_inference_steps=num_inference_steps,
                        callback=gradio_callback
                    ):
                        yield images, images
            
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
            None,
            self.defaults.get('output', {}).get('width', 512),
            self.defaults.get('output', {}).get('height', 512),
            self.defaults.get('output', {}).get('images_per_prompt', 1),
            self.defaults.get('memory', {}).get('model_cpu_offload', False),
            self.defaults.get('memory', {}).get('sequential_cpu_offload', False),
            self.defaults.get('memory', {}).get('vae_slicing', False),
            self.defaults.get('memory', {}).get('vae_tiling', False),
            self.defaults.get('pipeline', {}).get('lora_scale', 1.0),
            self.defaults.get('pipeline', {}).get('guidance_scale', 7.5),
            self.defaults.get('pipeline', {}).get('num_inference_steps', 50),
            self.defaults.get('pipeline', {}).get('max_sequence_length', 512),
        )
    
    def on_model_change(self, model_id: str, true_cfg: bool):
        model = next((item for item in self.models if item.get('id') == model_id), None)
        overrides = next((item for item in self.models if item.get('id') == model_id), None).get('overrides')

        model_selector = gr.Dropdown(
            info=model.get('description', None),
        )
        
        negative_prompt_input = gr.Textbox(
            visible=True if model.get('type') == 'sd3' or (model.get('type') == 'flux' and true_cfg) else False
        )
        
        true_cfg_checkbox = gr.Checkbox(
            visible=True if model.get('type') == 'flux' else False
        )
        
        if model.get('loras',[]) == []:
            lora_weights_selector = gr.Dropdown(
                choices=self.available_loras,
                value=None,
                visible=False,
            )
            lora_scale_slider = gr.Slider(visible=False)
        else:
            available_lora_ids = [["None", ""]] + [[item.get('name', ''), item.get('id', '')] for item in model.get('loras',[])]
            lora_weights_selector = gr.Dropdown(
                choices=available_lora_ids,
                value=available_lora_ids[0][1],
                visible=True,
            )
            lora_scale_slider = gr.Slider(
                value=overrides['pipeline']['lora_scale'],
                visible=True,
            )
            
        return [
            true_cfg_checkbox,
            negative_prompt_input,
            model_selector,
            lora_weights_selector,
            lora_weights_selector,
            lora_scale_slider,
            overrides['pipeline']['guidance_scale'], 
            overrides['pipeline']['num_inference_steps'], 
            overrides['pipeline']['max_sequence_length'],
        ]

    def on_lora_weights_selector_change(self, model_id, lora_weights_selector):
        model = next((item for item in self.models if item.get('id') == model_id), None)
        lora = next((item for item in model.get('loras', []) if item.get('id') == lora_weights_selector), None)
        description = None
        
        if lora is not None:
            description = lora['description']

        return gr.Dropdown(info=description)
    
    def on_true_cfg_checkbox_change(self, model_id, true_cfg):
        model = next((item for item in self.models if item.get('id') == model_id), None)
        
        return gr.Textbox(
            visible=True if (model.get('type') == 'flux' and true_cfg) else False
        )

    def generate_random_prompt(self, prompt):
        value = ""
        if prompt:
            value=f"{prompt}\n{generate_random_prompt()}"
        else:
            value=generate_random_prompt()
            
        return gr.Textbox(
            value=value,
        )
        
        
    def interface(self) -> gr.Blocks:
        """
        Constructs the Gradio interface for the Flux Image Generator.

        Returns:
            gr.Blocks: The constructed Gradio Blocks interface object.
        """
        self.logger.info("Constructing Gradio interface.")
        with gr.Blocks() as interface:
            image_state = gr.State([])
            files_state = gr.State([])
            
            gr.HTML('<h1 style="text-align: center; margin:1rem;">Text2Image Lab</h1>')
            gr.HTML('<p style="text-align: center; margin:1rem;">A Gradio-based application for bulk image generation, allowing users to create high-quality images using various models and LoRAs. Highly customizable, it supports multiple model combinations, offers fine-tuned control over generation parameters, and features an intuitive interface for efficient workflow. Ideal for artists, designers, and researchers seeking versatile and efficient image creation.</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        with gr.Column():
                            prompt_input = gr.Textbox(
                                label="✍🏼 Prompt",
                                info="Enter your prompt to create an image based on your description. ONE PROMPT PER LINE",
                                lines=5,
                                placeholder="Enter one prompt per line",
                            )
                            
                            random_prompt_btn = gr.Button(
                                value="",
                                variant="secondary",
                                size="lg",
                                icon=f"{pathlib.Path(__file__).parent.resolve()}/dice.png",
                            )
                        negative_prompt_input = gr.Textbox(
                            label="❌ Negative prompt",
                            info="Enter a negative prompt to exclude certain elements from the generated image. NEGATIVE PROMPT WILL AFFECT ALL PROMPTS",
                            lines=3,
                            placeholder="Enter global negative prompt",
                            value="(lowres, low quality, worst quality)",
                            visible=False,
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

                    with gr.Accordion("⚙️ Model Settings", open=False):
                        model_selector = gr.Dropdown(
                            choices=self.available_models,
                            label="Base Model",
                            info=self.models[0].get('description', ''),
                            value=self.available_models[0][1],
                        )
                        with gr.Row():
                            lora_weights_1_selector = gr.Dropdown(
                                choices=self.available_loras,
                                label="LoRA 1",
                                visible=True,
                                value=self.available_loras[0][1],
                            )
                            lora_weights_adapter_1_slider = gr.Slider(
                                label="LoRA 1 Adapter Weight",
                                info="The weight which will be multiplied by each adapter's output before summing them together.",
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.1,
                                visible=True,
                            )
                        with gr.Row():
                            lora_weights_2_selector = gr.Dropdown(
                                choices=self.available_loras,
                                label="LoRA 2",
                                visible=True,
                                value=self.available_loras[0][1],
                            )
                            lora_weights_adapter_2_slider = gr.Slider(
                                label="LoRA 2 Adapter Weight",
                                info="The weight which will be multiplied by each adapter's output before summing them together.",
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.1,
                                visible=True,
                            )

                    with gr.Accordion("🖼️ Output Settings", open=False):
                        with gr.Row():
                            width_input = gr.Number(
                                label="Width",
                                minimum=256,
                                maximum=2048,
                                value=self.defaults['output']['width'],
                            )
                            height_input = gr.Number(
                                label="Height",
                                minimum=256,
                                maximum=2048,
                                value=self.defaults['output']['height'],
                            )
                            images_per_prompt_input = gr.Number(
                                label="Images Per Prompt",
                                minimum=1,
                                value=self.defaults['output']['images_per_prompt'],
                            )

                    with gr.Accordion("⚗️ Pipeline Settings", open=False):
                        true_cfg_checkbox = gr.Checkbox(
                            label="Use True CFG",
                            info="FLUX.1-dev is a guidance distill model. The original CFG process, which required twice the number of inference steps, is distilled into a guidance scale, thereby modulating the DIT through the guidance scale to simulate the true CFG process with half the inference steps.",
                            value=False,
                            visible=True,
                        )
                        lora_scale_slider = gr.Slider(
                            label="LoRA Scale",
                            info="A scaling factor for the LoRA weights to control their influence on the model. The default value is 1.0, indicating full influence. Lower values decrease the impact of the LoRA weights.",
                            minimum=0.0,
                            maximum=3,
                            value=self.defaults['pipeline']['lora_scale'],
                            step=0.1,
                            visible=True,
                        )
                        guidance_scale_slider = gr.Slider(
                            label="Guidance Scale",
                            info="A scale factor for classifier-free guidance, which controls how much the model should adhere to the given prompt. Higher values lead to images more closely aligned with the prompt.",
                            minimum=0.0,
                            maximum=20.0,
                            value=self.defaults['pipeline']['guidance_scale'],
                            step=0.1,
                        )
                        num_inference_steps_slider = gr.Slider(
                            label="Number of Inference Steps",
                            info="The number of inference steps used for the generation process. More steps typically result in higher quality images, but also increase the computation time.",
                            minimum=1,
                            maximum=100,
                            value=self.defaults['pipeline']['num_inference_steps'],
                            step=1,
                        )
                        max_sequence_length_slider = gr.Slider(
                            label="Max Sequence Length",
                            info="The maximum sequence length for processing each prompt. This controls how much of the prompt text the model can process.",
                            minimum=1,
                            maximum=1024,
                            value=self.defaults['pipeline']['max_sequence_length'],
                            step=1,
                        )

                    with gr.Accordion("🪶 Memory Saving Settings", open=False):
                        model_cpu_offload_checkbox = gr.Checkbox(
                            label="Model CPU Offload",
                            info="Offloads the model to the CPU to manage memory efficiently. This can be useful when GPU memory is limited, as portions of the model are offloaded to the CPU during the inference process.",
                            value=self.defaults['memory']['model_cpu_offload'],
                        )
                        sequential_cpu_offload_checkbox = gr.Checkbox(
                            label="Sequential CPU Offload",
                            info="Offloads model layers one at a time during inference. This helps in managing memory consumption when GPU resources are constrained",
                            value=self.defaults['memory']['sequential_cpu_offload'],
                        )
                        vae_slicing_checkbox = gr.Checkbox(
                            label="VAE Slicing",
                            info="VAE slicing reduces the memory required during image processing by slicing the VAE operations into smaller chunks.",
                            value=self.defaults['memory']['vae_slicing'],
                        )
                        vae_tiling_checkbox = gr.Checkbox(
                            label="VAE Tiling",
                            info="Allows processing of large images by dividing them into smaller tiles. This can be useful for handling high-resolution inputs when memory is a concern.",
                            value=self.defaults['memory']['vae_tiling'],
                        ) 

                with gr.Column(scale=1):
                    image_gallery = gr.Gallery(
                        label="Generated Images",
                        format="png",
                        columns=3,
                        show_download_button=True,
                        show_label=True,
                        show_fullscreen_button=True,
                        value=[],
                    )
                            # Download button
                    def download_images(images, files_state):
                        zip_buffer = zip_images(images)
                        files_state.insert(0, zip_buffer)
                        
                        return files_state, files_state

                    download_btn = gr.Button(
                        value="Generate ZIP Bundle",
                        variant="primary",
                    )
                    download_file = gr.File(
                        label="Generated ZIP Files",
                        file_count="multiple",
                        height=150,
                    )
                    download_btn.click(
                        fn=download_images, 
                        inputs=[
                            image_state,
                            files_state,
                        ], 
                        outputs=[
                            download_file,
                            files_state,
                        ]
                    )

            random_prompt_btn.click(
                fn=self.generate_random_prompt,
                inputs= [
                    prompt_input,
                ],
                outputs=[
                    prompt_input,
                ],
            )
                    
            model_selector.change(
                fn=self.on_model_change,
                inputs= [
                    model_selector,
                    true_cfg_checkbox,
                ],
                outputs=[
                    true_cfg_checkbox,
                    negative_prompt_input,
                    model_selector,
                    lora_weights_1_selector,
                    lora_weights_adapter_1_slider,
                    lora_weights_2_selector,
                    lora_weights_adapter_2_slider,
                    lora_scale_slider,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    max_sequence_length_slider
                ],
            )
            
            true_cfg_checkbox.change(
                fn=self.on_true_cfg_checkbox_change,
                inputs= [
                    model_selector,
                    true_cfg_checkbox,
                ],
                outputs=[
                    negative_prompt_input,
                ],
            )
            
            lora_weights_1_selector.change(
                fn=self.on_lora_weights_selector_change,
                inputs= [
                    model_selector,
                    lora_weights_1_selector,
                ],
                outputs=[
                    lora_weights_1_selector,
                ],
            )

            lora_weights_2_selector.change(
                fn=self.on_lora_weights_selector_change,
                inputs= [
                    model_selector,
                    lora_weights_2_selector,
                ],
                outputs=[
                    lora_weights_2_selector,
                ],
            )

            generate_btn.click(
                fn=self.submit,
                inputs=[
                    image_state,
                    prompt_input,
                    negative_prompt_input,
                    model_selector,
                    lora_weights_1_selector,
                    lora_weights_adapter_1_slider,
                    lora_weights_2_selector,
                    lora_weights_adapter_2_slider,
                    width_input,
                    height_input,
                    images_per_prompt_input,
                    model_cpu_offload_checkbox,
                    sequential_cpu_offload_checkbox,
                    vae_slicing_checkbox,
                    vae_tiling_checkbox,
                    lora_scale_slider,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    max_sequence_length_slider,
                ],
                outputs=[image_gallery, image_state],
                queue=True,
                show_progress="minimal"
            )

            clear_btn.click(
                fn=self.clear,
                inputs=[],
                outputs=[
                    prompt_input,
                    model_selector,
                    lora_weights_1_selector,
                    lora_weights_2_selector,
                    width_input,
                    height_input,
                    images_per_prompt_input,
                    model_cpu_offload_checkbox,
                    sequential_cpu_offload_checkbox,
                    vae_slicing_checkbox,
                    vae_tiling_checkbox,
                    lora_scale_slider,
                    guidance_scale_slider,
                    num_inference_steps_slider,
                    max_sequence_length_slider,
                ]
            )

        return interface
