from diffusers import StableDiffusion3Pipeline
from typing import List, Optional, Callable, Generator, Tuple
import logging
import time
import torch
from PIL import Image
import hashlib

# Configure a logger for this module
logger = logging.getLogger(__name__)

class SD3ImageGenerator:
    """
    A class to initialize and generate images using the StableDiffusion3Pipeline.
    """

    def __init__(self):
        self.pipe = None
        self.previous_hash = None

    def initialize(
        self,
        model_id: str,
        lora_weights_id: Optional[str] = None,
        lora_weight_name: Optional[str] = None,
        lora_scale: float = 1.0,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        vae_slicing: bool = False,
        vae_tiling: bool = False,
    ):
        """
        Initializes the FluxPipeline with the given parameters.

        Parameters:
            model_id (str):
                The identifier for the pre-trained model to be loaded. This can be a local path or a repository name.

            lora_weights_id (Optional[str]):
                The identifier for loading LoRA (Low-Rank Adaptation) weights, if applicable. This can be used to fine-tune
                or adapt the model to specific tasks. If not provided, no LoRA weights will be loaded.

            lora_weight_name (Optional[str]):
                The specific name for the LoRA weights to be used. This allows more granular control over which LoRA
                weights are applied when multiple sets are available under the same `lora_weights_id`.

            lora_scale (float):
                A scaling factor for the LoRA weights to control their influence on the model. The default value is 1.0,
                indicating full influence. Lower values decrease the impact of the LoRA weights.

            model_cpu_offload (bool):
                Whether to enable offloading the model to the CPU to manage memory efficiently. This can be useful when
                GPU memory is limited, as portions of the model are offloaded to the CPU during the inference process.
                Default is `False`.

            sequential_cpu_offload (bool):
                If set to `True`, enables sequential CPU offloading, which offloads model layers one at a time during
                inference. This helps in managing memory consumption when GPU resources are constrained.
                Default is `False`.

            vae_slicing (bool):
                If `True`, enables slicing for the Variational Autoencoder (VAE) used in the model. VAE slicing reduces
                the memory required during image processing by slicing the VAE operations into smaller chunks.
                Default is `False`.

            vae_tiling (bool):
                If `True`, enables tiling for the VAE operations, which allows processing of large images by dividing
                them into smaller tiles. This can be useful for handling high-resolution inputs when memory is a concern.
                Default is `False`.

        Raises:
            RuntimeError: If loading the model or LoRA weights fails.
        """
        # Create a hash to avoid reinitializing with the same parameters
        param_string = f"{model_id}-{lora_weights_id or 'None'}-{lora_weight_name or 'None'}-{lora_scale}-{model_cpu_offload}-{sequential_cpu_offload}-{vae_slicing}-{vae_tiling}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        if self.previous_hash == current_hash:
            logger.info("Using existing pipeline instance as parameters are unchanged.")
            return

        logger.info("Initializing new pipeline with model_id: %s", model_id)
        self.previous_hash = current_hash

        try:
            # Initialize the pipeline from the model
            self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
            logger.info("Successfully loaded model '%s'", model_id)
        except Exception as e:
            logger.error("Failed to load the model '%s': %s", model_id, e, exc_info=True)
            raise RuntimeError(f"Failed to load the model '{model_id}': {e}")

        if lora_weights_id:
            self._load_and_fuse_lora_weights(lora_weights_id, lora_weight_name, lora_scale)

        if model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        
        if sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            
        if vae_slicing:
            self.pipe.enable_vae_slicing()
        
        if vae_tiling:
            self.pipe.enable_vae_tiling()

        if not model_cpu_offload and not sequential_cpu_offload and not vae_slicing and not vae_tiling:
            self.pipe = self.pipe.to("cuda")

    def _load_and_fuse_lora_weights(self, lora_weights_id: str, lora_weight_name: Optional[str], lora_scale: float):
        """
        Loads and fuses the LoRA weights into the model.

        Parameters:
            lora_weights_id (str): The ID for loading LoRA weights.
            lora_weight_name (Optional[str]): Specific name for LoRA weights to be used.
            lora_scale (float): Scale factor for LoRA weights.

        Raises:
            RuntimeError: If loading or fusing LoRA weights fails.
        """
        try:
            self.pipe.load_lora_weights(lora_weights_id, weight_name=lora_weight_name)
            self.pipe.fuse_lora(lora_scale=lora_scale)
            logger.info("Successfully loaded and fused LoRA weights '%s'", lora_weights_id)
        except Exception as e:
            logger.error("Failed to load or fuse LoRA weights '%s': %s", lora_weights_id, e, exc_info=True)
            raise RuntimeError(f"Failed to load or fuse LoRA weights: {e}")

    def _enable_cpu_offloading(self):
        """
        Enables CPU offloading for the pipeline.
        """
        try:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_sequential_cpu_offload()
            logger.info("Enabled CPU offload for the pipeline.")
        except Exception as e:
            logger.error("Failed to enable CPU offloading: %s", e, exc_info=True)
            raise RuntimeError("Failed to enable CPU offloading.")

    def generate(
            self,
            prompt_list: List[str],
            negative_prompt: str,
            width: int,
            height: int,
            images_per_prompt: int,
            guidance_scale: float,
            num_inference_steps: int,
            max_sequence_length: int,
            callback: Optional[Callable[[int, int, int, int], None]] = None
    ) -> Generator[List[Tuple[Image.Image, str]], None, None]:
        """
        Generates images based on prompts using the initialized pipeline.

        Parameters:
            prompt_list (List[str]):
                List of prompts for image generation. Each prompt represents a unique idea or scene to be generated.

            width (int):
                The width of the generated images in pixels.

            height (int):
                The height of the generated images in pixels.

            images_per_prompt (int):
                The number of images to generate per prompt provided in `prompt_list`.

            guidance_scale (float):
                A scale factor for classifier-free guidance, which controls how much the model should adhere to the given prompt.
                Higher values lead to images more closely aligned with the prompt.

            num_inference_steps (int):
                The number of inference steps used for the generation process. More steps typically result in higher quality images,
                but also increase the computation time.

            max_sequence_length (int):
                The maximum sequence length for processing each prompt. This controls how much of the prompt text the model can process.

            callback (Optional[Callable[[int, int, int, int], None]]):
                An optional callback function for tracking the progress of the image generation. The function is called with
                parameters (current_prompt_index, current_step, total_prompts, total_steps) to provide real-time updates.

        Yields:
            List[Tuple[Image.Image, str]]: A list of tuples containing generated images and their captions.
        """
        if not self.pipe:
            raise RuntimeError("Pipeline is not initialized. Please call 'initialize()' first.")

        images = []
        random_seed = int(time.time())
        generator = torch.Generator("cpu").manual_seed(random_seed)

        logger.info("Starting image generation for %d prompts", len(prompt_list))

        for idx, prompt in enumerate(prompt_list):
            logger.debug("Generating images for prompt %d: '%s'", idx + 1, prompt)

            def flux_callback(pipe, step_index, timestep, callback_kwargs):
                if callback:
                    callback(idx, step_index, len(prompt_list), num_inference_steps)
                return callback_kwargs

            try:
                # Generate images for the current prompt
                prompt_images = self.pipe(
                    prompt,
                    negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    generator=generator,
                    callback_on_step_end=flux_callback,
                    num_images_per_prompt=images_per_prompt
                ).images

                for img_idx, prompt_image in enumerate(prompt_images):
                    images.append((prompt_image, f"Image {idx + 1}-{img_idx + 1}"))
                    logger.info("Generated image %d for prompt %d", img_idx + 1, idx + 1)

            except Exception as e:
                logger.error("Failed to generate images for prompt '%s': %s", prompt, e, exc_info=True)
                raise RuntimeError(f"Failed to generate images for prompt '{prompt}': {e}")

            yield images.copy()

        logger.info("Image generation completed for all prompts.")

