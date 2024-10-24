from diffusers import FluxPipeline
from typing import List, Optional, Callable, Generator, Tuple
import logging
import time
import torch
from PIL import Image
import hashlib

# Configure a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FluxImageGenerator:
    """
    A class to initialize and generate images using the FluxPipeline.
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
        cpu_offload: bool = False
    ):
        """
        Initializes the FluxPipeline with the given parameters.

        Parameters:
            model_id (str): The model ID for loading the pre-trained model.
            lora_weights_id (Optional[str]): ID for loading LoRA weights, if any.
            lora_weight_name (Optional[str]): Specific name for LoRA weights to be used.
            lora_scale (float): Scale factor for LoRA weights.
            cpu_offload (bool): Whether to enable CPU offloading for memory efficiency.

        Raises:
            RuntimeError: If loading the model or LoRA weights fails.
        """
        # Create a hash to avoid reinitializing with the same parameters
        param_string = f"{model_id}-{lora_weights_id or 'None'}-{lora_weight_name or 'None'}-{lora_scale}-{cpu_offload}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        if self.previous_hash == current_hash:
            logger.info("Using existing pipeline instance as parameters are unchanged.")
            return

        logger.info("Initializing new pipeline with model_id: %s", model_id)
        self.previous_hash = current_hash

        try:
            # Initialize the pipeline from the model
            self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
            logger.info("Successfully loaded model '%s'", model_id)
        except Exception as e:
            logger.error("Failed to load the model '%s': %s", model_id, e, exc_info=True)
            raise RuntimeError(f"Failed to load the model '{model_id}': {e}")

        if lora_weights_id:
            self._load_and_fuse_lora_weights(lora_weights_id, lora_weight_name, lora_scale)

        if cpu_offload:
            self._enable_cpu_offloading()
        else:
            self.pipe = self.pipe.to("cuda")
            logger.info("Pipeline moved to CUDA.")

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
            prompt_list (List[str]): List of prompts for image generation.
            width (int): Width of generated images.
            height (int): Height of generated images.
            images_per_prompt (int): Number of images to generate per prompt.
            guidance_scale (float): Scale for classifier-free guidance.
            num_inference_steps (int): Number of inference steps for generation.
            max_sequence_length (int): The maximum sequence length for prompt processing.
            callback (Optional[Callable[[int, int, int, int], None]]): Callback function for progress tracking.

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

