from diffusers import FluxPipeline
from .flux_cfg_pipeline import FluxCFGPipeline
from typing import List, Optional, Callable, Generator, Tuple
import logging
import time
import torch
from PIL import Image
import hashlib

# Configure a logger for this module
logger = logging.getLogger(__name__)

class FluxImageGenerator:
    """
    A class to initialize and generate images using the FluxPipeline.
    """

    def __init__(self):
        self.pipe = None
        self.previous_hash = None
        self.true_cfg = False

    def initialize(
        self,
        model_id: str,
        lora_weight_1_id: Optional[str] = None,
        lora_weight_1_name: Optional[str] = None,
        lora_adapter_1_weight: float = 1.0,
        lora_weight_2_id: Optional[str] = None,
        lora_weight_2_name: Optional[str] = None,
        lora_adapter_2_weight: float = 1.0,
        lora_scale: float = 1.0,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        vae_slicing: bool = False,
        vae_tiling: bool = False,
        true_cfg: bool = False,
    ):
        # Create a hash to avoid reinitializing with the same parameters
        param_string = f"{model_id}-{lora_weight_1_id or 'None'}-{lora_weight_1_name or 'None'}-{lora_adapter_1_weight or 'None'}-{lora_weight_2_id or 'None'}-{lora_weight_2_name or 'None'}-{lora_adapter_2_weight or 'None'}{lora_scale}-{model_cpu_offload}-{sequential_cpu_offload}-{vae_slicing}-{vae_tiling}-{true_cfg}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        if self.previous_hash == current_hash:
            logger.info("Using existing pipeline instance as parameters are unchanged.")
            return

        logger.info("Initializing new pipeline with model_id: %s", model_id)
        self.previous_hash = current_hash

        try:
            # Initialize the pipeline from the model
            if true_cfg:
                self.true_cfg = True
                self.pipe = FluxCFGPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
            else:
                self.true_cfg = False
                self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)

            logger.info("Successfully loaded model '%s'", model_id)
        except Exception as e:
            logger.error("Failed to load the model '%s': %s", model_id, e, exc_info=True)
            raise RuntimeError(f"Failed to load the model '{model_id}': {e}")

        if lora_weight_1_id:
            self._load_and_fuse_lora_weights(
                lora_weight_1_id, 
                lora_weight_1_name,
                lora_adapter_1_weight,
                lora_weight_2_id, 
                lora_weight_2_name,
                lora_adapter_2_weight, 
                lora_scale
            )

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

    def _load_and_fuse_lora_weights(
        self, 
        lora_weight_1_id: str, 
        lora_weight_1_name: Optional[str],
        lora_adapter_1_weight: float,
        lora_weight_2_id: Optional[str],
        lora_weight_2_name: Optional[str],
        lora_adapter_2_weight: float,
        lora_scale: float
    ):
        try:
            self.pipe.load_lora_weights(lora_weight_1_id, weight_name=lora_weight_1_name, adapter_name="lora1")

            if lora_weight_2_id:
                self.pipe.load_lora_weights(lora_weight_2_id, weight_name=lora_weight_2_name, adapter_name="lora2")
                self.pipe.set_adapters(["lora1", "lora2"], adapter_weights=[lora_adapter_1_weight, lora_adapter_2_weight])
                self.pipe.fuse_lora(adapter_names=["lora1", "lora2"], lora_scale=lora_scale)
            else:
                self.pipe.fuse_lora(adapter_names=["lora1"], lora_scale=lora_scale)

            logger.info("Successfully loaded and fused LoRA weights")
        except Exception as e:
            logger.error("Failed to load or fuse LoRA weights '%s': %s", lora_weights_id, e, exc_info=True)
            raise RuntimeError(f"Failed to load or fuse LoRA weights: {e}")

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
                if self.true_cfg:
                    # Generate images for the current prompt
                    prompt_images = self.pipe(
                        prompt,
                        negative_prompt,
                        height=height,
                        width=width,
                        guidance_scale=1,
                        true_cfg=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        generator=generator,
                        callback_on_step_end=flux_callback,
                        num_images_per_prompt=images_per_prompt
                    ).images
                else:
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

