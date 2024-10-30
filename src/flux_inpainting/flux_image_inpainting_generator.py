from typing import List, Optional, Callable, Generator, Tuple
import logging
import time
import torch
from PIL import Image
import hashlib
from .controlnet_flux import FluxControlNetModel
from .transformer_flux import FluxTransformer2DModel
from .pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

# Configure a logger for this module
logger = logging.getLogger(__name__)

class FluxImageInpaintingGenerator:
    def __init__(self):
        self.pipe = None
        self.previous_hash = None
        self.true_cfg = False

    def initialize(
        self,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        vae_slicing: bool = False,
        vae_tiling: bool = False,
    ):
        # Create a hash to avoid reinitializing with the same parameters
        param_string = f"{model_cpu_offload}-{sequential_cpu_offload}-{vae_slicing}-{vae_tiling}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        if self.previous_hash == current_hash:
            logger.info("Using existing pipeline instance as parameters are unchanged.")
            return

        logger.info("Initializing new pipeline")
        self.previous_hash = current_hash

        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dytpe=torch.bfloat16
            )
            controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
            self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                controlnet=controlnet,
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
            self.pipe.transformer.to(torch.bfloat16)
            self.pipe.controlnet.to(torch.bfloat16)
            
            logger.info("Successfully loaded model")
        except Exception as e:
            logger.error("Failed to load the model: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to load the model: {e}")

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

    def generate(
            self,
            prompt,
            negative_prompt,
            control_image,
            control_mask,
            controlnet_conditioning_scale,
            guidance_scale: float,
            num_inference_steps: int,
            callback: Optional[Callable[[int, int, int, int], None]] = None
    ):
        if not self.pipe:
            raise RuntimeError("Pipeline is not initialized. Please call 'initialize()' first.")

        random_seed = int(time.time())
        generator = torch.Generator("cpu").manual_seed(random_seed)

        logger.info("Starting image generation")

        def flux_callback(pipe, step_index, timestep, callback_kwargs):
            if callback:
                callback(step_index, num_inference_steps)
            return callback_kwargs
        
        try:
            size = (1024, 1024)
            image_or = control_image.copy()
            control_image = control_image.convert("RGB").resize(size)
            control_mask = control_mask.convert("RGB").resize(size)
            result = self.pipe(
                prompt,
                negative_prompt,
                height=size[1],
                width=size[0],
                guidance_scale=guidance_scale,
                true_guidance_scale=guidance_scale,
                control_image=control_image,
                control_mask=control_mask,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback_on_step_end=flux_callback,
            ).images[0]

            logger.info("Image generation completed for all prompts.")  
            return result.resize((image_or.size[:2]))

        except Exception as e:
            logger.error("Failed to generate images for prompt '%s': %s", prompt, e, exc_info=True)
            raise RuntimeError(f"Failed to generate images for prompt '{prompt}': {e}")
