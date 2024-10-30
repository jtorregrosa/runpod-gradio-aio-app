from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image
from typing import Optional, Callable
import logging
import time
import torch
import hashlib
from PIL import Image, ImageOps

# Configure a logger for this module
logger = logging.getLogger(__name__)

class FluxImageUpscaler:
    """
    A class to initialize and generate upscaled images using FluxControlNetPipeline.
    """

    def __init__(self):
        self.pipe = None
        self.previous_hash = None
        self.controlnetId = 'jasperai/Flux.1-dev-Controlnet-Upscaler'
        self.modelId = 'black-forest-labs/FLUX.1-dev'

    def initialize(
        self,
        cpu_offload: bool
    ):
        """
        Initializes the upscaling pipeline with the given parameters.

        Parameters:
            cpu_offload (bool): Whether to enable CPU offloading for memory efficiency.

        Raises:
            RuntimeError: If loading the model or controlnet fails.
        """
        param_string = f"{self.modelId}-{self.controlnetId}-{cpu_offload}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        # Check if reinitialization is needed
        if self.previous_hash == current_hash:
            logger.info("Using existing pipeline instance as parameters are unchanged.")
            return

        logger.info("Initializing new pipeline with model_id: %s", self.modelId)
        self.previous_hash = current_hash
        
        try:
            controlnet = FluxControlNetModel.from_pretrained(self.controlnetId, torch_dtype=torch.bfloat16)
            pipe = FluxControlNetPipeline.from_pretrained(self.modelId, torch_dtype=torch.bfloat16, controlnet=controlnet, use_safetensors=True)
            logger.info("Successfully loaded controlnet and model.")
        except Exception as e:
            logger.error("Failed to load the controlnet or model: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to load the controlnet or model: {e}")

        if cpu_offload:
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
            logger.info("Enabled CPU offload for the pipeline.")
        else:
            pipe = pipe.to("cuda")
            logger.info("Pipeline moved to CUDA.")

        self.pipe = pipe

    def add_paddings(self, img: Image.Image) -> Image.Image:
        """
        Adds padding to make image dimensions divisible by 8.

        Parameters:
            img (Image.Image): The image to which padding will be added.

        Returns:
            Image.Image: The padded image.
        """
        width, height = img.size
        new_width = (width + 7) // 8 * 8
        new_height = (height + 7) // 8 * 8
        padding = (0, 0, new_width - width, new_height - height)
        img_padded = ImageOps.expand(img, padding, fill=(0, 0, 0))
        
        logger.debug("Added padding to image, new dimensions: %d x %d", new_width, new_height)
        return img_padded

    def generate(
            self,
            control_image: str,
            scale_factor: float,
            controlnet_conditioning_scale: float,
            guidance_scale: float,
            num_inference_steps: int,
            max_sequence_length: int,
            callback: Optional[Callable[[int, int], None]] = None
    ) -> Image.Image:
        """
        Generates an upscaled image based on the given control image.

        Parameters:
            control_image (str): Path to the control image.
            scale_factor (float): Scaling factor for resizing the control image.
            controlnet_conditioning_scale (float): Scale for controlnet conditioning.
            guidance_scale (float): Scale for classifier-free guidance.
            num_inference_steps (int): Number of inference steps for generation.
            max_sequence_length (int): The maximum sequence length for prompt processing.
            callback (Optional[Callable[[int, int], None]]): Callback function for progress tracking.

        Returns:
            Image.Image: The generated upscaled image.

        Raises:
            RuntimeError: If image generation fails.
        """
        if not self.pipe:
            raise RuntimeError("Pipeline is not initialized. Please call 'initialize()' first.")

        try:
            logger.info("Loading and processing control image from: %s", control_image)
            control_image = load_image(control_image)
            w, h = control_image.size
            control_image = control_image.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.LANCZOS)
            control_image = self.add_paddings(control_image)

            random_seed = int(time.time())
            generator = torch.Generator("cpu").manual_seed(random_seed)

            logger.info("Starting image generation with %d inference steps.", num_inference_steps)

            # Define callback for FluxControlNetPipeline to track inference steps
            def flux_callback(pipe, step_index, timestep, callback_kwargs):
                if callback:
                    callback(step_index, num_inference_steps)
                return callback_kwargs

            generated_image = self.pipe(
                prompt="",
                control_image=control_image,
                height=control_image.size[1],
                width=control_image.size[0],
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=generator,
                callback_on_step_end=flux_callback
            ).images[0]

            logger.info("Successfully generated the upscaled image.")
            return generated_image

        except Exception as e:
            logger.error("Failed to generate the upscaled image: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to generate the upscaled image: {e}")
