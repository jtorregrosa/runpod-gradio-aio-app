import time
import random
import logging
from typing import List, Optional, Tuple, Callable, Generator
from PIL import Image

# Configure a logger for this module
logger = logging.getLogger(__name__)

class FakeImageGenerator:
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
        true_cfg: bool = False,
    ):
        return
    
    def generate(
            self,
            prompt_list: List[str],
            width: int,
            height: int,
            images_per_prompt: int,
            num_inference_steps: int,
            callback: Callable[[int, int, int, int], None] = None
    ) -> Generator[List[Tuple[Image.Image, str]], None, None]:
        """
        Simulates image generation based on a list of prompts.

        Parameters:
            prompt_list (List[str]): A list of prompts to generate images for.
            width (int): The width of each generated image.
            height (int): The height of each generated image.
            images_per_prompt (int): Number of images to generate per prompt.
            num_inference_steps (int): Number of inference steps to simulate.
            callback (Callable[[int, int, int, int], None], optional): A callback to track progress.
                Receives (prompt_idx, step, total_prompts, total_steps).

        Yields:
            List[Tuple[Image.Image, str]]: A list of tuples containing the generated image and its caption.
        """
        images = []

        logger.info("Starting image generation for %d prompts.", len(prompt_list))

        # Iterate over each prompt in the prompt list
        for prompt_idx, prompt in enumerate(prompt_list):
            logger.debug("Generating images for prompt %d: '%s'", prompt_idx + 1, prompt)

            # Perform inference steps for each image
            for step in range(num_inference_steps):
                # Simulate processing time for each step
                time.sleep(0.05)

                # Update the callback to indicate progress for the current step, if provided
                if callback:
                    callback(prompt_idx, step, len(prompt_list), num_inference_steps)

                # Log the progress of inference steps
                logger.debug("Prompt %d, Step %d/%d", prompt_idx + 1, step + 1, num_inference_steps)

            # Generate images for the current prompt
            for img_idx in range(images_per_prompt):
                try:
                    # Create an image with a consistent random RGB color based on the prompt
                    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    random_image = Image.new('RGB', (width, height), random_color)

                    # Append image and caption to the gallery list
                    caption = f"Image {prompt_idx + 1}-{img_idx + 1}"
                    images.append((random_image, caption))

                    logger.info("Generated image %d for prompt %d", img_idx + 1, prompt_idx + 1)

                except Exception as e:
                    logger.error("Error generating image %d for prompt %d: %s", img_idx + 1, prompt_idx + 1, e, exc_info=True)

            # Yield a copy of the images list to update the gallery in real-time
            yield images.copy()

        logger.info("Image generation completed for all prompts.")