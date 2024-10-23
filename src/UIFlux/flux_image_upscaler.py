from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from diffusers.utils import load_image
from typing import Optional
import time
import torch
import numpy as np
import hashlib
from PIL import Image, ImageOps

class FluxImageUpscaler:
    def __init__(self):
        self.pipe = None
        self.previous_hash = None
        self.controlnetId = 'jasperai/Flux.1-dev-Controlnet-Upscaler'
        self.modelId = 'black-forest-labs/FLUX.1-dev'

    def initialize(
        self,
        cpu_offload: bool
    ):
        param_string = f"{self.modelId}-{self.controlnetId}-{cpu_offload}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        # Check if reinitialization is needed
        if self.previous_hash == current_hash:
            return

        self.previous_hash = current_hash
        
        controlnet = FluxControlNetModel.from_pretrained(self.controlnetId, torch_dtype=torch.bfloat16)
        pipe = FluxControlNetPipeline.from_pretrained(self.modelId, torch_dtype=torch.bfloat16, controlnet=controlnet, use_safetensors=True)
    
        if cpu_offload:
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to("cuda")   
    
        self.pipe = pipe

    def add_paddings(self, img):
        width, height = img.size
        new_width = (width + 7) // 8 * 8
        new_height = (height + 7) // 8 * 8
        padding = (0, 0, new_width - width, new_height - height)
        img_padded = ImageOps.expand(img, padding, fill=(0, 0, 0))
        
        return img_padded
    
    def generate(
            self,
            control_image,
            scale_factor,
            controlnet_conditioning_scale,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
            callback
    ):
        random_seed = int(time.time())
        generator = torch.Generator("cpu").manual_seed(random_seed)
        control_image = load_image(control_image)
        w, h = control_image.size
        control_image = control_image.resize((int(w * scale_factor), int(h * scale_factor)), Image.Resampling.LANCZOS)
        control_image = self.add_paddings(control_image)
        
        def flux_callback(pipe, step_index, timestep, callback_kwargs):
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

        return generated_image
