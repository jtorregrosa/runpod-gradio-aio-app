from diffusers import FluxPipeline
from typing import Optional
import time
import torch
import hashlib

class FluxImageGenerator:
    def __init__(self):
        self.pipe = None
        self.previous_hash = None

    def initialize(
        self,
        model_id: str, 
        lora_weights_id: Optional[str], 
        lora_weight_name: Optional[str],
        lora_scale: float, 
        cpu_offload: bool
    ):
        param_string = f"{model_id}-{lora_weights_id or 'None'}-{lora_weight_name or 'None'}-{lora_scale}-{cpu_offload}"
        current_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        # Check if reinitialization is needed
        if self.previous_hash == current_hash:
            return

        self.previous_hash = current_hash
        
        try:
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model '{model_id}': {e}")
    
        if lora_weights_id:
            try:
                pipe.load_lora_weights(lora_weights_id, weight_name=lora_weight_name)
                pipe.fuse_lora(lora_scale=lora_scale)
            except Exception as e:
                raise RuntimeError(f"Failed to load or fuse LoRA weights: {e}")
    
        if cpu_offload:
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to("cuda")   
    
        self.pipe = pipe
    
    def generate(
            self,
            prompt_list,
            model,
            lora_weights,
            width,
            height,
            images_per_prompt,
            lora_scale,
            guidance_scale,
            num_inference_steps,
            max_sequence_length,
            callback
    ):
        images = []
        random_seed = int(time.time())
        generator = torch.Generator("cpu").manual_seed(random_seed)

        # Iterate over each prompt in the prompt list
        for idx, prompt in enumerate(prompt_list):
            def flux_callback(pipe, step_index, timestep, callback_kwargs):
                callback(idx, step_index, len(prompt_list), num_inference_steps)
                return callback_kwargs
                
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

            yield images.copy()
