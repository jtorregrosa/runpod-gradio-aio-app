class FakeImageGenerator:

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

        # Iterate over each prompt in the prompt list
        for idx, prompt in enumerate(prompt_list):
            # Perform inference steps for each image
            for step in range(num_inference_steps):
                # Simulate processing time for each step
                time.sleep(0.05)

                # Update the callback to indicate progress for the current step
                callback(idx, step, len(prompt_list), num_inference_steps)
            for img_idx in range(images_per_prompt):
                # Create an image with a consistent random RGB color based on the prompt
                random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                random_image = Image.new('RGB', (width, height), random_color)

                # Append image and caption to the gallery list
                images.append((random_image, f"Image {idx + 1}-{img_idx + 1}"))

            # Yield a copy of the images list to update the gallery in real-time
            yield images.copy()