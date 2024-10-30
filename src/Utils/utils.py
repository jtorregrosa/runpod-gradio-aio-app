import yaml
import logging
import psutil
import random
import torch
import os
import zipfile
import io
import tempfile
from PIL import Image
import base64
import platform
from typing import Any

logger = logging.getLogger(__name__)

def generate_random_prompt():
    # Components for each type of scene
    landscape_adjectives = ["serene", "misty", "vast", "rolling", "snowy", "peaceful", "rocky"]
    landscape_types = ["mountain range", "forest", "desert", "beach", "hills", "tundra", "lake"]
    landscape_details = ["under a clear blue sky", "filled with ancient trees", "stretching to the horizon", "at sunrise", "bathed in soft morning light", "under a full moon", "reflecting the stars"]

    nature_elements = ["river", "meadow", "cliffside", "waterfall", "grove", "field"]
    nature_descriptions = ["winding through a dense forest", "bursting with wildflowers", "covered in lush moss", "cascading over rocky ledges", "filled with blooming lavender", "surrounded by tall oaks", "expansive and golden"]

    animal_adjectives = ["playful", "graceful", "majestic", "curious", "solitary"]
    animal_types = ["herd of deer", "pack of wolves", "eagle", "family of ducks", "group of flamingos", "fox", "pair of rabbits"]
    animal_actions = ["grazing in a clearing", "running through the snow", "soaring high above the valley", "swimming along a pond", "wading in shallow waters", "watching from behind the trees", "hopping through the underbrush"]

    human_activities = ["couple enjoying a quiet picnic", "hiker reaching the summit", "group of friends around a campfire", "family exploring a forest trail", "painter capturing the view", "photographer kneeling for the perfect shot", "young child playing near a creek"]
    human_descriptions = ["under a tree", "surrounded by colorful leaves", "with a breathtaking view", "in the golden sunset", "with paintbrush in hand", "focused and attentive", "laughing and carefree"]

    # Select a random type of scene
    scene_type = random.choice(["landscape", "nature", "animal", "human"])

    if scene_type == "landscape":
        prompt = f"{random.choice(landscape_adjectives)} {random.choice(landscape_types)} {random.choice(landscape_details)}."
    elif scene_type == "nature":
        prompt = f"A {random.choice(nature_elements)} {random.choice(nature_descriptions)}."
    elif scene_type == "animal":
        prompt = f"A {random.choice(animal_adjectives)} {random.choice(animal_types)} {random.choice(animal_actions)}."
    else:  # human
        prompt = f"A {random.choice(human_activities)} {random.choice(human_descriptions)}."

    return prompt

def load_yaml(yaml_file_path: str) -> Any:
    """
    Loads a YAML file and returns its contents as a Python object.

    Parameters:
        yaml_file_path (str): Path to the YAML file to be loaded.

    Returns:
        Any: The contents of the YAML file parsed as a Python object (e.g., dict, list).

    Raises:
        FileNotFoundError: If the YAML file is not found at the specified path.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    try:
        logger.info(f"Attempting to load YAML file from: {yaml_file_path}")
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        logger.info(f"Successfully loaded YAML file from: {yaml_file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"YAML file not found at path: {yaml_file_path}", exc_info=True)
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file at: {yaml_file_path} - {e}", exc_info=True)
        raise

def zip_images(images_list):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(temp.name, 'w') as zip_file:
        for i, image in enumerate(images_list):
            # Decode base64 image to bytes
            byteIO = io.BytesIO()
            image[0].save(byteIO, format='PNG')
            img_bytes = byteIO
            # Write each image to the zip file
            zip_file.writestr(f'{image[1]}.png', img_bytes.getvalue())
    return temp.name

# Helper function to create an ASCII progress bar for a given percentage
def get_ascii_bar(percent, bar_length=20):
    filled_length = int(round(bar_length * percent / 100))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    return f"[{bar}] {percent:.2f}%"

# Function to gather CPU information
def get_cpu_info():
    cpu_name = platform.processor()  # Getting CPU name
    cpu_percent = psutil.cpu_percent(interval=1)
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    # Return formatted HTML for CPU information
    return f"""
    <div>
        <h3 style="text-align: center; margin:1rem;">CPU</h3>
        <div style="border: 1px solid #40444b; border-radius: 10px; padding: 10px; margin: 10px;">
            <p><strong>CPU Name:</strong> {cpu_name}</p>
            <p><strong>CPU Usage:</strong> {get_ascii_bar(cpu_percent)}</p>
            <p><strong>Physical CPU Cores:</strong> {physical_cores}</p>
            <p><strong>Logical CPU Cores:</strong> {logical_cores}</p>
        </div>
    </div>
    """

# Function to gather GPU information
def get_gpu_info():
    # GPU information
    gpu_info = ""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert bytes to GB
            gpu_memory_used = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert bytes to GB
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            gpu_cores = torch.cuda.get_device_properties(i).multi_processor_count  # Number of GPU cores (SMs)

            # Formatting GPU information as HTML
            gpu_info += f"""
            <div>
                <h4>GPU {i} ({gpu_name})</h4>
                <p><strong>Memory Usage:</strong> {get_ascii_bar(gpu_memory_percent)} ({gpu_memory_used:.2f} GB / {gpu_memory_total:.2f} GB)</p>
                <p><strong>Number of GPU Cores (Multiprocessors):</strong> {gpu_cores}</p>
            </div>
            """
    else:
        gpu_info = "<p>No GPU detected.</p>"

    # Return GPU information HTML
    return f"""
    <div>
        <h3 style="text-align: center; margin:1rem;">GPU</h3>
        <div style="border: 1px solid #40444b; border-radius: 10px; padding: 10px; margin: 10px;">
            {gpu_info}
        </div>
    </div>
    """

# Function to gather system memory information
def get_memory_info():
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_total = memory.total / (1024 ** 3)  # Convert bytes to GB
    memory_used = memory.used / (1024 ** 3)  # Convert bytes to GB
    memory_available = memory.available / (1024 ** 3)  # Convert bytes to GB

    # Return formatted HTML for memory usage including total, used, and available RAM
    return f"""
    <div>
        <h3 style="text-align: center; margin:1rem;">Memory</h3>
        <div style="border: 1px solid #40444b; border-radius: 10px; padding: 10px; margin: 10px;">
            <p><strong>Memory Usage:</strong> {get_ascii_bar(memory_percent)} ({memory_used:.2f} GB / {memory_total:.2f} GB)</p>
            <p><strong>Available Memory:</strong> {memory_available:.2f} GB</p>
        </div>
    <div>
    """

def get_filesystem_info():
    workspace_path = os.environ.get("WORKSPACE", "/workspace")
    try:
        usage = psutil.disk_usage(workspace_path)
        total = usage.total / (1024 ** 3)  # Convert bytes to GB
        used = usage.used / (1024 ** 3)  # Convert bytes to GB
        free = usage.free / (1024 ** 3)  # Convert bytes to GB
        percent = usage.percent

        # Formatting filesystem information as HTML
        filesystem_info = f"""
        <div>
            <h4>Workspace Path ({workspace_path})</h4>
            <p><strong>Usage:</strong> {get_ascii_bar(percent)} ({used:.2f} GB / {total:.2f} GB)</p>
            <p><strong>Available Space:</strong> {free:.2f} GB</p>
        </div>
        """
    except FileNotFoundError:
        filesystem_info = f"<p>Path '{workspace_path}' not found.</p>"
    except PermissionError:
        filesystem_info = f"<p>Permission denied to access '{workspace_path}'.</p>"

    # Return filesystem information HTML
    return f"""
    <div>
        <h3 style="text-align: center; margin:1rem;">Filesystem</h3>
        <div style="border: 1px solid #40444b; border-radius: 10px; padding: 10px; margin: 10px;">
            {filesystem_info}
        </div>
    </div>
    """