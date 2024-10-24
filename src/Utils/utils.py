import yaml
import logging
from typing import Any

logger = logging.getLogger(__name__)

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