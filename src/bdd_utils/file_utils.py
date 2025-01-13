# Contains helper methods for common file operations like read, write, search
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def resolve_config_paths(config: dict) -> dict:
    """Resolve environment variables in the config dictionary."""
    for key, value in config.items():
        if isinstance(value, str):  # Check if the value is a string
            # Check for the pattern ${ENV_VAR:default_value}
            match = re.match(r"\${(\w+):(.+)}", value)
            if match:
                env_var = match.group(1)
                default_value = match.group(2)
                # Get the environment variable or fallback to the default value
                resolved_value = os.getenv(env_var, default_value)
                config[key] = resolved_value
    return config


def load_config(config_path: str) -> dict:
    """Load the config file and resolve environment variables."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Resolve environment variables within the config
    return resolve_config_paths(config)


def read_json(file_path: str) -> Any:
    """Read a JSON file and return its content."""
    with open(file_path, "r") as f:
        return json.load(f)


def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read a YAML file and return its content."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def find_image_and_read(root_folder: str, image_name: str) -> Optional[any]:
    """
    Search for an image by name in the root folder and its subdirectories.
    If found, read and return the image.

    Args:
        root_folder (str): The root folder to start the search.
        image_name (str): The name of the image to search for.

    Returns:
        Optional[any]: The image read using OpenCV (cv2), or None if not found.
    """
    root_path = Path(root_folder)
    image_path = next(root_path.rglob(image_name), None)
    return image_path
