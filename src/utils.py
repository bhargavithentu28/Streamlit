"""
Utility functions for Electrical Equipment Image Classification.
Provides configuration loading, logging setup, seed setting, directory creation,
image validation, hashing for duplicate detection, and JSON I/O.
"""

import os
import json
import random
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError

LOGGER_NAME = "eeic"


def setup_logger(name: str = LOGGER_NAME, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


logger = setup_logger()


def load_config(config_path: str = "configs/config.json") -> Dict[str, Any]:
    """Loads a JSON configuration file safely."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.debug(f"Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in configuration file {config_path}: {e}")


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Saves arbitrary serializable data to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.debug(f"Saved JSON data to {file_path}")


def load_json(file_path: str) -> Any:
    """Loads JSON data from file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found at {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Ensure deterministic operations where available
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass
    logger.debug(f"Random seed set to {seed}")


def ensure_dir(path: str) -> Path:
    """Ensures directory exists and returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_image_hash(image_path: str) -> Optional[str]:
    """
    Computes a content hash of an image based on normalized downscaled pixel data
    to detect exact and near-duplicate images across splits.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and small thumbnail for perceptual equality
            img_thumb = img.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            pixel_bytes = img_thumb.tobytes()
            return hashlib.md5(pixel_bytes).hexdigest()
    except Exception:
        return None


def validate_image_file(
    file_path: str,
    supported_extensions: Optional[List[str]] = None,
    max_size_mb: float = 20.0
) -> Tuple[bool, Optional[str]]:
    """
    Validates whether a file is a readable, uncorrupted, valid image.
    Returns (is_valid, error_reason).
    """
    p = Path(file_path)
    if not p.is_file():
        return False, f"File does not exist: {file_path}"

    if supported_extensions:
        exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in supported_extensions]
        if p.suffix.lower() not in exts:
            return False, f"Unsupported file extension: {p.suffix}. Allowed: {exts}"

    # Check file size
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.2f} MB) exceeds limit ({max_size_mb:.2f} MB)"

    try:
        with Image.open(p) as img:
            img.verify()  # Verifies file header and integrity
        # Re-open to verify image data can actually be decoded
        with Image.open(p) as img:
            img.load()
            if img.width == 0 or img.height == 0:
                return False, "Image has 0 dimensions"
        return True, None
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"Corrupted or invalid image: {str(e)}"
