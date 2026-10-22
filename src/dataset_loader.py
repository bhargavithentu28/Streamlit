"""
Dataset Loader & Pipeline for Electrical Equipment Image Classification.
Constructs optimized tf.data.Dataset pipelines for training, validation, and testing
with consistent resizing, normalization (1/255), and training-only data augmentation.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import setup_logger, load_config

logger = setup_logger("eeic.loader")


def build_augmentation_layer(aug_config: Optional[Dict[str, Any]] = None) -> keras.Sequential:
    """
    Builds a sequential layer containing training-only image augmentations.
    Augmentation is active only during training (training=True).
    """
    if aug_config is None:
        aug_config = {
            "random_flip": "horizontal",
            "random_rotation": 0.1,
            "random_zoom": 0.1,
            "random_contrast": 0.1
        }

    aug_layers = []

    flip_mode = aug_config.get("random_flip", "horizontal")
    if flip_mode:
        aug_layers.append(layers.RandomFlip(flip_mode))

    rot_factor = aug_config.get("random_rotation", 0.1)
    if rot_factor:
        aug_layers.append(layers.RandomRotation(rot_factor, fill_mode="reflect"))

    zoom_factor = aug_config.get("random_zoom", 0.1)
    if zoom_factor:
        aug_layers.append(layers.RandomZoom(zoom_factor, fill_mode="reflect"))

    contrast_factor = aug_config.get("random_contrast", 0.1)
    if contrast_factor:
        aug_layers.append(layers.RandomContrast(contrast_factor))

    return keras.Sequential(aug_layers, name="data_augmentation")


def load_dataset_from_directory(
    directory: str,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 16,
    is_training: bool = False,
    shuffle: bool = True,
    seed: int = 42,
    aug_layer: Optional[keras.Sequential] = None
) -> Tuple[tf.data.Dataset, List[str]]:
    """
    Loads images from a split folder (organized into class subfolders)
    into an optimized tf.data.Dataset with normalization and caching.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    # Use image_dataset_from_directory
    raw_ds = keras.utils.image_dataset_from_directory(
        directory=str(dir_path),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed
    )

    class_names = raw_ds.class_names
    logger.info(f"Loaded '{dir_path.name}' dataset: {len(class_names)} classes ({class_names})")

    # Rescaling layer for [0, 1] normalization
    rescaling_layer = layers.Rescaling(1.0 / 255.0)

    def preprocess(images, labels):
        # Images are initially float32/uint8 from image_dataset_from_directory
        norm_images = rescaling_layer(images)
        if is_training and aug_layer is not None:
            norm_images = aug_layer(norm_images, training=True)
        return norm_images, labels

    ds = raw_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache in memory and prefetch for CPU/GPU overlap
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds, class_names


def get_dataset_splits(
    config_path: str = "configs/config.json",
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Convenience function to load train, val, and test datasets based on config.
    Returns: (train_ds, val_ds, test_ds, class_names)
    """
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})

    tr_dir = train_dir or paths.get("train_dir", "data/processed/train")
    va_dir = val_dir or paths.get("val_dir", "data/processed/val")
    te_dir = test_dir or paths.get("test_dir", "data/processed/test")

    img_size = tuple(cfg.get("image_size", [128, 128]))
    batch_size = cfg.get("batch_size", 16)
    seed = cfg.get("seed", 42)
    aug_cfg = cfg.get("augmentation", {})

    aug_layer = build_augmentation_layer(aug_cfg)

    train_ds, class_names = load_dataset_from_directory(
        directory=tr_dir,
        image_size=img_size,
        batch_size=batch_size,
        is_training=True,
        shuffle=True,
        seed=seed,
        aug_layer=aug_layer
    )

    val_ds, val_classes = load_dataset_from_directory(
        directory=va_dir,
        image_size=img_size,
        batch_size=batch_size,
        is_training=False,
        shuffle=False,
        seed=seed
    )

    test_ds, test_classes = load_dataset_from_directory(
        directory=te_dir,
        image_size=img_size,
        batch_size=batch_size,
        is_training=False,
        shuffle=False,
        seed=seed
    )

    if class_names != val_classes or class_names != test_classes:
        logger.warning(
            f"Class names mismatch across splits! Train: {class_names}, Val: {val_classes}, Test: {test_classes}"
        )

    return train_ds, val_ds, test_ds, class_names
