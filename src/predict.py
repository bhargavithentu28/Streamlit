"""
Inference & Prediction Module for Electrical Equipment Image Classification.
Loads trained Keras models, pre-processes input images with identical training pipeline
rescaling and normalization, validates inputs, and outputs prediction probabilities and confidence scores.
"""

import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow import keras

from src.utils import (
    load_config,
    load_json,
    setup_logger,
    validate_image_file
)

logger = setup_logger("eeic.predict")


class EquipmentPredictor:
    """
    Reusable inference engine for electrical equipment image classification.
    Manages model caching, preprocessing validation, and structured output formatting.
    """

    def __init__(
        self,
        config_path: str = "configs/config.json",
        model_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None
    ):
        self.config_path = config_path
        self.config = load_config(config_path)

        paths = self.config.get("paths", {})
        self.model_path = Path(model_path or paths.get("model_path", "models/best_model.keras"))
        self.class_mapping_path = Path(class_mapping_path or paths.get("class_mapping_path", "models/class_mapping.json"))

        self.image_size = tuple(self.config.get("image_size", [128, 128]))
        self.supported_formats = self.config.get("supported_formats", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
        self.confidence_threshold = self.config.get("inference", {}).get("confidence_threshold", 0.50)
        self.max_upload_size_mb = self.config.get("inference", {}).get("max_upload_size_mb", 15.0)

        self._model: Optional[keras.Model] = None
        self._class_mapping: Optional[Dict[str, Any]] = None
        self._idx_to_class: Optional[Dict[int, str]] = None

    def is_model_ready(self) -> Tuple[bool, Optional[str]]:
        """Checks if the trained model weights and class mapping are present."""
        if not self.model_path.is_file():
            return False, f"Model file not found at: {self.model_path}. Train the model first via `python -m src.train`."
        if not self.class_mapping_path.is_file():
            return False, f"Class mapping file not found at: {self.class_mapping_path}."
        return True, None

    def load_model(self) -> None:
        """Loads model and class mapping if not already in memory."""
        ready, reason = self.is_model_ready()
        if not ready:
            raise FileNotFoundError(reason)

        if self._model is None:
            logger.info(f"Loading trained CNN model from {self.model_path}...")
            self._model = keras.models.load_model(str(self.model_path))

        if self._class_mapping is None:
            self._class_mapping = load_json(str(self.class_mapping_path))
            raw_idx_to_class = self._class_mapping.get("idx_to_class", {})
            # Ensure integer keys
            self._idx_to_class = {int(k): v for k, v in raw_idx_to_class.items()}

    def preprocess_image(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Validates, converts to RGB, resizes, and normalizes image for inference.
        Returns: (preprocessed_tensor_batch, original_pil_image)
        """
        # Load into PIL Image
        if isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, (str, Path)):
            p = Path(image_input)
            is_valid, reason = validate_image_file(str(p), self.supported_formats, self.max_upload_size_mb)
            if not is_valid:
                raise ValueError(f"Invalid input image: {reason}")
            pil_img = Image.open(p)
        elif isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input))
        elif hasattr(image_input, "read"):
            # File-like object (e.g., Streamlit UploadedFile)
            image_input.seek(0)
            pil_img = Image.open(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Convert to RGB mode
        try:
            rgb_img = pil_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to convert image to RGB: {e}")

        # Resize to input dimensions
        resized_img = rgb_img.resize(self.image_size, Image.Resampling.BILINEAR)

        # Convert to numpy float32 and normalize to [0, 1] identically to training pipeline
        img_array = np.array(resized_img, dtype=np.float32) / 255.0

        # Expand batch dimension (1, H, W, C)
        tensor_batch = np.expand_dims(img_array, axis=0)

        return tensor_batch, rgb_img

    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]
    ) -> Dict[str, Any]:
        """
        Executes inference on an image and returns predictions, probabilities,
        confidence scores, and validation status.
        """
        self.load_model()
        tensor_batch, pil_img = self.preprocess_image(image_input)

        # Run inference
        probs = self._model.predict(tensor_batch, verbose=0)[0]

        # Numerical validation
        if not np.all(np.isfinite(probs)):
            raise ValueError("Model produced non-finite (NaN or Inf) probability values.")

        pred_idx = int(np.argmax(probs))
        top_prob = float(probs[pred_idx])

        if self._idx_to_class and pred_idx in self._idx_to_class:
            predicted_class = self._idx_to_class[pred_idx]
        else:
            predicted_class = f"Class_{pred_idx}"

        # Build full probability distribution sorted descending
        prob_distribution: Dict[str, float] = {}
        for idx, p in enumerate(probs):
            name = self._idx_to_class.get(idx, f"Class_{idx}") if self._idx_to_class else f"Class_{idx}"
            prob_distribution[name] = round(float(p), 4)

        sorted_probs = dict(sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True))

        is_low_confidence = (top_prob < self.confidence_threshold)
        warning_msg = None
        if is_low_confidence:
            warning_msg = (
                f"Confidence ({top_prob * 100:.1f}%) is below the configured threshold "
                f"({self.confidence_threshold * 100:.0f}%). The image might be ambiguous, "
                "poorly lit, occluded, or outside the trained equipment categories."
            )

        return {
            "predicted_class": predicted_class,
            "predicted_index": pred_idx,
            "confidence": top_prob,
            "confidence_percentage": f"{top_prob * 100:.2f}%",
            "probabilities": sorted_probs,
            "is_low_confidence": is_low_confidence,
            "warning": warning_msg,
            "image_size": pil_img.size,
            "disclaimer": (
                "Notice: This model classifies electrical equipment categories (e.g. transformer, "
                "insulator, circuit breaker) and does NOT diagnose electrical faults, fire hazards, "
                "or structural damage."
            )
        }
