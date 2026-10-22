"""
Automated unit tests for inference and prediction engine:
- Missing model error handling
- Preprocessing and normalization checks
- Prediction dictionary schema and finite probabilities
- Low confidence warning detection
- Corrupt/unsupported image rejection
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from src.predict import EquipmentPredictor
from src.model import build_cnn_model
from src.utils import save_json


@pytest.fixture
def dummy_model_env(tmp_path):
    """Sets up a temporary trained model and class mapping environment for testing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.keras"
    mapping_path = model_dir / "class_mapping.json"
    cfg_path = tmp_path / "config.json"

    classes = ["transformer", "circuit_breaker", "insulator"]
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=len(classes))
    model.save(str(model_path))

    mapping = {
        "class_to_idx": {cls: i for i, cls in enumerate(classes)},
        "idx_to_class": {str(i): cls for i, cls in enumerate(classes)},
        "classes": classes,
        "num_classes": len(classes)
    }
    save_json(mapping, str(mapping_path))

    cfg = {
        "image_size": [32, 32],
        "channels": 3,
        "paths": {
            "model_path": str(model_path),
            "class_mapping_path": str(mapping_path)
        },
        "inference": {
            "confidence_threshold": 0.60,
            "max_upload_size_mb": 10.0
        },
        "supported_formats": [".jpg", ".jpeg", ".png"]
    }
    save_json(cfg, str(cfg_path))

    return cfg_path, model_path, mapping_path


def test_missing_model_error(tmp_path):
    """Validates that EquipmentPredictor reports clear error when model file is absent."""
    missing_cfg = tmp_path / "missing_cfg.json"
    save_json({
        "image_size": [32, 32],
        "paths": {
            "model_path": str(tmp_path / "non_existent.keras"),
            "class_mapping_path": str(tmp_path / "non_existent.json")
        }
    }, str(missing_cfg))

    predictor = EquipmentPredictor(config_path=str(missing_cfg))
    ready, reason = predictor.is_model_ready()
    assert ready is False
    assert "not found" in reason

    with pytest.raises(FileNotFoundError):
        predictor.predict("some_image.png")


def test_preprocessing_output(dummy_model_env, tmp_path):
    """Validates image normalization in [0, 1] and correct tensor batch shape."""
    cfg_path, _, _ = dummy_model_env
    predictor = EquipmentPredictor(config_path=str(cfg_path))

    test_img = Image.new("RGB", (100, 80), color=(128, 200, 50))
    tensor, pil_out = predictor.preprocess_image(test_img)

    assert tensor.shape == (1, 32, 32, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_prediction_output_schema(dummy_model_env, tmp_path):
    """Validates full inference output dictionary schema and numerical validity."""
    cfg_path, _, _ = dummy_model_env
    predictor = EquipmentPredictor(config_path=str(cfg_path))

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(50, 150, 200)).save(img_path)

    result = predictor.predict(str(img_path))

    assert "predicted_class" in result
    assert result["predicted_class"] in ["transformer", "circuit_breaker", "insulator"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "confidence_percentage" in result
    assert "%" in result["confidence_percentage"]
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    assert np.isclose(sum(result["probabilities"].values()), 1.0, atol=1e-2)
    assert "is_low_confidence" in result
    assert "disclaimer" in result


def test_invalid_image_upload(dummy_model_env, tmp_path):
    """Validates graceful failure when an invalid image is provided."""
    cfg_path, _, _ = dummy_model_env
    predictor = EquipmentPredictor(config_path=str(cfg_path))

    invalid_file = tmp_path / "broken.jpg"
    invalid_file.write_text("not an image at all")

    with pytest.raises(ValueError):
        predictor.predict(str(invalid_file))
