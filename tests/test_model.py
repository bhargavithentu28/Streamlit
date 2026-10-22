"""
Automated unit tests for CNN model architecture:
- Layer construction and dynamic class output shapes
- Forward pass tensor inference
- Model serialization and reloading (.keras format)
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model import build_cnn_model, get_model_summary_str


def test_build_cnn_model_output_shape():
    """Validates that CNN output layer dynamically adapts to requested number of classes."""
    for num_classes in [3, 6, 8]:
        model = build_cnn_model(input_shape=(64, 64, 3), num_classes=num_classes)
        assert model.output_shape == (None, num_classes)
        assert model.input_shape == (None, 64, 64, 3)
        assert model.loss == "categorical_crossentropy"


def test_build_cnn_invalid_classes():
    """Validates rejection when num_classes < 2."""
    with pytest.raises(ValueError):
        build_cnn_model(num_classes=1)


def test_model_forward_pass():
    """Validates forward pass with random float tensor producing valid probabilities."""
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=4)
    dummy_input = np.random.rand(2, 32, 32, 3).astype(np.float32)
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (2, 4)
    # Output of softmax should sum to 1.0 per sample
    assert np.allclose(np.sum(output, axis=-1), [1.0, 1.0], atol=1e-5)
    assert np.all(np.isfinite(output))


def test_model_save_and_reload(tmp_path):
    """Validates model serialization and re-loading in .keras format."""
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=3)
    save_path = tmp_path / "test_model.keras"

    model.save(str(save_path))
    assert save_path.is_file()

    reloaded = keras.models.load_model(str(save_path))
    assert reloaded.output_shape == (None, 3)

    dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    pred_orig = model(dummy_input, training=False).numpy()
    pred_reload = reloaded(dummy_input, training=False).numpy()

    assert np.allclose(pred_orig, pred_reload, atol=1e-5)


def test_model_summary_str():
    """Validates that model summary generator returns non-empty formatted string."""
    model = build_cnn_model(input_shape=(32, 32, 3), num_classes=3)
    summary_text = get_model_summary_str(model)
    assert "conv" in summary_text
    assert "predictions" in summary_text
