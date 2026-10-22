"""
Convolutional Neural Network (CNN) Architecture for Electrical Equipment Image Classification.
Defines modular, configurable CNN models for multi-class classification.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import setup_logger

logger = setup_logger("eeic.model")


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 6,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.35,
    name: str = "electrical_equipment_cnn"
) -> keras.Model:
    """
    Constructs a multi-class CNN architecture designed for electrical equipment images.

    Architecture summary:
    - Conv Block 1: 32 filters (3x3), ReLU, BatchNorm, 32 filters, MaxPool(2x2), Dropout(0.2)
    - Conv Block 2: 64 filters (3x3), ReLU, BatchNorm, 64 filters, MaxPool(2x2), Dropout(0.25)
    - Conv Block 3: 128 filters (3x3), ReLU, BatchNorm, MaxPool(2x2), Dropout(0.3)
    - Conv Block 4: 256 filters (3x3), ReLU, BatchNorm, MaxPool(2x2), Dropout(0.35)
    - Dense Head: GlobalAveragePooling2D, Dense(128, ReLU), Dropout(dropout_rate)
    - Output: Dense(num_classes, Softmax)
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 for multi-class classification; got {num_classes}")

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.20, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(dropout_rate, name="drop_head")(x)

    # Multi-class Softmax Output
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    logger.info(f"Built CNN model '{name}' with {num_classes} classes and input shape {input_shape}.")
    return model


def get_model_summary_str(model: keras.Model) -> str:
    """Returns the model.summary() as a string."""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return "\n".join(summary_lines)
