"""
Model Training Pipeline for Electrical Equipment Image Classification.
Trains the CNN model, handles callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau),
saves weights in .keras format, stores metadata, and generates learning curves.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.utils import (
    load_config,
    save_json,
    load_json,
    set_seed,
    ensure_dir,
    setup_logger
)
from src.dataset_loader import get_dataset_splits
from src.model import build_cnn_model, get_model_summary_str

logger = setup_logger("eeic.train")


def plot_learning_curves(history: keras.callbacks.History, save_path: str) -> None:
    """Plots and saves loss and accuracy learning curves."""
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")  # Dark slate

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#e2e8f0")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")

    # Accuracy Plot
    axes[0].plot(epochs_range, hist.get("accuracy", []), label="Train Accuracy", color="#38bdf8", lw=2.2)
    if "val_accuracy" in hist:
        axes[0].plot(epochs_range, hist["val_accuracy"], label="Val Accuracy", color="#34d399", lw=2.2, linestyle="--")
    axes[0].set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[0].set_ylim([0, 1.05])

    # Loss Plot
    axes[1].plot(epochs_range, hist.get("loss", []), label="Train Loss", color="#f43f5e", lw=2.2)
    if "val_loss" in hist:
        axes[1].plot(epochs_range, hist["val_loss"], label="Val Loss", color="#fbbf24", lw=2.2, linestyle="--")
    axes[1].set_title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Loss (Categorical Crossentropy)", fontsize=11)
    axes[1].legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved learning curves plot to {save_path}")


def train_model(
    config_path: str = "configs/config.json",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes model training with early stopping and checkpointing.
    """
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    paths = cfg.get("paths", {})
    model_dir = ensure_dir(paths.get("model_dir", "models"))
    results_dir = ensure_dir(paths.get("results_dir", "results"))

    best_model_path = paths.get("model_path", "models/best_model.keras")
    model_metadata_path = paths.get("model_metadata_path", "models/model_metadata.json")
    class_mapping_path = paths.get("class_mapping_path", "models/class_mapping.json")

    epochs = epochs_override if epochs_override is not None else cfg.get("epochs", 20)
    lr = lr_override if lr_override is not None else cfg.get("learning_rate", 0.001)
    img_size = tuple(cfg.get("image_size", [128, 128]))

    logger.info(f"Loading dataset splits from {paths.get('processed_data_dir', 'data/processed')}...")
    train_ds, val_ds, test_ds, class_names = get_dataset_splits(config_path)

    num_classes = len(class_names)
    logger.info(f"Loaded {num_classes} classes: {class_names}")

    # Build model
    input_shape = (img_size[0], img_size[1], cfg.get("channels", 3))
    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=lr
    )

    training_cfg = cfg.get("training", {})
    early_patience = training_cfg.get("early_stopping_patience", 6)
    reduce_patience = training_cfg.get("reduce_lr_patience", 3)
    reduce_factor = training_cfg.get("reduce_lr_factor", 0.5)
    min_lr = training_cfg.get("min_lr", 1e-6)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_factor,
            patience=reduce_patience,
            min_lr=min_lr,
            verbose=1
        )
    ]

    logger.info(f"Starting model training for up to {epochs} epochs...")
    start_time = datetime.utcnow()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    duration_sec = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Training completed in {duration_sec:.2f} seconds.")

    # Save final model explicitly if not already saved by checkpoint
    if not os.path.exists(best_model_path):
        model.save(best_model_path)
        logger.info(f"Saved best model checkpoint to {best_model_path}")

    # Plot learning curves
    curves_path = str(results_dir / "learning_curves.png")
    plot_learning_curves(history, curves_path)

    # Save training metadata
    train_metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "training_duration_seconds": round(duration_sec, 2),
        "learning_rate": lr,
        "input_shape": list(input_shape),
        "classes": class_names,
        "num_classes": num_classes,
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "final_train_accuracy": round(float(history.history["accuracy"][-1]), 4),
        "final_val_loss": round(float(history.history["val_loss"][-1]), 4) if "val_loss" in history.history else None,
        "final_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4) if "val_accuracy" in history.history else None,
        "best_val_loss": round(float(min(history.history["val_loss"])), 4) if "val_loss" in history.history else None,
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()}
    }

    save_json(train_metadata, model_metadata_path)
    logger.info(f"Saved model training metadata to {model_metadata_path}")

    return train_metadata


def main():
    parser = argparse.ArgumentParser(description="Train Electrical Equipment Image Classification CNN")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs to train (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")

    args = parser.parse_args()
    train_model(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr
    )


if __name__ == "__main__":
    main()
