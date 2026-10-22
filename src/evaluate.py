"""
Model Evaluation Pipeline for Electrical Equipment Image Classification.
Evaluates the trained model on an independent held-out test set, generating:
- Test loss, accuracy, precision, recall, F1-score
- Confusion matrix heatmap
- Detailed classification reports (JSON and Markdown)
- Correct and incorrect qualitative prediction sample grids
- Class distribution charts
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from src.utils import (
    load_config,
    save_json,
    load_json,
    ensure_dir,
    setup_logger
)
from src.dataset_loader import load_dataset_from_directory

logger = setup_logger("eeic.eval")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: str,
    normalize: bool = True
) -> None:
    """Plots and saves a professional confusion matrix heatmap."""
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#334155"
    )

    ax.tick_params(colors="#e2e8f0", labelsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#f8fafc")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#e2e8f0")

    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Class", fontsize=12, fontweight="bold", labelpad=10)
    plt.title(
        f"Normalized Confusion Matrix ({'Normalized' if normalize else 'Raw Counts'})",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {save_path}")


def plot_class_distribution(
    class_counts: Dict[str, Dict[str, int]],
    classes: List[str],
    save_path: str
) -> None:
    """Plots grouped bar chart for train, validation, and test split distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    x = np.arange(len(classes))
    width = 0.25

    train_vals = [class_counts.get("train", {}).get(c, 0) for c in classes]
    val_vals = [class_counts.get("val", {}).get(c, 0) for c in classes]
    test_vals = [class_counts.get("test", {}).get(c, 0) for c in classes]

    ax.bar(x - width, train_vals, width, label="Train", color="#38bdf8")
    ax.bar(x, val_vals, width, label="Validation", color="#34d399")
    ax.bar(x + width, test_vals, width, label="Test", color="#fbbf24")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right", color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.3, color="#64748b", axis="y")

    ax.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_title("Dataset Split Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved class distribution plot to {save_path}")


def plot_prediction_samples(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    save_path: str,
    max_samples: int = 8
) -> None:
    """Plots a grid of sample test predictions with true/predicted labels and confidence."""
    n = min(len(images), max_samples)
    if n == 0:
        return

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.8))
    fig.patch.set_facecolor("#0f172a")

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < n:
            ax.set_facecolor("#1e293b")
            img = images[idx]
            # If normalized in [0, 1]
            if img.max() <= 1.0:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img.astype(np.uint8))
            ax.axis("off")

            is_correct = (true_labels[idx] == pred_labels[idx])
            title_color = "#34d399" if is_correct else "#f43f5e"
            mark = "✓" if is_correct else "✗"

            ax.set_title(
                f"{mark} True: {true_labels[idx]}\nPred: {pred_labels[idx]} ({confidences[idx]*100:.1f}%)",
                fontsize=9.5,
                color=title_color,
                fontweight="bold",
                pad=6
            )
        else:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved prediction samples plot to {save_path}")


def evaluate_model(
    config_path: str = "configs/config.json",
    model_path_override: Optional[str] = None,
    test_dir_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main evaluation pipeline on independent test dataset.
    """
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    results_dir = ensure_dir(paths.get("results_dir", "results"))

    model_path = model_path_override or paths.get("model_path", "models/best_model.keras")
    test_dir = test_dir_override or paths.get("test_dir", "data/processed/test")
    class_mapping_path = paths.get("class_mapping_path", "models/class_mapping.json")

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please train the model first.")

    logger.info(f"Loading trained model from {model_path}...")
    model = keras.models.load_model(model_path)

    img_size = tuple(cfg.get("image_size", [128, 128]))
    batch_size = cfg.get("batch_size", 16)
    seed = cfg.get("seed", 42)

    logger.info(f"Loading test dataset from {test_dir}...")
    test_ds, test_classes = load_dataset_from_directory(
        directory=test_dir,
        image_size=img_size,
        batch_size=batch_size,
        is_training=False,
        shuffle=False,
        seed=seed
    )

    # Evaluate test metrics
    logger.info("Computing test loss and accuracy...")
    eval_results = model.evaluate(test_ds, verbose=1, return_dict=True)

    # Collect predictions and ground truth labels
    all_images = []
    y_true_indices = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true_indices.append(np.argmax(labels.numpy(), axis=-1))
        # Keep unbatched images for sample plotting
        for img in images.numpy():
            if len(all_images) < 24:
                all_images.append(img)

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_true_indices = np.concatenate(y_true_indices, axis=0)
    y_pred_indices = np.argmax(y_pred_probs, axis=-1)
    y_pred_confidences = np.max(y_pred_probs, axis=-1)

    # Classification report
    report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=test_classes,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=test_classes,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(len(test_classes))))

    # Save visual artifacts
    cm_path = str(results_dir / "confusion_matrix.png")
    plot_confusion_matrix(cm, test_classes, cm_path)

    # Plot qualitative samples
    true_labels_str = [test_classes[i] for i in y_true_indices[:len(all_images)]]
    pred_labels_str = [test_classes[i] for i in y_pred_indices[:len(all_images)]]
    samples_path = str(results_dir / "sample_predictions.png")
    plot_prediction_samples(
        all_images,
        true_labels_str,
        pred_labels_str,
        y_pred_confidences[:len(all_images)].tolist(),
        samples_path,
        max_samples=8
    )

    # Plot class distribution if summary is available
    summary_path = Path("data/processed/dataset_summary.json")
    if summary_path.is_file():
        summary_data = load_json(str(summary_path))
        plot_class_distribution(
            summary_data.get("class_split_counts", {}),
            test_classes,
            str(results_dir / "class_distribution.png")
        )

    # Save textual and JSON reports
    with open(results_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    eval_summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": str(Path(model_path).resolve()),
        "test_dir": str(Path(test_dir).resolve()),
        "test_loss": round(float(eval_results.get("loss", 0.0)), 4),
        "test_accuracy": round(float(eval_results.get("accuracy", 0.0)), 4),
        "macro_avg_f1": round(float(report_dict.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "weighted_avg_f1": round(float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)), 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": test_classes,
        "total_test_samples": int(len(y_true_indices))
    }

    metrics_json_path = results_dir / "evaluation_metrics.json"
    save_json(eval_summary, str(metrics_json_path))

    # Generate Markdown report
    generate_markdown_evaluation_report(eval_summary, results_dir / "evaluation_report.md")

    logger.info(
        f"Evaluation complete! Test Accuracy: {eval_summary['test_accuracy'] * 100:.2f}%, "
        f"Test Loss: {eval_summary['test_loss']:.4f}, Macro F1: {eval_summary['macro_avg_f1']:.4f}"
    )
    return eval_summary


def generate_markdown_evaluation_report(summary: Dict[str, Any], report_path: Path) -> None:
    """Generates an aesthetic Markdown evaluation summary document."""
    cr = summary["classification_report"]
    md = [
        "# Model Evaluation Report",
        "",
        f"**Date:** {summary['timestamp']}",
        f"**Model:** `{summary['model_path']}`",
        f"**Test Set:** `{summary['test_dir']}`",
        f"**Total Held-Out Test Images:** {summary['total_test_samples']}",
        "",
        "## Overall Test Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :---: |",
        f"| **Test Accuracy** | **{summary['test_accuracy'] * 100:.2f}%** |",
        f"| **Test Loss (Crossentropy)** | {summary['test_loss']:.4f} |",
        f"| **Macro Avg F1-Score** | {summary['macro_avg_f1']:.4f} |",
        f"| **Weighted Avg F1-Score** | {summary['weighted_avg_f1']:.4f} |",
        "",
        "## Class-Wise Performance",
        "",
        "| Equipment Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for cls in summary["classes"]:
        if cls in cr:
            p = cr[cls]["precision"]
            r = cr[cls]["recall"]
            f1 = cr[cls]["f1-score"]
            sup = int(cr[cls]["support"])
            md.append(f"| {cls} | {p:.2f} | {r:.2f} | {f1:.2f} | {sup} |")

    md.extend([
        "",
        "## Artifacts Generated",
        "",
        "- `results/confusion_matrix.png`: Normalized confusion matrix heatmap.",
        "- `results/sample_predictions.png`: Qualitative prediction sample grid with ground truth tags.",
        "- `results/class_distribution.png`: Split distributions across equipment classes.",
        "- `results/classification_report.json`: Machine-readable per-class performance data.",
        "- `results/classification_report.txt`: Plain-text classification report.",
        ""
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Electrical Equipment Image Classification CNN")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .keras model")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to test set directory")

    args = parser.parse_args()
    evaluate_model(
        config_path=args.config,
        model_path_override=args.model,
        test_dir_override=args.test_dir
    )


if __name__ == "__main__":
    main()
