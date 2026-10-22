# Model Evaluation Report

**Date:** 2026-10-22T14:39:35.894710Z
**Model:** `C:\Users\bharg\OneDrive\EEIC\models\best_model.keras`
**Test Set:** `C:\Users\bharg\OneDrive\EEIC\data\processed\test`
**Total Held-Out Test Images:** 18

## Overall Test Metrics

| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | **16.67%** |
| **Test Loss (Crossentropy)** | 1.7908 |
| **Macro Avg F1-Score** | 0.0476 |
| **Weighted Avg F1-Score** | 0.0476 |

## Class-Wise Performance

| Equipment Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| circuit_breaker | 0.00 | 0.00 | 0.00 | 3 |
| current_transformer | 0.00 | 0.00 | 0.00 | 3 |
| disconnect_switch | 0.17 | 1.00 | 0.29 | 3 |
| insulator | 0.00 | 0.00 | 0.00 | 3 |
| transformer | 0.00 | 0.00 | 0.00 | 3 |
| voltage_transformer | 0.00 | 0.00 | 0.00 | 3 |

## Artifacts Generated

- `results/confusion_matrix.png`: Normalized confusion matrix heatmap.
- `results/sample_predictions.png`: Qualitative prediction sample grid with ground truth tags.
- `results/class_distribution.png`: Split distributions across equipment classes.
- `results/classification_report.json`: Machine-readable per-class performance data.
- `results/classification_report.txt`: Plain-text classification report.
