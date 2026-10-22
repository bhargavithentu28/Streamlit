# Results and Evaluation Artifacts

This directory stores outputs generated during model evaluation and dataset preparation.

## Generated Artifacts

- `confusion_matrix.png`: Normalized and raw count confusion matrix across all evaluated classes on the independent test set.
- `learning_curves.png`: Training and validation loss and accuracy curves across all trained epochs.
- `class_distribution.png`: Bar chart displaying sample counts per class across train, validation, and test splits.
- `sample_predictions.png`: Visual montage of test predictions showing true label, predicted label, and confidence score (including both correct and misclassified samples).
- `classification_report.txt`: Standard scikit-learn classification report (precision, recall, f1-score, support).
- `classification_metrics.json`: Machine-readable evaluation summary containing loss, accuracy, macro averages, weighted averages, and per-class metrics.
- `dataset_preparation_report.md`: Markdown summary of image validation, deduplication, and split allocations.
