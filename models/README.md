# Models Directory

This directory stores trained model checkpoints, final weights in Keras format, and associated metadata.

## Files

- `best_model.keras`: The best model checkpoint saved during training using validation loss monitoring.
- `class_mapping.json`: The integer-to-class and class-to-integer mappings corresponding to the trained model.
- `model_metadata.json`: Architectural hyperparameters, input shapes, training history summary, and creation timestamps.

## Important Notes

- Do not commit large binary model files to version control repositories.
- The web application automatically detects if `best_model.keras` and `class_mapping.json` are present. If missing, it provides guidance on running the training pipeline.
