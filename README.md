# Electrical Equipment Image Classification (EEIC)

[![GitHub Repo](https://img.shields.io/badge/GitHub-bhargavithentu28%2FStreamlit-181717.svg?style=flat&logo=github)](https://github.com/bhargavithentu28/Streamlit)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-ready Deep Learning Computer Vision application built with Python, TensorFlow/Keras, and Streamlit to classify critical electrical equipment categories commonly found in substations and distribution power grids.

🔗 **Repository:** [https://github.com/bhargavithentu28/Streamlit](https://github.com/bhargavithentu28/Streamlit)  
🌐 **Local Web Application:** [http://localhost:8501](http://localhost:8501)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [System Requirements](#system-requirements)
5. [Repository Structure](#repository-structure)
6. [Supported Equipment Classes](#supported-equipment-classes)
7. [Installation & Windows Setup](#installation--windows-setup)
8. [Dataset Preparation Pipeline](#dataset-preparation-pipeline)
9. [CNN Model Architecture & Training](#cnn-model-architecture--training)
10. [Model Evaluation & Metrics](#model-evaluation--metrics)
11. [Streamlit Web Application](#streamlit-web-application)
12. [Inference & Prediction Workflow](#inference--prediction-workflow)
13. [Automated Testing](#automated-testing)
14. [Troubleshooting Common Windows & TensorFlow Issues](#troubleshooting-common-windows--tensorflow-issues)
15. [Ethical, Safety, & Operational Boundaries](#ethical-safety--operational-boundaries)
16. [Future Enhancements](#future-enhancements)

---

## 🔍 Project Overview

The **Electrical Equipment Image Classification (EEIC)** system is an industrial computer vision pipeline capable of detecting apparatus classes from high-voltage substation imagery. It automates inventory auditing, enhances drone inspection workflows, and provides maintenance dispatchers with fast, structured visual apparatus recognition.

The system features:
- A multi-block Convolutional Neural Network (CNN) with Batch Normalization, Dropout regularization, and Global Average Pooling.
- Robust data preparation with automated perceptual hashing to eliminate cross-split data leakage.
- Independent test set evaluation with confusion matrix heatmaps, classification reports, and qualitative prediction grids.
- A glassmorphic dark-themed Streamlit web interface for interactive inference and performance analytics.

---

## 🎯 Problem Statement

Power utility substations contain diverse, specialized physical assets operating at high potentials (69 kV to 765 kV+). Routine maintenance, safety zoning, and drone-based inspection require accurate identification of high-voltage apparatus. Manual tagging of thousands of aerial and ground inspection photos is labor-intensive and prone to human cataloging errors.

The EEIC system provides an automated, standardized computer vision pipeline to classify images into canonical equipment categories.

---

## ✨ Key Features

- **End-to-End Pipeline:** Covers raw dataset ingestion, integrity validation, stratified splitting, augmentation, CNN model training, model checkpointing, evaluation, and interactive web inference.
- **Leakage Prevention:** Employs perceptual image hashing to detect near-duplicate images across dataset splits.
- **Dynamic Adaptability:** CNN dynamically configures its output softmax dimension based on discovered classes in the dataset.
- **Reliable Windows & CPU Execution:** Configured for seamless execution on standard Windows developer laptops without requiring dedicated GPU hardware.
- **Transparent Metrics:** Does not fabricate or hardcode training statistics; all metrics and visualizations reflect actual model execution.
- **Confidence Calibration & Warnings:** Displays per-class probability distributions and triggers low-confidence warnings if predictions fall below safety thresholds.

---

## 💻 System Requirements

- **Operating System:** Windows 10 / Windows 11 (64-bit), or Linux / macOS.
- **Python:** 3.11.x (verified compatible with TensorFlow 2.15–2.17).
- **RAM:** Minimum 8 GB (16 GB recommended).
- **Disk Space:** ~2 GB for virtual environment and model artifacts.
- **Hardware Acceleration:** CPU execution by default; CUDA-capable GPU optional.

---

## 📂 Repository Structure

```
EEIC/
├── app.py                     # Streamlit multi-tab web application
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Version control exclusions
├── configs/
│   └── config.json            # Centralized system & hyperparameter config
├── src/
│   ├── __init__.py
│   ├── utils.py               # Config loading, hashing, seed setting, validation
│   ├── data_preparation.py    # Stratified dataset split & integrity validation
│   ├── dataset_loader.py      # Optimized tf.data pipelines & augmentations
│   ├── model.py               # CNN architecture specification & compilation
│   ├── train.py               # Training loop with callbacks and learning curves
│   ├── evaluate.py            # Independent test set evaluation & report generator
│   └── predict.py             # Inference engine, probability ranking, disclaimers
├── tests/
│   ├── test_validation.py     # Image integrity, corruption, and hash tests
│   ├── test_data_preparation.py # Dataset split, counting, and synthetic tests
│   ├── test_model.py          # CNN architecture, shapes, and forward pass tests
│   └── test_prediction.py     # Predictor schema, thresholds, and missing weights tests
├── scripts/
│   ├── setup_windows.ps1      # Automated PowerShell environment setup
│   ├── run_app.ps1            # One-click web application launcher
│   └── train_model.ps1        # One-click train & evaluate pipeline
├── data/
│   ├── README.md              # Dataset folder instructions
│   ├── raw/                   # User-provided or synthetic class subfolders
│   └── processed/             # Output train/val/test splits & summaries
├── models/
│   ├── README.md              # Checkpoint directory documentation
│   ├── best_model.keras       # Trained CNN weights
│   ├── class_mapping.json     # Integer <-> Class name bijection
│   └── model_metadata.json    # Training parameters and history
└── results/
    ├── README.md              # Evaluation outputs guide
    ├── confusion_matrix.png   # Normalized test confusion matrix heatmap
    ├── learning_curves.png    # Training vs. validation loss/accuracy curves
    ├── class_distribution.png # Class counts per split
    ├── sample_predictions.png # Qualitative test prediction sample grid
    ├── classification_report.json # Machine-readable test metrics
    ├── classification_report.txt  # Plaintext precision/recall/F1 table
    └── evaluation_report.md   # Markdown evaluation summary
```

---

## ⚡ Supported Equipment Classes

The pipeline natively recognizes standard substation apparatus:

| Class Name | Electrical Apparatus Description |
| :--- | :--- |
| **`transformer`** | High-voltage power transformers, distribution units, radiators, and primary bushings. |
| **`circuit_breaker`** | High/medium-voltage SF6, vacuum, or air-blast circuit interruption breakers. |
| **`insulator`** | Ceramic, porcelain, or composite suspension disc strings and station post insulators. |
| **`disconnect_switch`** | Air-break disconnectors, bus isolators, and mechanical grounding switches. |
| **`current_transformer`** | Toroidal or pedestal instrument transformers measuring phase currents. |
| **`voltage_transformer`** | Potential / voltage transformers (PTs/VTs) for voltage sensing and protection. |

*Note:* If user datasets contain fewer or more categories, the pipeline dynamically detects available classes without code modifications.

---

## ⚙️ Installation & Windows Setup

### 1. Clone & Navigate to Repository
```powershell
git clone https://github.com/bhargavithentu28/Streamlit.git
cd Streamlit
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
```

### 3. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:**
> If script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation Pipeline

Place your raw image folders inside `data/raw/`:
```
data/raw/
├── transformer/
├── circuit_breaker/
├── insulator/
├── disconnect_switch/
├── current_transformer/
└── voltage_transformer/
```

Run dataset preparation:
```powershell
python -m src.data_preparation
```

### What this utility does:
1. Validates all image files, rejecting corrupt, empty, or unreadable files.
2. Applies perceptual image hashing to discard exact or near-duplicates, preventing cross-split data leakage.
3. Performs a stratified split into **70% Train**, **15% Validation**, and **15% Test** partitions.
4. Generates `class_mapping.json`, `dataset_summary.json`, and `dataset_preparation_report.md`.
5. *Smoke Testing Fallback:* If `data/raw` is empty, it automatically constructs a synthetic dataset with distinct electrical equipment silhouettes for immediate pipeline verification.

---

## 🧠 CNN Model Architecture & Training

### Architecture Specifications
The CNN is built with modular Keras Functional API blocks:
- **Input:** $128 \times 128 \times 3$ RGB images normalized to $[0.0, 1.0]$.
- **Data Augmentation (Train only):** Random flip, $\pm 10\%$ rotation, $\pm 10\%$ zoom, and contrast adjustment.
- **Conv Block 1:** $2 \times$ `Conv2D(32, 3x3)` + `BatchNorm` + `MaxPool(2x2)` + `Dropout(0.20)`
- **Conv Block 2:** $2 \times$ `Conv2D(64, 3x3)` + `BatchNorm` + `MaxPool(2x2)` + `Dropout(0.25)`
- **Conv Block 3:** `Conv2D(128, 3x3)` + `BatchNorm` + `MaxPool(2x2)` + `Dropout(0.30)`
- **Conv Block 4:** `Conv2D(256, 3x3)` + `BatchNorm` + `MaxPool(2x2)` + `Dropout(0.35)`
- **Classifier Head:** `GlobalAveragePooling2D` + `Dense(128, ReLU)` + `BatchNorm` + `Dropout(0.35)` + `Dense(N, Softmax)`

### Launching Training
```powershell
python -m src.train --epochs 20 --batch_size 16 --lr 0.001
```

The training process includes:
- `EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)`
- `ModelCheckpoint(filepath="models/best_model.keras", save_best_only=True)`
- `ReduceLROnPlateau(factor=0.5, patience=3)`
- Generates `results/learning_curves.png` and `models/model_metadata.json`.

---

## 📊 Model Evaluation & Metrics

Evaluate the best model against the independent, held-out test split:
```powershell
python -m src.evaluate
```

This generates:
- **`results/confusion_matrix.png`:** Normalized confusion matrix displaying true vs. predicted classes.
- **`results/sample_predictions.png`:** Test image grid showing predictions, confidence, and correctness tags.
- **`results/class_distribution.png`:** Visual breakdown of train, validation, and test splits.
- **`results/evaluation_metrics.json` & `results/classification_report.txt`:** Macro and per-class precision, recall, and F1 scores.

---

## 🌐 Streamlit Web Application

Launch the local web dashboard:
```powershell
streamlit run app.py
```
Open your browser at: **`http://localhost:8501`**

### Application Tabs:
1. **🏠 Home Dashboard:** Project status KPIs, end-to-end architecture diagram, quick PowerShell commands.
2. **🔍 Image Classification:** Upload JPG/PNG equipment photos or select sample test images; inspect top predicted class, confidence percentage, and full probability distribution.
3. **📊 Model Performance:** Displays held-out test accuracy, confusion matrix heatmap, and training curves.
4. **📁 Dataset Information:** Class split distributions, data pipeline parameters, and integrity metrics.
5. **ℹ️ About The Project:** Technical background, CNN block explanation, and operational safety disclaimers.

---

## 🧪 Automated Testing

Execute the comprehensive `pytest` test suite:
```powershell
pytest tests/ -v
```

Covered test suites:
- `test_validation.py`: Formats, corrupt file detection, missing paths, perceptual hash consistency.
- `test_data_preparation.py`: Scanning, stratified proportions, synthetic dataset generation.
- `test_model.py`: Dynamic output shapes, forward pass checks, serialization and re-loading in `.keras` format.
- `test_prediction.py`: Preprocessing normalization, inference schema, low-confidence warnings, invalid inputs.

---

## 🛠️ Troubleshooting Common Windows & TensorFlow Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Activate.ps1 cannot be loaded because running scripts is disabled`** | PowerShell ExecutionPolicy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your terminal session. |
| **`ImportError: DLL load failed while importing _pywrap_tensorflow_internal`** | Missing Microsoft Visual C++ Redistributable | Install the latest [Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
| **`No module named 'src'`** | Python module path not recognized | Run scripts using `python -m src.<module_name>` from the root directory `c:\Users\bharg\OneDrive\EEIC`. |
| **`Out of Memory (OOM) during training`** | High batch size or large image dimensions | Reduce batch size to `8` or `16` in `configs/config.json`. |
| **`Model file not found` in Web App** | Inference triggered before training | Run `python -m src.data_preparation` followed by `python -m src.train` to train and save weights. |

---

## ⚠️ Ethical, Safety, & Operational Boundaries

1. **Equipment Identification Only:** This classifier identifies physical categories of high-voltage apparatus. It is **not** trained to diagnose internal electrical faults, dielectric breakdown, dissolved gas levels, thermal hotspots, or corona discharge.
2. **Safety Criticality:** Never rely on automated vision predictions alone for electrical safety clearance, lockout/tagout procedures, or energized line verification.
3. **Environmental Variability:** Real-world classification accuracy depends heavily on lighting, weather, camera angle, and background clutter. Low-confidence warnings must be inspected by certified electrical engineers.

---

## 🚀 Future Enhancements

- **Object Detection (YOLOv8 / Faster R-CNN):** Transition from single-image classification to multi-apparatus bounding box detection for full-substation panoramic drone shots.
- **Thermal Infrared (FLIR) Multi-Modal Fusion:** Combining optical RGB with thermal infrared imagery to locate abnormal heating in bushings and contacts.
- **ONNX Runtime Export:** Converting the trained `.keras` checkpoint to ONNX format for microsecond edge deployment on drone onboard processors.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
