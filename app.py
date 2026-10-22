"""
Electrical Equipment Image Classification - Streamlit Web Application.
Provides a modern, responsive interface for uploading images, running CNN inference,
inspecting evaluation metrics, reviewing dataset distributions, and exploring the pipeline.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import load_config, load_json
from src.predict import EquipmentPredictor

# Page Configuration
st.set_page_config(
    page_title="Electrical Equipment Image Classification",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Modern Dark Glassmorphic Theme with Electric Accents)
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Gradient Headers */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Stat Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.12);
    }
    .pred-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 0.6rem;
    }
    .pred-class {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
        text-transform: capitalize;
    }
    .pred-conf {
        font-size: 1.15rem;
        font-weight: 600;
        color: #34d399;
    }

    /* Alert / Disclaimer */
    .disclaimer-card {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.8rem 1.1rem;
        color: #cbd5e1;
        font-size: 0.85rem;
        margin-top: 1rem;
    }

    /* Empty state styling */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        background: rgba(30, 41, 59, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: #94a3b8;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_predictor() -> EquipmentPredictor:
    """Instantiates and caches the prediction engine."""
    return EquipmentPredictor()


def format_class_name(raw_name: str) -> str:
    """Formats directory class name to user-friendly title."""
    return raw_name.replace("_", " ").title()


def load_all_metadata() -> Dict[str, Any]:
    """Reads available model metadata, dataset summary, and evaluation metrics."""
    meta: Dict[str, Any] = {
        "model_exists": False,
        "mapping_exists": False,
        "evaluation_exists": False,
        "dataset_exists": False,
        "eval_data": None,
        "summary_data": None,
        "train_data": None
    }

    if Path("models/best_model.keras").is_file():
        meta["model_exists"] = True

    if Path("models/class_mapping.json").is_file():
        meta["mapping_exists"] = True

    if Path("models/model_metadata.json").is_file():
        meta["train_data"] = load_json("models/model_metadata.json")

    if Path("results/evaluation_metrics.json").is_file():
        meta["evaluation_exists"] = True
        meta["eval_data"] = load_json("results/evaluation_metrics.json")

    if Path("data/processed/dataset_summary.json").is_file():
        meta["dataset_exists"] = True
        meta["summary_data"] = load_json("data/processed/dataset_summary.json")

    return meta


def render_sidebar(meta: Dict[str, Any]) -> str:
    """Renders the sidebar navigation and operational status badges."""
    with st.sidebar:
        st.markdown("## ⚡ **EEIC System**")
        st.caption("Deep Learning Vision Pipeline")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            [
                "🏠 Home Dashboard",
                "🔍 Image Classification",
                "📊 Model Performance",
                "📁 Dataset Information",
                "ℹ️ About The Project"
            ],
            key="main_navigation"
        )

        st.markdown("---")
        st.markdown("### **System Status**")

        if meta["model_exists"]:
            st.success("● Model Trained & Ready")
        else:
            st.warning("○ Model Untrained")

        if meta["dataset_exists"]:
            n_classes = meta["summary_data"].get("num_classes", 0)
            st.info(f"● Dataset Prepared ({n_classes} Classes)")
        else:
            st.caption("○ Dataset Not Prepared")

        if meta["evaluation_exists"]:
            acc = meta["eval_data"].get("test_accuracy", 0.0) * 100
            st.success(f"● Test Accuracy: {acc:.1f}%")
        else:
            st.caption("○ Unverified Test Metrics")

        st.markdown("---")
        st.caption("Environment: CPU-compatible TensorFlow 2.17.1 on Windows")

    return page


def render_home_dashboard(meta: Dict[str, Any]):
    """Renders Project Overview, Model Status, and Workflow."""
    st.markdown('<div class="hero-title">Electrical Equipment Image Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">High-precision computer vision pipeline identifying critical power substation and distribution apparatus.</div>',
        unsafe_allow_html=True
    )

    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value">{'Trained' if meta['model_exists'] else 'Pending'}</div>
                <div class="metric-sub">Keras CNN Format</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        classes_count = meta["summary_data"].get("num_classes", "N/A") if meta["summary_data"] else (
            len(meta["eval_data"].get("classes", [])) if meta["eval_data"] else "0"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Classes Covered</div>
                <div class="metric-value">{classes_count}</div>
                <div class="metric-sub">Equipment Types</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        test_acc = f"{meta['eval_data']['test_accuracy'] * 100:.1f}%" if meta["eval_data"] else "Pending"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value" style="color: #38bdf8;">{test_acc}</div>
                <div class="metric-sub">Independent Test Set</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        macro_f1 = f"{meta['eval_data']['macro_avg_f1']:.3f}" if meta["eval_data"] else "Pending"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Macro Avg F1</div>
                <div class="metric-value" style="color: #34d399;">{macro_f1}</div>
                <div class="metric-sub">Harmonic Mean</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🔄 **End-to-End Workflow**")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[Raw Images] --> B[Data Validation & Stratified Split]
            B --> C[Data Augmentation & Normalization]
            C --> D[4-Block CNN Feature Extraction]
            D --> E[Adam Optimization & Early Stopping]
            E --> F[Saved Model & Metadata]
            F --> G[Independent Held-Out Evaluation]
            F --> H[Streamlit Interactive Inference]
        ```
        """
    )

    st.markdown("### 🚀 **Quick Commands**")
    st.code(
        """# 1. Prepare raw dataset with stratified split & leakage checks
python -m src.data_preparation

# 2. Train Convolutional Neural Network with EarlyStopping
python -m src.train --epochs 20

# 3. Evaluate model on held-out test set
python -m src.evaluate

# 4. Launch web application
streamlit run app.py
        """,
        language="powershell"
    )


def render_image_classification():
    """Renders interactive image classification tab."""
    st.markdown("## 🔍 **Equipment Category Inference**")
    st.markdown("Upload a photo of an electrical equipment unit to identify its apparatus class.")

    predictor = get_predictor()
    is_ready, reason = predictor.is_model_ready()

    if not is_ready:
        st.error(f"⚠️ **Model Not Ready:** {reason}")
        st.info("Run `python -m src.train` in your terminal to train and save the model before testing inference.")
        return

    col_upload, col_result = st.columns([1.1, 1.3], gap="large")

    with col_upload:
        st.markdown("#### **Upload Image**")
        uploaded_file = st.file_uploader(
            "Select an image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="equipment_image_uploader"
        )

        # Or sample picker from test directory if available
        test_samples_dir = Path("data/processed/test")
        sample_choice = None
        if test_samples_dir.is_dir():
            sample_files = list(test_samples_dir.rglob("*.png")) + list(test_samples_dir.rglob("*.jpg"))
            if sample_files:
                st.markdown("**— or select a test set sample —**")
                sample_names = ["(None)"] + [f"{f.parent.name}/{f.name}" for f in sample_files[:15]]
                selected_sample = st.selectbox("Choose sample image:", sample_names, key="sample_image_select")
                if selected_sample != "(None)":
                    sample_choice = str(test_samples_dir / selected_sample)

        active_img_source = uploaded_file if uploaded_file is not None else sample_choice

        if active_img_source is not None:
            try:
                preview_img = Image.open(active_img_source)
                st.image(preview_img, caption="Image Preview", use_container_width=True)
                predict_clicked = st.button("⚡ Run Classification", type="primary", use_container_width=True)
            except Exception as e:
                st.error(f"Could not open image: {e}")
                predict_clicked = False
        else:
            predict_clicked = False
            st.info("Upload an electrical apparatus image or choose a sample to begin inference.")

    with col_result:
        st.markdown("#### **Inference Analysis**")
        if predict_clicked and active_img_source is not None:
            with st.spinner("Processing image and executing CNN inference..."):
                try:
                    result = predictor.predict(active_img_source)

                    # Display Prediction Card
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <span class="pred-badge">PREDICTED CATEGORY</span>
                            <div class="pred-class">{format_class_name(result['predicted_class'])}</div>
                            <div class="pred-conf">Confidence: {result['confidence_percentage']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if result["is_low_confidence"]:
                        st.warning(f"⚠️ {result['warning']}")

                    # Class Probabilities Bar Chart
                    st.markdown("##### **Probability Distribution Across Categories**")
                    df_probs = pd.DataFrame(
                        [{"Category": format_class_name(k), "Probability (%)": v * 100} for k, v in result["probabilities"].items()]
                    )
                    st.bar_chart(df_probs.set_index("Category"))

                    st.markdown(
                        f"""
                        <div class="disclaimer-card">
                            <strong>Engineering Scope:</strong> {result['disclaimer']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <h3>Awaiting Image Input</h3>
                    <p>Load an image on the left and click <strong>Run Classification</strong> to inspect predicted category and probability distributions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_model_performance(meta: Dict[str, Any]):
    """Renders actual measured test metrics, confusion matrix, and training curves."""
    st.markdown("## 📊 **Model Performance & Evaluation**")

    if not meta["evaluation_exists"]:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No Evaluation Artifacts Found</h3>
                <p>Independent test metrics have not yet been generated. Run the evaluation script:</p>
                <code>python -m src.evaluate</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    eval_data = meta["eval_data"]

    # KPI summary row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{eval_data['test_accuracy'] * 100:.2f}%")
    c2.metric("Test Loss", f"{eval_data['test_loss']:.4f}")
    c3.metric("Macro Avg F1", f"{eval_data['macro_avg_f1']:.4f}")
    c4.metric("Test Samples", eval_data.get("total_test_samples", "N/A"))

    st.markdown("---")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("#### **Normalized Confusion Matrix**")
        cm_path = Path("results/confusion_matrix.png")
        if cm_path.is_file():
            st.image(str(cm_path), caption="Confusion Matrix on Held-Out Test Set", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with col_right:
        st.markdown("#### **Training & Validation Curves**")
        curves_path = Path("results/learning_curves.png")
        if curves_path.is_file():
            st.image(str(curves_path), caption="Training & Validation Accuracy/Loss Across Epochs", use_container_width=True)
        else:
            st.info("Learning curves plot not found.")

    st.markdown("#### **Per-Class Classification Metrics**")
    cr = eval_data.get("classification_report", {})
    classes = eval_data.get("classes", [])

    rows = []
    for cls in classes:
        if cls in cr:
            rows.append({
                "Equipment Category": format_class_name(cls),
                "Precision": f"{cr[cls]['precision']:.3f}",
                "Recall": f"{cr[cls]['recall']:.3f}",
                "F1-Score": f"{cr[cls]['f1-score']:.3f}",
                "Support (Images)": int(cr[cls]["support"])
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Sample predictions grid
    samples_path = Path("results/sample_predictions.png")
    if samples_path.is_file():
        st.markdown("#### **Qualitative Test Set Predictions**")
        st.image(str(samples_path), caption="Sample Predictions with Ground Truth and Confidence", use_container_width=True)


def render_dataset_information(meta: Dict[str, Any]):
    """Renders dataset distribution, split breakdown, and preprocessing specs."""
    st.markdown("## 📁 **Dataset Information & Preprocessing**")

    if not meta["dataset_exists"]:
        st.markdown(
            """
            <div class="empty-state">
                <h3>Dataset Summary Not Generated</h3>
                <p>Run the automated preparation tool to inspect, validate, and split your dataset:</p>
                <code>python -m src.data_preparation</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    summary = meta["summary_data"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Processed", summary.get("total_split_images", 0))
    c2.metric("Train Split", f"{summary.get('split_counts', {}).get('train', 0)} ({summary.get('split_percentages', {}).get('train', 0):.1f}%)")
    c3.metric("Validation Split", f"{summary.get('split_counts', {}).get('val', 0)} ({summary.get('split_percentages', {}).get('val', 0):.1f}%)")
    c4.metric("Test Split", f"{summary.get('split_counts', {}).get('test', 0)} ({summary.get('split_percentages', {}).get('test', 0):.1f}%)")

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1], gap="medium")

    with col1:
        st.markdown("#### **Class Distribution Across Splits**")
        dist_img = Path("results/class_distribution.png")
        if dist_img.is_file():
            st.image(str(dist_img), caption="Split Counts by Equipment Category", use_container_width=True)
        else:
            # Render as table
            class_counts = summary.get("class_split_counts", {})
            classes = summary.get("classes", [])
            table_data = []
            for cls in classes:
                tr = class_counts.get("train", {}).get(cls, 0)
                va = class_counts.get("val", {}).get(cls, 0)
                te = class_counts.get("test", {}).get(cls, 0)
                table_data.append({
                    "Category": format_class_name(cls),
                    "Train": tr,
                    "Validation": va,
                    "Test": te,
                    "Total": tr + va + te
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    with col2:
        st.markdown("#### **Data Pipeline Specifications**")
        st.markdown(
            """
            - **Input Resolution:** 128 × 128 pixels (RGB, 3 channels)
            - **Pixel Normalization:** Scaled strictly to `[0.0, 1.0]` via `1.0 / 255.0`
            - **Data Leakage Safeguard:** Perceptual MD5 hashing detects duplicate and near-duplicate images across splits.
            - **Data Augmentation (Train only):**
                - Random Horizontal Flip
                - Random Rotation (±10%)
                - Random Zoom (±10%)
                - Random Contrast (±10%)
            - **Validation / Test Pipeline:** Deterministic scaling, strictly no augmentation.
            """
        )

        invalid_count = len(summary.get("invalid_files", []))
        dup_count = len(summary.get("duplicates_detected", []))
        st.markdown(
            f"""
            - **Corrupt / Unreadable Images Skipped:** `{invalid_count}`
            - **Duplicate Images Dropped:** `{dup_count}`
            """
        )


def render_about_project():
    """Renders architectural design, technologies, and engineering boundaries."""
    st.markdown("## ℹ️ **About the EEIC Project**")

    st.markdown(
        """
        ### 🎯 Problem Statement
        In high-voltage substations and electrical distribution networks, accurate visual inventory
        and apparatus categorization are essential for asset management, drone inspection surveys,
        and operational documentation. This system provides automated classification of core electrical
        equipment using deep convolutional neural networks.

        ### 🧠 CNN Architecture
        The classification backbone uses a 4-block deep convolutional network:
        1. **Block 1:** Conv2D (32, 3x3) + BatchNorm + Conv2D (32, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.20)
        2. **Block 2:** Conv2D (64, 3x3) + BatchNorm + Conv2D (64, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.25)
        3. **Block 3:** Conv2D (128, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.30)
        4. **Block 4:** Conv2D (256, 3x3) + BatchNorm + MaxPool(2x2) + Dropout(0.35)
        5. **Classification Head:** GlobalAveragePooling2D + Dense(128, ReLU) + BatchNorm + Dropout(0.35) + Dense(N, Softmax)

        ### 🛠️ Technology Stack
        - **Core Framework:** Python 3.11 & TensorFlow / Keras 2.17
        - **Image Processing:** Pillow & NumPy
        - **Evaluation & Metrics:** scikit-learn, Matplotlib, Seaborn
        - **User Interface:** Streamlit (glassmorphic dark UI)
        - **Test Suite:** pytest

        ### ⚠️ Operational Disclaimer & Safety Boundaries
        - **Equipment Identification Only:** This classifier identifies apparatus categories (e.g., *Transformer*, *Circuit Breaker*, *Insulator*).
        - **No Fault Detection:** This application does **NOT** detect internal electrical faults, corona discharge, SF6 gas leaks, thermal anomalies, or mechanical fatigue.
        - **Human Oversight:** Critical grid operations must always involve certified electrical engineering inspection.
        """
    )


def main():
    meta = load_all_metadata()
    page = render_sidebar(meta)

    if page == "🏠 Home Dashboard":
        render_home_dashboard(meta)
    elif page == "🔍 Image Classification":
        render_image_classification()
    elif page == "📊 Model Performance":
        render_model_performance(meta)
    elif page == "📁 Dataset Information":
        render_dataset_information(meta)
    elif page == "ℹ️ About The Project":
        render_about_project()


if __name__ == "__main__":
    main()

# Vercel Serverless Function Discovery Compatibility
def app(environ, start_response):
    """WSGI callable for Vercel deployment discovery."""
    status = '200 OK'
    headers = [('Content-Type', 'text/plain; charset=utf-8')]
    start_response(status, headers)
    return [b"Electrical Equipment Image Classification Backend Active"]

handler = app
application = app

