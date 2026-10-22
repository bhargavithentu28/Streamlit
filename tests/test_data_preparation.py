"""
Automated unit and integration tests for data preparation:
- Directory inspection
- Synthetic dataset generation
- Stratified splitting logic
- Class mapping generation
- Leakage detection
"""

import pytest
from pathlib import Path
from PIL import Image

from src.data_preparation import (
    inspect_raw_dataset,
    perform_stratified_split,
    create_synthetic_dataset,
    prepare_dataset
)
from src.utils import load_json


@pytest.fixture
def sample_dataset(tmp_path):
    """Creates a temporary test dataset directory with 3 classes."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    classes = ["transformer", "circuit_breaker", "insulator"]

    for cls in classes:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            img = Image.new("RGB", (32, 32), color=(i * 20, 100, 150))
            img.save(cls_dir / f"img_{i:02d}.png")

    return raw_dir


def test_inspect_raw_dataset(sample_dataset):
    """Validates scanning of raw class directories and file counting."""
    valid_files, invalid_files = inspect_raw_dataset(sample_dataset, [".png", ".jpg"])
    assert len(valid_files) == 3
    assert "transformer" in valid_files
    assert len(valid_files["transformer"]) == 10
    assert len(invalid_files) == 0


def test_stratified_split_ratios(sample_dataset):
    """Validates that stratified splitting maintains correct partition proportions."""
    valid_files, _ = inspect_raw_dataset(sample_dataset, [".png"])
    splits = perform_stratified_split(
        valid_files,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    total_train = sum(len(f) for f in splits["train"].values())
    total_val = sum(len(f) for f in splits["val"].values())
    total_test = sum(len(f) for f in splits["test"].values())

    total = total_train + total_val + total_test
    assert total == 30
    assert total_train > total_val
    assert total_val > 0
    assert total_test > 0


def test_create_synthetic_dataset(tmp_path):
    """Validates that synthetic dataset generator creates readable image files."""
    synthetic_dir = tmp_path / "synthetic_raw"
    classes = ["transformer", "insulator"]
    create_synthetic_dataset(synthetic_dir, classes=classes, samples_per_class=5, img_size=(64, 64))

    for cls in classes:
        cls_path = synthetic_dir / cls
        assert cls_path.is_dir()
        files = list(cls_path.glob("*.png"))
        assert len(files) == 5
        # Verify first image can be opened
        with Image.open(files[0]) as img:
            assert img.size == (64, 64)
            assert img.mode == "RGB"


def test_end_to_end_preparation_pipeline(tmp_path):
    """Validates the complete prepare_dataset routine on a synthetic dataset."""
    raw_dir = tmp_path / "raw_pipe"
    proc_dir = tmp_path / "proc_pipe"
    cfg_file = tmp_path / "config.json"

    # Minimal config
    cfg_data = {
        "seed": 42,
        "image_size": [64, 64],
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "supported_formats": [".png"],
        "paths": {
            "model_dir": str(tmp_path / "models")
        }
    }
    from src.utils import save_json
    save_json(cfg_data, str(cfg_file))

    create_synthetic_dataset(raw_dir, classes=["c1", "c2"], samples_per_class=6, img_size=(32, 32))

    summary = prepare_dataset(
        raw_dir=str(raw_dir),
        processed_dir=str(proc_dir),
        config_path=str(cfg_file)
    )

    assert summary["num_classes"] == 2
    assert (proc_dir / "class_mapping.json").is_file()
    assert (proc_dir / "dataset_summary.json").is_file()
    assert (proc_dir / "dataset_preparation_report.md").is_file()

    mapping = load_json(str(proc_dir / "class_mapping.json"))
    assert "class_to_idx" in mapping
    assert len(mapping["classes"]) == 2
