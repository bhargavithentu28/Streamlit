"""
Data Preparation Pipeline for Electrical Equipment Image Classification.

Validates images, detects duplicates and corrupt files, performs stratified splitting
(train, validation, test), outputs class mappings, and produces preparation reports.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont

from src.utils import (
    load_config,
    save_json,
    set_seed,
    ensure_dir,
    setup_logger,
    validate_image_file,
    compute_image_hash
)

logger = setup_logger("eeic.data_prep")


def inspect_raw_dataset(
    raw_dir: Path,
    supported_formats: List[str]
) -> Tuple[Dict[str, List[Path]], List[Tuple[Path, str]]]:
    """
    Scans the raw directory for class subfolders, collects valid images,
    and records invalid/corrupted files.
    """
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    # Discover candidate class folders (ignore hidden directories)
    class_folders = [
        d for d in raw_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ("train", "val", "test", "processed")
    ]

    valid_class_files: Dict[str, List[Path]] = defaultdict(list)
    invalid_files: List[Tuple[Path, str]] = []

    for folder in sorted(class_folders, key=lambda d: d.name.lower()):
        class_name = folder.name
        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue
            is_valid, reason = validate_image_file(str(file_path), supported_formats)
            if is_valid:
                valid_class_files[class_name].append(file_path)
            else:
                invalid_files.append((file_path, reason or "Unknown validation failure"))

    return dict(valid_class_files), invalid_files


def deduplicate_images(
    class_images: Dict[str, List[Path]]
) -> Tuple[Dict[str, List[Path]], List[Tuple[Path, Path]]]:
    """
    Detects identical/near-identical images across or within classes using perceptual hashing
    to prevent data leakage across train/val/test splits.
    """
    seen_hashes: Dict[str, Path] = {}
    deduped_class_images: Dict[str, List[Path]] = defaultdict(list)
    duplicates_detected: List[Tuple[Path, Path]] = []

    for class_name, files in class_images.items():
        for file_path in files:
            img_hash = compute_image_hash(str(file_path))
            if img_hash is None:
                continue
            if img_hash in seen_hashes:
                duplicates_detected.append((file_path, seen_hashes[img_hash]))
                logger.warning(
                    f"Duplicate detected: {file_path.name} is identical/near-duplicate of {seen_hashes[img_hash].name}"
                )
            else:
                seen_hashes[img_hash] = file_path
                deduped_class_images[class_name].append(file_path)

    return dict(deduped_class_images), duplicates_detected


def perform_stratified_split(
    class_images: Dict[str, List[Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Performs stratified train/val/test split across all classes.
    Returns: { "train": {class: [paths]}, "val": {class: [paths]}, "test": {class: [paths]} }
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0; got {total_ratio:.3f}")

    splits: Dict[str, Dict[str, List[Path]]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list)
    }

    test_and_val_ratio = val_ratio + test_ratio

    for class_name, files in class_images.items():
        n_samples = len(files)
        if n_samples == 0:
            continue
        elif n_samples < 3:
            # Not enough samples to split all 3 ways; put all in train with warning
            logger.warning(
                f"Class '{class_name}' has only {n_samples} samples. Assigning all to train split."
            )
            splits["train"][class_name].extend(files)
            continue

        # First split into train vs (val + test)
        try:
            train_files, val_test_files = train_test_split(
                files,
                test_size=test_and_val_ratio,
                random_state=seed,
                shuffle=True
            )
        except Exception as e:
            logger.warning(f"Stratified split error on class {class_name}: {e}. Falling back to indexed split.")
            n_train = max(1, int(n_samples * train_ratio))
            train_files = files[:n_train]
            val_test_files = files[n_train:]

        splits["train"][class_name].extend(train_files)

        # Second split: val vs test
        if len(val_test_files) >= 2:
            val_prop = val_ratio / test_and_val_ratio
            val_files, test_files = train_test_split(
                val_test_files,
                train_size=val_prop,
                random_state=seed,
                shuffle=True
            )
            splits["val"][class_name].extend(val_files)
            splits["test"][class_name].extend(test_files)
        elif len(val_test_files) == 1:
            splits["val"][class_name].extend(val_test_files)

    return {
        split_name: dict(class_dict)
        for split_name, class_dict in splits.items()
    }


def copy_split_files(
    splits: Dict[str, Dict[str, List[Path]]],
    output_dir: Path
) -> None:
    """Copies split files to output_dir/split_name/class_name/."""
    for split_name, class_dict in splits.items():
        for class_name, file_paths in class_dict.items():
            target_class_dir = output_dir / split_name / class_name
            ensure_dir(str(target_class_dir))
            for p in file_paths:
                dest = target_class_dir / p.name
                # Avoid collision if filenames share names
                if dest.exists() and dest.resolve() != p.resolve():
                    dest = target_class_dir / f"{p.stem}_{p.stat().st_mtime_ns}{p.suffix}"
                shutil.copy2(p, dest)


def generate_preparation_report(
    summary: Dict[str, Any],
    report_path: Path
) -> None:
    """Writes a comprehensive Markdown report of dataset preparation."""
    md = [
        "# Dataset Preparation Report",
        "",
        f"**Date:** {summary['timestamp']}",
        f"**Random Seed:** {summary['seed']}",
        f"**Raw Directory:** `{summary['raw_dir']}`",
        f"**Processed Directory:** `{summary['processed_dir']}`",
        "",
        "## Summary Metrics",
        "",
        f"- **Total Raw Images Discovered:** {summary['total_raw_images']}",
        f"- **Valid Images Processed:** {summary['total_valid_images']}",
        f"- **Corrupt/Invalid Images Skipped:** {len(summary['invalid_files'])}",
        f"- **Duplicate Images Discarded (Leakage Prevention):** {len(summary['duplicates_detected'])}",
        f"- **Identified Equipment Classes ({len(summary['classes'])}):** {', '.join(summary['classes'])}",
        "",
        "## Split Distribution",
        "",
        "| Split | Count | Percentage |",
        "| :--- | :--- | :--- |",
        f"| Train | {summary['split_counts']['train']} | {summary['split_percentages']['train']:.1f}% |",
        f"| Validation | {summary['split_counts']['val']} | {summary['split_percentages']['val']:.1f}% |",
        f"| Test | {summary['split_counts']['test']} | {summary['split_percentages']['test']:.1f}% |",
        f"| **Total** | **{summary['total_split_images']}** | **100.0%** |",
        "",
        "## Class Distribution Across Splits",
        "",
        "| Class Name | Train | Validation | Test | Total |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for cls in summary["classes"]:
        tr = summary["class_split_counts"]["train"].get(cls, 0)
        va = summary["class_split_counts"]["val"].get(cls, 0)
        te = summary["class_split_counts"]["test"].get(cls, 0)
        tot = tr + va + te
        md.append(f"| {cls} | {tr} | {va} | {te} | {tot} |")

    if summary["invalid_files"]:
        md.extend([
            "",
            "## Invalid / Unreadable Files",
            "",
            "| File Path | Reason |",
            "| :--- | :--- |"
        ])
        for fpath, reason in summary["invalid_files"][:20]:
            md.append(f"| `{fpath}` | {reason} |")
        if len(summary["invalid_files"]) > 20:
            md.append(f"| ... and {len(summary['invalid_files']) - 20} more | |")

    if summary["duplicates_detected"]:
        md.extend([
            "",
            "## Duplicate Files Skipped",
            "",
            "| Duplicate File | Original Match |",
            "| :--- | :--- |"
        ])
        for dup, orig in summary["duplicates_detected"][:15]:
            md.append(f"| `{dup}` | `{orig}` |")

    md.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def create_synthetic_dataset(
    output_raw_dir: Path,
    classes: Optional[List[str]] = None,
    samples_per_class: int = 15,
    img_size: Tuple[int, int] = (128, 128)
) -> None:
    """
    Creates a small, deterministic synthetic dataset of electrical equipment silhouettes / patterns
    strictly for testing pipeline mechanics, automated unit/integration tests, and smoke testing.
    Never used as real-world classification benchmarks.
    """
    if classes is None:
        classes = [
            "transformer",
            "circuit_breaker",
            "insulator",
            "disconnect_switch",
            "current_transformer",
            "voltage_transformer"
        ]

    distinct_colors = [
        (70, 130, 180), (220, 20, 60), (34, 139, 34), (218, 165, 32),
        (138, 43, 226), (255, 140, 0), (20, 180, 200), (200, 50, 150)
    ]

    palette = {
        "transformer": (70, 130, 180),
        "circuit_breaker": (220, 20, 60),
        "insulator": (34, 139, 34),
        "disconnect_switch": (218, 165, 32),
        "current_transformer": (138, 43, 226),
        "voltage_transformer": (255, 140, 0)
    }

    logger.info(f"Generating synthetic pipeline dataset in {output_raw_dir} ({len(classes)} classes)...")

    for class_idx, class_name in enumerate(classes):
        class_dir = output_raw_dir / class_name
        ensure_dir(str(class_dir))
        base_color = palette.get(class_name, distinct_colors[class_idx % len(distinct_colors)])

        for idx in range(samples_per_class):
            bg_lum = 230 + (idx * 3) % 25
            img = Image.new("RGB", img_size, color=(bg_lum, bg_lum, bg_lum))
            draw = ImageDraw.Draw(img)

            # Draw a distinct shape per class with slight variation
            delta = (idx * 2) % 10
            if "transformer" in class_name and "current" not in class_name and "voltage" not in class_name:
                # Big block with cooling fins
                draw.rectangle([25 + delta, 35, 95 - delta, 95], fill=base_color, outline=(20, 20, 20), width=2)
                for f_x in range(30, 90, 10):
                    draw.line([(f_x, 35), (f_x, 95)], fill=(200, 200, 200), width=1)
                # Bushing on top
                draw.polygon([(55, 35), (60, 15), (65, 35)], fill=(200, 100, 50))
            elif "circuit_breaker" in class_name:
                # Tall tank / pole structure
                draw.rectangle([45, 20 + delta, 83, 110 - delta], fill=base_color, outline=(20, 20, 20), width=2)
                draw.ellipse([52, 45 + delta, 76, 75 + delta], fill=(255, 255, 255))
            elif "insulator" in class_name:
                # Corrugated disc stacks
                for y in range(25, 100, 14):
                    draw.ellipse([35 - (delta % 4), y, 93 + (delta % 4), y + 10], fill=base_color, outline=(0, 0, 0))
            elif "disconnect_switch" in class_name:
                # Diagonal blade open/closed
                draw.line([(30, 90), (95, 35 + delta)], fill=base_color, width=5)
                draw.rectangle([25, 85, 35, 95], fill=(50, 50, 50))
                draw.rectangle([90, 30 + delta, 100, 40 + delta], fill=(50, 50, 50))
            elif "current_transformer" in class_name:
                # Ring / Toroid core
                draw.ellipse([28 + delta, 28 + delta, 100 - delta, 100 - delta], fill=base_color, outline=(0, 0, 0), width=3)
                draw.ellipse([48, 48, 80, 80], fill=(240, 240, 245), outline=(0, 0, 0), width=2)
            elif "voltage_transformer" in class_name:
                # Voltage transformer (pedestal with bushing)
                draw.rectangle([40 + delta, 60, 88 - delta, 105], fill=base_color, outline=(0, 0, 0), width=2)
                draw.line([(64, 60), (64, 25)], fill=(120, 120, 120), width=4)
                draw.ellipse([56, 18, 72, 30], fill=(255, 200, 50))
            else:
                # Generic distinct geometric marker for custom class
                x1 = 20 + (class_idx * 12 + delta) % 35
                y1 = 20 + (idx * 5) % 25
                draw.rectangle([x1, y1, x1 + 50, y1 + 50], fill=base_color, outline=(0, 0, 0), width=2)
                draw.polygon([(x1 + 10, y1 + 10), (x1 + 40, y1 + 10), (x1 + 25, y1 + 40)], fill=(255, 255, 255))

            # Add unique text tag inside image
            draw.text((4, 4), f"{class_name}_{idx}", fill=(0, 0, 0))

            img_path = class_dir / f"{class_name}_{idx+1:03d}.png"
            img.save(img_path, format="PNG")


def prepare_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    config_path: str = "configs/config.json",
    force_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for dataset preparation.
    Reads config, scans raw directory, handles deduplication, performs stratified splitting,
    copies files, writes class_mapping.json, and outputs summary reports.
    """
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    supported_formats = cfg.get("supported_formats", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    split_ratio = cfg.get("split_ratio", {"train": 0.70, "val": 0.15, "test": 0.15})

    train_ratio = split_ratio.get("train", 0.70)
    val_ratio = split_ratio.get("val", 0.15)
    test_ratio = split_ratio.get("test", 0.15)

    # Check if raw_path has any valid class folders
    needs_synthetic = force_synthetic
    if not raw_path.exists() or not any(raw_path.iterdir()):
        needs_synthetic = True

    if needs_synthetic:
        logger.info(f"No existing raw dataset found in {raw_path}. Generating synthetic dataset for verification...")
        create_synthetic_dataset(raw_path, samples_per_class=16)

    # Step 1: Scan and validate images
    valid_class_files, invalid_files = inspect_raw_dataset(raw_path, supported_formats)

    if not valid_class_files:
        raise ValueError(
            f"No valid images found in {raw_path}. Please provide subdirectories with valid image files."
        )

    total_raw_found = sum(len(f) for f in valid_class_files.values()) + len(invalid_files)
    logger.info(f"Discovered {total_raw_found} raw files across {len(valid_class_files)} classes.")

    # Step 2: Deduplication to prevent leakage
    deduped_class_images, duplicates_detected = deduplicate_images(valid_class_files)
    total_valid = sum(len(f) for f in deduped_class_images.values())

    # Step 3: Class mapping
    classes = sorted(list(deduped_class_images.keys()))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

    class_mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "classes": classes,
        "num_classes": len(classes)
    }

    # Step 4: Stratified split
    splits = perform_stratified_split(
        deduped_class_images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Step 5: Clean output directory and copy files
    for sub in ["train", "val", "test"]:
        sub_dir = processed_path / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)

    copy_split_files(splits, processed_path)

    # Count splits
    split_counts = {
        s: sum(len(f) for f in splits[s].values())
        for s in ["train", "val", "test"]
    }
    total_split_images = sum(split_counts.values())

    class_split_counts: Dict[str, Dict[str, int]] = {
        s: {cls: len(splits[s].get(cls, [])) for cls in classes}
        for s in ["train", "val", "test"]
    }

    split_percentages = {
        s: (split_counts[s] / total_split_images * 100.0) if total_split_images > 0 else 0.0
        for s in ["train", "val", "test"]
    }

    # Step 6: Save class mapping and metadata
    # Save in both processed_path and models_dir for inference convenience
    ensure_dir(str(processed_path))
    save_json(class_mapping, str(processed_path / "class_mapping.json"))

    model_dir = Path(cfg.get("paths", {}).get("model_dir", "models"))
    ensure_dir(str(model_dir))
    save_json(class_mapping, str(model_dir / "class_mapping.json"))

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "raw_dir": str(raw_path.resolve()),
        "processed_dir": str(processed_path.resolve()),
        "seed": seed,
        "classes": classes,
        "num_classes": len(classes),
        "total_raw_images": total_raw_found,
        "total_valid_images": total_valid,
        "total_split_images": total_split_images,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "split_counts": split_counts,
        "split_percentages": split_percentages,
        "class_split_counts": class_split_counts,
        "invalid_files": [(str(f), r) for f, r in invalid_files],
        "duplicates_detected": [(str(d), str(o)) for d, o in duplicates_detected]
    }

    summary_file = processed_path / "dataset_summary.json"
    save_json(summary, str(summary_file))

    report_file = processed_path / "dataset_preparation_report.md"
    generate_preparation_report(summary, report_file)

    logger.info(
        f"Data preparation complete! Total: {total_split_images} images "
        f"(Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']})."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Electrical Equipment Image Classification")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to processed dataset directory")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to configuration file")
    parser.add_argument("--synthetic", action="store_true", help="Force generation of synthetic dataset")

    args = parser.parse_args()
    prepare_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        config_path=args.config,
        force_synthetic=args.synthetic
    )


if __name__ == "__main__":
    main()
