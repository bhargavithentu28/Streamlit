"""
Automated unit tests for validation utilities:
- Image format validation
- Corrupted file detection
- Image content hashing for data leakage prevention
- Missing file handling
"""

import os
import pytest
from pathlib import Path
from PIL import Image

from src.utils import validate_image_file, compute_image_hash, ensure_dir


@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture providing temporary test directory."""
    test_dir = tmp_path / "val_tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def test_valid_image(temp_test_dir):
    """Validates that a correctly formatted PNG image passes check."""
    valid_path = temp_test_dir / "valid_test.png"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(valid_path, format="PNG")

    is_valid, reason = validate_image_file(str(valid_path), [".jpg", ".jpeg", ".png"])
    assert is_valid is True
    assert reason is None


def test_unsupported_extension(temp_test_dir):
    """Validates rejection of unsupported file extension."""
    txt_path = temp_test_dir / "dummy.txt"
    txt_path.write_text("not an image")

    is_valid, reason = validate_image_file(str(txt_path), [".jpg", ".png"])
    assert is_valid is False
    assert "Unsupported file extension" in reason


def test_corrupt_image(temp_test_dir):
    """Validates that a corrupted image file (truncated bytes) is detected and rejected."""
    corrupt_path = temp_test_dir / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")  # Incomplete header

    is_valid, reason = validate_image_file(str(corrupt_path), [".jpg", ".jpeg", ".png"])
    assert is_valid is False
    assert "Corrupted or invalid" in reason


def test_missing_file_detection(temp_test_dir):
    """Validates handling of non-existent image paths."""
    missing_path = temp_test_dir / "does_not_exist.png"
    is_valid, reason = validate_image_file(str(missing_path))
    assert is_valid is False
    assert "File does not exist" in reason


def test_image_hashing_for_duplicate_detection(temp_test_dir):
    """Validates that identical images produce identical hashes, and distinct images produce distinct hashes."""
    img1_path = temp_test_dir / "img1.png"
    img2_path = temp_test_dir / "img2.png"
    diff_path = temp_test_dir / "diff.png"

    img_a = Image.new("RGB", (64, 64), color="blue")
    img_a.save(img1_path)
    img_a.save(img2_path)

    img_b = Image.new("RGB", (64, 64), color="yellow")
    img_b.save(diff_path)

    hash1 = compute_image_hash(str(img1_path))
    hash2 = compute_image_hash(str(img2_path))
    hash_diff = compute_image_hash(str(diff_path))

    assert hash1 is not None
    assert hash1 == hash2
    assert hash1 != hash_diff
