"""Preprocessing pipeline tests.

Mirrors the synthetic checks from MATLAB tests/run_tests.m.
"""

import numpy as np
import pytest

from app.services.preprocessing import crop_fundus_fov, pad_to_square, preprocess_fundus


def test_crop_fundus_fov_synthetic():
    """Test with a synthetic circular FOV (matches MATLAB run_tests.m)."""
    # Create 100x120 image with a circle at center (60, 50) radius 40
    yy, xx = np.mgrid[0:100, 0:120]
    mask = ((xx - 60) ** 2 + (yy - 50) ** 2) <= 40 ** 2
    synthetic = np.zeros((100, 120, 3), dtype=np.uint8)
    synthetic[mask] = 100

    cropped, meta = crop_fundus_fov(synthetic)
    assert meta["success"] is True
    assert cropped.shape[0] >= 80
    assert cropped.shape[1] >= 80


def test_crop_fundus_fov_black_image():
    """All-black image should fail FOV detection."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped, meta = crop_fundus_fov(black)
    assert meta["success"] is False


def test_pad_to_square():
    """Non-square image should be padded to square."""
    rect = np.ones((100, 200, 3), dtype=np.uint8) * 128
    squared = pad_to_square(rect)
    assert squared.shape[0] == squared.shape[1] == 200


def test_pad_to_square_already_square():
    """Already-square image should be unchanged."""
    sq = np.ones((100, 100, 3), dtype=np.uint8)
    result = pad_to_square(sq)
    assert result.shape == (100, 100, 3)
    assert np.array_equal(result, sq)


def test_preprocess_fundus():
    """Full pipeline should produce (target, target, 3) float32 in [0, 1]."""
    # Synthetic circular fundus
    yy, xx = np.mgrid[0:100, 0:120]
    mask = ((xx - 60) ** 2 + (yy - 50) ** 2) <= 40 ** 2
    synthetic = np.zeros((100, 120, 3), dtype=np.uint8)
    synthetic[mask] = 100

    processed = preprocess_fundus(synthetic, target_size=64)
    assert processed.shape == (64, 64, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min()
    assert processed.max() <= 1.0


def test_preprocess_fundus_default_size():
    """Default target size should be 448."""
    img = np.random.randint(20, 200, (300, 400, 3), dtype=np.uint8)
    processed = preprocess_fundus(img)
    assert processed.shape == (448, 448, 3)
