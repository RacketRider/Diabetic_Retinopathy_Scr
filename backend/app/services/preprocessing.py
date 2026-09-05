"""Fundus image preprocessing pipeline.

Python port of MATLAB functions: cropFundusFOV, padToSquare, preprocessFundus.
All operations are in-memory — no disk I/O.
"""

import cv2
import numpy as np
from PIL import Image


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy RGB array."""
    return np.array(img, dtype=np.uint8)


def crop_fundus_fov(image: np.ndarray, threshold: int = 15) -> tuple[np.ndarray, dict]:
    """Port of MATLAB cropFundusFOV — detect circular FOV and crop.

    Finds the bounding box of all pixels brighter than `threshold`
    (which removes the black background around the circular fundus).
    """
    gray = image.mean(axis=2).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    mask = gray > threshold

    if not np.any(mask):
        return image, {"success": False, "reason": "no_fov_detected"}

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    cropped = image[rmin : rmax + 1, cmin : cmax + 1]
    return cropped, {
        "success": True,
        "bbox": [cmin, rmin, cmax - cmin, rmax - rmin],
    }


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Port of MATLAB padToSquare — zero-pad to square aspect ratio."""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    if image.ndim == 3:
        padded = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((size, size), dtype=image.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w] = image
    return padded


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalization for illumination normalization.

    Applied on the L channel of LAB color space to preserve color.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus(image: np.ndarray, target_size: int = 448, enhance: bool = False) -> np.ndarray:
    """Full preprocessing pipeline matching MATLAB preprocessFundus.

    Args:
        image: RGB uint8 numpy array
        target_size: Output spatial dimension (default 448 for E1)
        enhance: If True, apply CLAHE for borderline-quality images

    Returns:
        float32 array of shape (target_size, target_size, 3) in [0, 1]
    """
    cropped, meta = crop_fundus_fov(image)
    if meta["success"]:
        squared = pad_to_square(cropped)
    else:
        squared = pad_to_square(image)

    if enhance:
        squared = apply_clahe(squared)

    resized = cv2.resize(
        squared, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
    )

    # Normalize to [0, 1] float32 — matches MATLAB im2single / single() conversion
    normalized = resized.astype(np.float32) / 255.0
    return normalized
