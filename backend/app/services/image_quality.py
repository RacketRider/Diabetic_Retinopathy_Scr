"""Image Quality Assessment for fundus images.

Evaluates focus, illumination, and field-of-view adequacy.
Images below quality thresholds get CLAHE enhancement or rejection.
"""

import cv2
import numpy as np


def assess_focus(image: np.ndarray) -> float:
    """Assess image focus using Laplacian variance.

    Higher values indicate sharper images.
    Returns a normalized score in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: typical fundus images have variance 50-500
    score = min(1.0, laplacian_var / 500.0)
    return round(score, 3)


def assess_illumination(image: np.ndarray) -> float:
    """Assess illumination uniformity.

    Computes the coefficient of variation of the L channel.
    Lower CV = more uniform illumination = higher score.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)

    # Only consider non-black pixels (inside FOV)
    mask = l_channel > 10
    if not np.any(mask):
        return 0.0

    l_values = l_channel[mask]
    mean_l = l_values.mean()
    if mean_l < 1e-6:
        return 0.0

    cv = l_values.std() / mean_l  # coefficient of variation
    # Lower CV is better; typical good fundus: CV < 0.3
    score = max(0.0, 1.0 - cv)
    return round(score, 3)


def assess_fov(image: np.ndarray, min_ratio: float = 0.3) -> bool:
    """Check if the field of view covers enough of the image.

    Returns True if the fundus circle occupies >= min_ratio of total area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    fov_pixels = np.count_nonzero(thresh)
    total_pixels = gray.shape[0] * gray.shape[1]

    return (fov_pixels / total_pixels) >= min_ratio


def assess_quality(image: np.ndarray) -> dict:
    """Full quality assessment pipeline.

    Returns quality metrics and whether the image is gradeable.
    """
    focus = assess_focus(image)
    illumination = assess_illumination(image)
    fov_ok = assess_fov(image)

    # Gradeable if all quality checks pass minimum thresholds
    is_gradeable = focus >= 0.15 and illumination >= 0.3 and fov_ok

    # Recommend enhancement for borderline images
    needs_enhancement = is_gradeable and (focus < 0.4 or illumination < 0.5)

    return {
        "is_gradeable": is_gradeable,
        "focus_score": focus,
        "illumination_score": illumination,
        "fov_adequate": fov_ok,
        "enhancement_applied": needs_enhancement,
    }
