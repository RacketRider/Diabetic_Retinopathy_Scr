"""ONNX Runtime inference service for DR classification.

Loads the exported ResNet-101 E1 model and runs inference.
All processing is in-memory — zero disk persistence.
"""

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.config import (
    CLASS_NAMES,
    E1_THRESHOLD,
    INPUT_SIZE,
    LEVEL_DESCRIPTIONS,
    LEVEL_MAP,
    MODEL_PATH,
    REFERABLE_CLASSES,
)
from app.services.preprocessing import pil_to_numpy, preprocess_fundus

logger = logging.getLogger(__name__)

# Module-level session — loaded once, shared across requests
_session: ort.InferenceSession | None = None


def load_model(model_path: str | None = None) -> bool:
    """Load the ONNX model at startup. Returns True if successful."""
    global _session
    path = model_path or MODEL_PATH

    if not Path(path).exists():
        logger.warning("Model file not found: %s", path)
        return False

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    _session = ort.InferenceSession(path, opts, providers=providers)
    inp = _session.get_inputs()[0]
    out = _session.get_outputs()[0]
    logger.info(
        "Model loaded: %s | Input: %s %s | Output: %s %s | Providers: %s",
        path, inp.name, inp.shape, out.name, out.shape, _session.get_providers(),
    )
    return True


def is_model_loaded() -> bool:
    """Check if the ONNX model is loaded and ready."""
    return _session is not None


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def predict(image_np: np.ndarray) -> dict:
    """Run DR classification on a raw RGB uint8 image.

    Args:
        image_np: RGB uint8 numpy array of any size.

    Returns:
        Dict with grading, confidence, quality, and timing info.
    """
    if _session is None:
        raise RuntimeError("Model not loaded")

    t0 = time.perf_counter()

    # Preprocess: crop FOV → pad to square → resize → normalize
    preprocessed = preprocess_fundus(image_np, target_size=INPUT_SIZE)

    # ONNX expects NCHW float32: (1, 3, H, W)
    input_tensor = np.transpose(preprocessed, (2, 0, 1))  # HWC → CHW
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    # Run inference
    input_name = _session.get_inputs()[0].name
    outputs = _session.run(None, {input_name: input_tensor})
    raw = outputs[0][0]

    # Softmax probabilities
    probs = _softmax(raw)

    # Build response
    class_probs = {name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)}
    referable_prob = sum(class_probs[c] for c in REFERABLE_CLASSES)
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    level = LEVEL_MAP[predicted_class]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "grading": {
            "level": level,
            "label": predicted_class,
            "description": LEVEL_DESCRIPTIONS[level],
            "is_referable": referable_prob >= E1_THRESHOLD,
        },
        "confidence": {
            "calibrated_score": round(referable_prob, 4),
            "class_probabilities": class_probs,
            "referable_probability": round(referable_prob, 4),
            "threshold_used": E1_THRESHOLD,
        },
        "processing_time_ms": round(elapsed_ms, 1),
    }
