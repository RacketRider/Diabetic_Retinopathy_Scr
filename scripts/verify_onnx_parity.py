"""Verify that ONNX model output matches MATLAB predictions.

Usage:
    python scripts/verify_onnx_parity.py [--model PATH] [--image PATH]

Compares ONNX Runtime output against known MATLAB predictions.
Acceptable tolerance: max absolute difference < 1e-3.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.services.preprocessing import preprocess_fundus

DEFAULT_MODEL = "models/dr_resnet101_e1.onnx"
CLASS_NAMES = ["Severe", "No_DR", "Mild", "Proliferate_DR", "Moderate"]


def load_and_preprocess(image_path: str, target_size: int = 448) -> np.ndarray:
    """Load image and run through the Python preprocessing pipeline."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    preprocessed = preprocess_fundus(img_array, target_size=target_size)
    # HWC -> NCHW
    tensor = np.transpose(preprocessed, (2, 0, 1))[np.newaxis, ...]
    return tensor.astype(np.float32)


def run_inference(model_path: str, input_tensor: np.ndarray) -> np.ndarray:
    """Run ONNX inference and return raw output."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, opts, providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    print(f"Model input:  {input_name} {input_shape}")
    print(f"Model output: {session.get_outputs()[0].name} {output_shape}")

    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0][0]


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def main():
    parser = argparse.ArgumentParser(description="Verify ONNX model parity")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=str, required=True,
                        help="Path to a test fundus image")
    parser.add_argument("--matlab-probs", type=str, default=None,
                        help="Optional: comma-separated MATLAB probabilities for comparison")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    print(f"\n{'='*60}")
    print(f"ONNX Parity Verification")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"{'='*60}\n")

    tensor = load_and_preprocess(args.image)
    print(f"Preprocessed tensor shape: {tensor.shape}")
    print(f"Value range: [{tensor.min():.4f}, {tensor.max():.4f}]\n")

    raw_output = run_inference(args.model, tensor)
    probs = softmax(raw_output)

    print(f"\nRaw logits: {raw_output}")
    print(f"\nClass probabilities:")
    for name, prob in zip(CLASS_NAMES, probs):
        marker = " ← predicted" if prob == probs.max() else ""
        print(f"  {name:20s}: {prob:.4f}{marker}")

    predicted = CLASS_NAMES[int(np.argmax(probs))]
    referable_prob = sum(probs[i] for i, c in enumerate(CLASS_NAMES)
                         if c in {"Moderate", "Severe", "Proliferate_DR"})
    print(f"\nPredicted class: {predicted}")
    print(f"Referable probability: {referable_prob:.4f}")
    print(f"Referable (threshold 0.164): {'YES' if referable_prob >= 0.164 else 'NO'}")

    if args.matlab_probs:
        matlab = np.array([float(x) for x in args.matlab_probs.split(",")])
        diff = np.abs(probs - matlab)
        max_diff = diff.max()
        print(f"\n--- MATLAB Comparison ---")
        print(f"MATLAB probs: {matlab}")
        print(f"Max absolute difference: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("✅ PASS: ONNX output matches MATLAB within 1e-3 tolerance")
        else:
            print("❌ FAIL: ONNX output diverges from MATLAB beyond 1e-3")
            sys.exit(1)
    else:
        print("\n⚠️  No MATLAB reference provided — run with --matlab-probs to verify parity")


if __name__ == "__main__":
    main()
