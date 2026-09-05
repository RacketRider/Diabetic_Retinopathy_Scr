"""INT8 dynamic quantization for ONNX model — reduces ~175 MB → ~44 MB.

Usage:
    python scripts/quantize_model.py [--input PATH] [--output PATH]
"""
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

DEFAULT_INPUT = "models/dr_resnet101_e1.onnx"
DEFAULT_OUTPUT = "models/dr_resnet101_e1_int8.onnx"


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"Input ONNX model path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output quantized model path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    print(f"Quantizing {input_path} → {output_path}")
    print(f"Input size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )

    output_size = output_path.stat().st_size / 1024 / 1024
    input_size = input_path.stat().st_size / 1024 / 1024
    ratio = (1 - output_size / input_size) * 100
    print(f"Output size: {output_size:.1f} MB ({ratio:.0f}% reduction)")
    print("\n⚠️  Verify AUC on validation set before deploying.")


if __name__ == "__main__":
    main()
