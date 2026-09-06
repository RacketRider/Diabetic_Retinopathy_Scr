"""Export MATLAB ResNet-101 (V3_E1_HighResFOV448) weights to ONNX and PyTorch.

Reconstructs the ResNet v1 architecture, loads the exported MATLAB parameters,
verifies mathematical parity against MATLAB predictions, and exports to ONNX.
"""

import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_WEIGHTS = PROJECT_ROOT / "models" / "e1_raw_weights.mat"
OUTPUT_ONNX = PROJECT_ROOT / "models" / "dr_resnet101_e1.onnx"
OUTPUT_PT = PROJECT_ROOT / "models" / "dr_resnet101_e1.pt"


class BottleneckV1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        # ResNet v1 (Caffe / MATLAB): stride is applied at conv1 (1x1)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet101V1(nn.Module):
    """Exact PyTorch replica of MATLAB dlnetwork ResNet-101."""

    def __init__(self, num_classes=5):
        super().__init__()
        # Channel mean normalization (R=0.381954, G=0.264720, B=0.186135)
        self.register_buffer(
            "input_mean",
            torch.tensor([0.381954, 0.264720, 0.186135], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        # pool1 padding: [0, 1, 0, 1] in MATLAB [top, bottom, left, right] -> (left, right, top, bottom) in PyTorch
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inplanes = 64
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 23, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4, eps=1e-5),
            )
        layers = [BottleneckV1(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(BottleneckV1(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Subtract mean
        x = x - self.input_mean
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(self.pad(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_weights(model: ResNet101V1, mat_path: str | Path):
    data = sio.loadmat(str(mat_path))

    # Conv1 and BN1
    model.conv1.weight.data = torch.from_numpy(
        data["conv1__Weights"].transpose(3, 2, 0, 1)
    )
    model.bn1.weight.data = torch.from_numpy(data["bn_conv1__Scale"].flatten())
    model.bn1.bias.data = torch.from_numpy(data["bn_conv1__Offset"].flatten())
    model.bn1.running_mean.data = torch.from_numpy(
        data["bn_conv1__TrainedMean"].flatten()
    )
    model.bn1.running_var.data = torch.from_numpy(
        data["bn_conv1__TrainedVariance"].flatten()
    )

    def get_matlab_prefix(stage, block_idx):
        if stage == 1:
            return f"2{['a', 'b', 'c'][block_idx]}"
        elif stage == 2:
            return ["3a", "3b1", "3b2", "3b3"][block_idx]
        elif stage == 3:
            return "4a" if block_idx == 0 else f"4b{block_idx}"
        elif stage == 4:
            return f"5{['a', 'b', 'c'][block_idx]}"

    stages = [
        (1, model.layer1, 3),
        (2, model.layer2, 4),
        (3, model.layer3, 23),
        (4, model.layer4, 3),
    ]

    for s_idx, layer, count in stages:
        for b_idx in range(count):
            block = layer[b_idx]
            p = get_matlab_prefix(s_idx, b_idx)
            # conv1
            block.conv1.weight.data = torch.from_numpy(
                data[f"res{p}_branch2a__Weights"].transpose(3, 2, 0, 1)
            )
            block.bn1.weight.data = torch.from_numpy(
                data[f"bn{p}_branch2a__Scale"].flatten()
            )
            block.bn1.bias.data = torch.from_numpy(
                data[f"bn{p}_branch2a__Offset"].flatten()
            )
            block.bn1.running_mean.data = torch.from_numpy(
                data[f"bn{p}_branch2a__TrainedMean"].flatten()
            )
            block.bn1.running_var.data = torch.from_numpy(
                data[f"bn{p}_branch2a__TrainedVariance"].flatten()
            )
            # conv2
            block.conv2.weight.data = torch.from_numpy(
                data[f"res{p}_branch2b__Weights"].transpose(3, 2, 0, 1)
            )
            block.bn2.weight.data = torch.from_numpy(
                data[f"bn{p}_branch2b__Scale"].flatten()
            )
            block.bn2.bias.data = torch.from_numpy(
                data[f"bn{p}_branch2b__Offset"].flatten()
            )
            block.bn2.running_mean.data = torch.from_numpy(
                data[f"bn{p}_branch2b__TrainedMean"].flatten()
            )
            block.bn2.running_var.data = torch.from_numpy(
                data[f"bn{p}_branch2b__TrainedVariance"].flatten()
            )
            # conv3
            block.conv3.weight.data = torch.from_numpy(
                data[f"res{p}_branch2c__Weights"].transpose(3, 2, 0, 1)
            )
            block.bn3.weight.data = torch.from_numpy(
                data[f"bn{p}_branch2c__Scale"].flatten()
            )
            block.bn3.bias.data = torch.from_numpy(
                data[f"bn{p}_branch2c__Offset"].flatten()
            )
            block.bn3.running_mean.data = torch.from_numpy(
                data[f"bn{p}_branch2c__TrainedMean"].flatten()
            )
            block.bn3.running_var.data = torch.from_numpy(
                data[f"bn{p}_branch2c__TrainedVariance"].flatten()
            )
            # downsample
            if block.downsample is not None:
                block.downsample[0].weight.data = torch.from_numpy(
                    data[f"res{p}_branch1__Weights"].transpose(3, 2, 0, 1)
                )
                block.downsample[1].weight.data = torch.from_numpy(
                    data[f"bn{p}_branch1__Scale"].flatten()
                )
                block.downsample[1].bias.data = torch.from_numpy(
                    data[f"bn{p}_branch1__Offset"].flatten()
                )
                block.downsample[1].running_mean.data = torch.from_numpy(
                    data[f"bn{p}_branch1__TrainedMean"].flatten()
                )
                block.downsample[1].running_var.data = torch.from_numpy(
                    data[f"bn{p}_branch1__TrainedVariance"].flatten()
                )

    # FC
    model.fc.weight.data = torch.from_numpy(data["fc1000__Weights"])
    model.fc.bias.data = torch.from_numpy(data["fc1000__Bias"].flatten())


def main():
    print(f"Loading weights from {RAW_WEIGHTS}...")
    model = ResNet101V1(num_classes=5)
    load_weights(model, RAW_WEIGHTS)
    model.eval()

    # Test 1: Zero input
    with torch.no_grad():
        x_zero = torch.zeros((1, 3, 448, 448), dtype=torch.float32)
        logits_zero = model(x_zero)
        y_zero = torch.softmax(logits_zero, dim=1).numpy()[0]
        print("Zero input prediction:")
        print("  PyTorch:", np.round(y_zero, 4).tolist())
        matlab_zero = np.array([0.2972, 0.1439, 0.5564, 0.0022, 0.0004])
        print("  MATLAB: ", matlab_zero.tolist())
        diff_zero = np.max(np.abs(y_zero - matlab_zero))
        print(f"  Max absolute difference: {diff_zero:.6f}")

    # Test 2: Real sample image
    sample_path = PROJECT_ROOT / "frontend" / "public" / "samples" / "no_dr.jpg"
    img = Image.open(sample_path).convert("RGB").resize((448, 448))
    img_np = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        logits_sample = model(tensor)
        y_sample = torch.softmax(logits_sample, dim=1).numpy()[0]
        print(f"\nReal image ({sample_path.name}) prediction:")
        print("  PyTorch:", np.round(y_sample, 4).tolist())
        matlab_sample = np.array([0.0022, 0.0009, 0.9162, 0.0808, 0.0000])
        print("  MATLAB: ", matlab_sample.tolist())
        diff_sample = np.max(np.abs(y_sample - matlab_sample))
        print(f"  Max absolute difference: {diff_sample:.6f}")

    # Save PyTorch checkpoint
    print(f"\nSaving PyTorch model to {OUTPUT_PT}...")
    torch.save(model.state_dict(), OUTPUT_PT)

    # Export to ONNX
    print(f"Exporting to ONNX at {OUTPUT_ONNX}...")
    dummy_input = torch.randn(1, 3, 448, 448, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_ONNX),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[OK] Successfully exported ONNX model to {OUTPUT_ONNX}")
    print(f"  File size: {OUTPUT_ONNX.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
