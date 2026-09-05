import os
from pathlib import Path

# All configuration via environment variables — no secrets in code
MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parents[2] / ".." / "models" / "dr_resnet101_e1.onnx"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
RATE_LIMIT = os.getenv("RATE_LIMIT", "15/minute")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "448"))

# Upload constraints
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_PIXELS = 25_000_000     # pixel bomb defense
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIMES = {"image/jpeg", "image/png"}
MIN_DIMENSION = 256

# Model constants (from MATLAB run_tests.m — this order must match ONNX output)
CLASS_NAMES = ["Severe", "No_DR", "Mild", "Proliferate_DR", "Moderate"]
REFERABLE_CLASSES = {"Moderate", "Severe", "Proliferate_DR"}
E1_THRESHOLD = 0.164  # from artifacts/results/V3/V3_E1_HighResFOV448/selected_threshold.txt
LEVEL_MAP = {"No_DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferate_DR": 4}
LEVEL_DESCRIPTIONS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy",
}
