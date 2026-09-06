import io
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, "backend")
from app.main import app
from app.services.inference import load_model, MODEL_PATH
from fastapi.testclient import TestClient

client = TestClient(app)
load_model(MODEL_PATH)

dataset_base = Path(r"C:\Users\Abhij\Downloads\archive (1)\colored_images\colored_images")
categories = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

print("Testing genuine clinical fundus images from dataset:\n")
for cat in categories:
    cat_dir = dataset_base / cat
    img_path = next(cat_dir.glob("*.png"))
    w, h = Image.open(img_path).size
    with open(img_path, "rb") as f:
        resp = client.post(
            "/api/v1/predict",
            files={"image": (img_path.name, f.read(), "image/png")},
        )

    if resp.status_code == 200:
        data = resp.json()
        g = data["grading"]
        c = data["confidence"]
        q = data["quality"]
        print(f"=== Ground Truth: {cat} ({w}x{h}, {img_path.name}) ===")
        print(f"  Predicted Label : {g['label']} (ICDR Level {g['level']})")
        print(f"  Description     : {g['description']}")
        print(f"  Referable DR    : {'YES' if g['is_referable'] else 'NO'} (score: {c['calibrated_score']:.4f})")
        print(f"  Probabilities   : {c['class_probabilities']}")
        print(f"  Image Quality   : Gradeable={q['is_gradeable']}, Focus={q['focus_score']}, Illum={q['illumination_score']}")
        print(f"  Inference Time  : {data['processing_time_ms']} ms\n")
    else:
        print(f"FAILED for {cat}: {resp.status_code} - {resp.text}\n")
