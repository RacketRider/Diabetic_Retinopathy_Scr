"""Complete End-to-End Pipeline Verification Script.

Tests the full Diabetic Retinopathy screening stack:
1. Model loading and weights verification
2. Real fundus images across all DR grades
3. FastAPI endpoints: /health, /predict, /gradcam, /report, /quality
4. Clinical report PDF generation
5. Edge cases: resolution, aspect ratio, grayscale, corrupted, tiny, oversized
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app
from app.services.inference import load_model, is_model_loaded, predict, MODEL_PATH
from fastapi.testclient import TestClient

client = TestClient(app)

SAMPLE_DIR = PROJECT_ROOT / "frontend" / "public" / "samples"
SAMPLES = ["no_dr.jpg", "mild.jpg", "moderate.jpg", "severe.jpg", "proliferative.jpg"]


def test_1_model_loading():
    print("\n" + "=" * 70)
    print("TEST 1: Model Loading and Session Verification")
    print("=" * 70)

    model_file = Path(MODEL_PATH)
    print(f"Model path: {model_file}")
    assert model_file.exists(), f"Model file not found at {model_file}"
    print(f"Model file size: {model_file.stat().st_size / 1024 / 1024:.2f} MB")

    loaded = load_model(str(model_file))
    print(f"load_model() returned: {loaded}")
    assert loaded, "load_model() returned False"
    assert is_model_loaded(), "is_model_loaded() returned False"

    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    health_data = resp.json()
    print(f"Health check response: {health_data}")
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True
    print("[PASS] Model loaded and /health endpoint healthy.")


def test_2_real_images_inference():
    print("\n" + "=" * 70)
    print("TEST 2: Real Fundus Image Inference Across All DR Severity Grades")
    print("=" * 70)

    results = {}
    for sample_name in SAMPLES:
        sample_path = SAMPLE_DIR / sample_name
        assert sample_path.exists(), f"Sample image not found: {sample_path}"

        img = Image.open(sample_path)
        orig_w, orig_h = img.size

        with open(sample_path, "rb") as f:
            t0 = time.perf_counter()
            resp = client.post(
                "/api/v1/predict",
                files={"image": (sample_name, f.read(), "image/jpeg")},
            )
            elapsed = (time.perf_counter() - t0) * 1000

        assert resp.status_code == 200, f"Predict failed for {sample_name}: {resp.text}"
        data = resp.json()
        grading = data["grading"]
        confidence = data["confidence"]
        quality = data["quality"]

        print(f"\nImage: {sample_name} ({orig_w}x{orig_h})")
        print(f"  Predicted Class : {grading['label']} (ICDR Level {grading['level']})")
        print(f"  Description     : {grading['description']}")
        print(f"  Referable DR    : {'YES' if grading['is_referable'] else 'NO'} (score: {confidence['calibrated_score']:.4f}, threshold: {confidence['threshold_used']})")
        print(f"  Probabilities   : {confidence['class_probabilities']}")
        print(f"  Quality Gradeable: {quality['is_gradeable']} (Focus: {quality['focus_score']:.2f}, Illum: {quality['illumination_score']:.2f})")
        print(f"  Server Time     : {data['processing_time_ms']} ms | Total Round-trip: {elapsed:.1f} ms")

        results[sample_name] = data

    print("\n[PASS] All 5 sample images processed through full predict pipeline.")
    return results


def test_3_gradcam_endpoint():
    print("\n" + "=" * 70)
    print("TEST 3: Grad-CAM Explainability Endpoint (/api/v1/gradcam)")
    print("=" * 70)

    sample_path = SAMPLE_DIR / "severe.jpg"
    with open(sample_path, "rb") as f:
        resp = client.post(
            "/api/v1/gradcam",
            files={"image": ("severe.jpg", f.read(), "image/jpeg")},
        )

    assert resp.status_code == 200, f"Grad-CAM failed: {resp.text}"
    data = resp.json()
    gradcam = data["gradcam"]

    assert "overlay_base64" in gradcam and len(gradcam["overlay_base64"]) > 1000
    assert "attention_regions" in gradcam
    assert "clinical_evidence" in gradcam

    print(f"Grad-CAM overlay base64 length: {len(gradcam['overlay_base64'])} characters")
    print(f"Attention regions detected: {len(gradcam['attention_regions'])}")
    for i, r in enumerate(gradcam["attention_regions"][:3]):
        print(f"  Region {i+1}: type={r['type']}, bbox={r['bbox']}, confidence={r['confidence']}")
    print(f"Clinical Evidence Text: {gradcam['clinical_evidence']}")
    print("[PASS] Grad-CAM explainability endpoint fully functional.")


def test_4_clinical_report_generation():
    print("\n" + "=" * 70)
    print("TEST 4: Clinical Report PDF Generation (/api/v1/report)")
    print("=" * 70)

    sample_path = SAMPLE_DIR / "moderate.jpg"
    with open(sample_path, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": ("moderate.jpg", f.read(), "image/jpeg")},
        )

    assert resp.status_code == 200, f"Report generation failed: {resp.text}"
    assert resp.headers.get("content-type") == "application/pdf"
    pdf_bytes = resp.content
    assert len(pdf_bytes) > 2000, "PDF bytes suspiciously small"
    assert pdf_bytes[:4] == b"%PDF", "Response is not a valid PDF file (missing %PDF header)"

    out_pdf = PROJECT_ROOT / "test_dr_report.pdf"
    with open(out_pdf, "wb") as f:
        f.write(pdf_bytes)
    print(f"Report PDF generated successfully: {out_pdf} ({len(pdf_bytes)} bytes)")
    print("[PASS] Clinical report generator fully functional.")


def test_5_image_quality_endpoint():
    print("\n" + "=" * 70)
    print("TEST 5: Image Quality Assessment Endpoint (/api/v1/quality)")
    print("=" * 70)

    sample_path = SAMPLE_DIR / "mild.jpg"
    with open(sample_path, "rb") as f:
        resp = client.post(
            "/api/v1/quality",
            files={"image": ("mild.jpg", f.read(), "image/jpeg")},
        )

    assert resp.status_code == 200, f"Quality check failed: {resp.text}"
    q = resp.json()
    print(f"Quality result: {q}")
    assert "is_gradeable" in q
    assert "focus_score" in q
    assert "illumination_score" in q
    assert "fov_adequate" in q
    print("[PASS] Quality endpoint functional.")


def test_6_edge_cases():
    print("\n" + "=" * 70)
    print("TEST 6: Edge Case Testing")
    print("=" * 70)

    edge_tests = []

    # Case A: Different resolution (High-res 1200x1200)
    img_high = Image.new("RGB", (1200, 1200), color=(120, 60, 40))
    buf = io.BytesIO()
    img_high.save(buf, format="JPEG")
    resp = client.post("/api/v1/quality", files={"image": ("highres.jpg", buf.getvalue(), "image/jpeg")})
    edge_tests.append(("High Resolution (1200x1200)", resp.status_code in (200, 400), f"Status {resp.status_code}"))

    # Case B: Non-square Aspect Ratio (16:9 widescreen: 1280x720)
    img_wide = Image.new("RGB", (1280, 720), color=(100, 50, 30))
    buf = io.BytesIO()
    img_wide.save(buf, format="JPEG")
    resp = client.post("/api/v1/quality", files={"image": ("wide.jpg", buf.getvalue(), "image/jpeg")})
    edge_tests.append(("Non-square Aspect Ratio (1280x720)", resp.status_code in (200, 400), f"Status {resp.status_code}"))

    # Case C: Grayscale image (mode L converted to RGB)
    img_gray = Image.new("L", (512, 512), color=128).convert("RGB")
    buf = io.BytesIO()
    img_gray.save(buf, format="JPEG")
    resp = client.post("/api/v1/quality", files={"image": ("gray.jpg", buf.getvalue(), "image/jpeg")})
    edge_tests.append(("Grayscale RGB Image", resp.status_code in (200, 400), f"Status {resp.status_code}"))

    # Case D: Black / No-target image (should fail quality gracefully)
    img_black = Image.new("RGB", (512, 512), color=(0, 0, 0))
    buf = io.BytesIO()
    img_black.save(buf, format="JPEG")
    resp = client.post("/api/v1/predict", files={"image": ("black.jpg", buf.getvalue(), "image/jpeg")})
    edge_tests.append(("All-Black / No-FOV Image Rejected Gracefully", resp.status_code == 400, f"Status {resp.status_code} (Rejection expected)"))

    # Case E: Too Small (< 256x256)
    img_tiny = Image.new("RGB", (100, 100), color=(100, 50, 50))
    buf = io.BytesIO()
    img_tiny.save(buf, format="JPEG")
    resp = client.post("/api/v1/predict", files={"image": ("tiny.jpg", buf.getvalue(), "image/jpeg")})
    edge_tests.append(("Tiny Image (< 256px) Rejected", resp.status_code == 400, f"Status {resp.status_code}"))

    # Case F: Invalid Extension / Executable File
    resp = client.post("/api/v1/predict", files={"image": ("exploit.exe", b"MZ" + b"\x00" * 50, "application/octet-stream")})
    edge_tests.append(("Non-Image Extension Rejected", resp.status_code == 400, f"Status {resp.status_code}"))

    # Case G: Corrupt Image Content
    resp = client.post("/api/v1/predict", files={"image": ("corrupted.jpg", b"NOT_A_REAL_IMAGE_BYTES", "image/jpeg")})
    edge_tests.append(("Corrupt File Bytes Rejected", resp.status_code == 400, f"Status {resp.status_code}"))

    # Case H: Oversized Image File (> 15 MB)
    oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (16 * 1024 * 1024)
    resp = client.post("/api/v1/predict", files={"image": ("oversized.jpg", oversized, "image/jpeg")})
    edge_tests.append(("Oversized File (>15MB) Rejected (413)", resp.status_code == 413, f"Status {resp.status_code}"))

    for name, passed, detail in edge_tests:
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"  {status_str} {name}: {detail}")
        assert passed, f"Edge case failed: {name}"

    print("\n[PASS] All 8 edge cases passed successfully.")


def main():
    print("=" * 70)
    print("STARTING FULL TECHNICAL VERIFICATION SUITE")
    print("=" * 70)

    test_1_model_loading()
    results = test_2_real_images_inference()
    test_3_gradcam_endpoint()
    test_4_clinical_report_generation()
    test_5_image_quality_endpoint()
    test_6_edge_cases()

    print("\n" + "=" * 70)
    print("ALL VERIFICATION SUITES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
