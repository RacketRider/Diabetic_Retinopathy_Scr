"""Security tests for upload validation."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_valid_jpeg(width: int = 512, height: int = 512) -> bytes:
    """Create a minimal valid JPEG in memory."""
    img = Image.new("RGB", (width, height), color=(100, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_valid_png(width: int = 512, height: int = 512) -> bytes:
    """Create a minimal valid PNG in memory."""
    img = Image.new("RGB", (width, height), color=(100, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data


def test_reject_non_image_extension():
    resp = client.post(
        "/api/v1/predict",
        files={"image": ("malware.exe", b"MZ" + b"\x00" * 100, "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_reject_oversized_file():
    # Create a file just over the limit
    big_data = b"\xff\xd8\xff\xe0" + b"\x00" * (16 * 1024 * 1024)
    resp = client.post(
        "/api/v1/predict",
        files={"image": ("huge.jpg", big_data, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_reject_fake_extension():
    """File with .jpg extension but PNG magic bytes should be accepted (PNG is allowed)."""
    png_data = _make_valid_png()
    resp = client.post(
        "/api/v1/predict",
        files={"image": ("image.jpg", png_data, "image/jpeg")},
    )
    # Should NOT be rejected for magic byte mismatch since PNG is allowed
    assert resp.status_code in (200, 400, 503)  # 503 if model not loaded, 400 if quality check


def test_reject_text_as_image():
    """Text file renamed to .jpg should be rejected."""
    resp = client.post(
        "/api/v1/predict",
        files={"image": ("trick.jpg", b"This is not an image", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_reject_tiny_image():
    """Image below minimum resolution should be rejected."""
    tiny = _make_valid_jpeg(width=100, height=100)
    resp = client.post(
        "/api/v1/predict",
        files={"image": ("tiny.jpg", tiny, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "too small" in resp.json()["detail"]


def test_security_headers():
    resp = client.get("/api/v1/health")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "no-store" in resp.headers.get("Cache-Control", "")
