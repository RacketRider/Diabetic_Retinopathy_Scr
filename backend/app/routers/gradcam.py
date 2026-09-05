"""Grad-CAM explainability endpoint."""

import time

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import INPUT_SIZE, RATE_LIMIT
from app.schemas.models import GradCAMResponse
from app.security import validate_upload
from app.services.gradcam_service import (
    compute_gradcam_from_onnx,
    detect_attention_regions,
    generate_clinical_evidence,
    image_to_base64,
    overlay_gradcam,
)
from app.services.inference import is_model_loaded, predict, MODEL_PATH
from app.services.preprocessing import pil_to_numpy, preprocess_fundus

router = APIRouter(prefix="/api/v1", tags=["explainability"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/gradcam", response_model=GradCAMResponse)
@limiter.limit(RATE_LIMIT)
async def gradcam_analysis(request: Request, image: UploadFile = File(...)):
    """Generate Grad-CAM explainability overlay for a fundus image.

    Returns the DR classification plus an attention heatmap overlay,
    detected lesion regions, and clinical evidence text.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    # Validate and process
    clean_image = await validate_upload(image)
    image_np = pil_to_numpy(clean_image)

    # Run inference
    result = predict(image_np)

    # Preprocess for Grad-CAM
    preprocessed = preprocess_fundus(image_np, target_size=INPUT_SIZE)
    input_tensor = np.transpose(preprocessed, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    # Compute attention map
    cam = compute_gradcam_from_onnx(MODEL_PATH, input_tensor)

    # Generate overlay on original image
    overlay = overlay_gradcam(image_np, cam, alpha=0.5)
    overlay_b64 = image_to_base64(overlay)

    # Detect lesion regions
    import cv2
    cam_fullres = np.array(cv2.resize(cam, (image_np.shape[1], image_np.shape[0])))
    regions = detect_attention_regions(cam_fullres)

    # Clinical evidence
    grading = result["grading"]
    evidence = generate_clinical_evidence(regions, grading["label"], grading["level"])

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "grading": grading,
        "confidence": result["confidence"],
        "gradcam": {
            "overlay_base64": overlay_b64,
            "attention_regions": regions,
            "clinical_evidence": evidence,
        },
        "processing_time_ms": round(elapsed_ms, 1),
    }
