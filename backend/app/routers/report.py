"""Clinical report generation endpoint."""

import time

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import INPUT_SIZE, RATE_LIMIT
from app.security import validate_upload
from app.services.gradcam_service import (
    compute_gradcam_from_onnx,
    detect_attention_regions,
    generate_clinical_evidence,
    overlay_gradcam,
)
from app.services.image_quality import assess_quality
from app.services.inference import is_model_loaded, predict, MODEL_PATH
from app.services.preprocessing import pil_to_numpy, preprocess_fundus
from app.services.report_generator import generate_report

router = APIRouter(prefix="/api/v1", tags=["reports"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/report")
@limiter.limit(RATE_LIMIT)
async def generate_clinical_report(request: Request, image: UploadFile = File(...)):
    """Generate a downloadable PDF clinical screening report.

    Includes DR grading, Grad-CAM overlay, lesion evidence,
    and clinical recommendation.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Validate and process
    clean_image = await validate_upload(image)
    image_np = pil_to_numpy(clean_image)

    # Quality check
    quality = assess_quality(image_np)

    # Inference
    result = predict(image_np)

    # Grad-CAM
    preprocessed = preprocess_fundus(image_np, target_size=INPUT_SIZE)
    input_tensor = np.transpose(preprocessed, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    cam = compute_gradcam_from_onnx(MODEL_PATH, input_tensor)

    import cv2
    cam_fullres = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    regions = detect_attention_regions(cam_fullres)
    evidence = generate_clinical_evidence(
        regions, result["grading"]["label"], result["grading"]["level"]
    )

    # Generate PDF
    pdf_bytes = generate_report(
        grading=result["grading"],
        confidence=result["confidence"],
        quality=quality,
        clinical_evidence=evidence,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=dr_screening_report.pdf",
            "Cache-Control": "no-store",
        },
    )
