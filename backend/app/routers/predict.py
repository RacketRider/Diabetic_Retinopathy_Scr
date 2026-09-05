"""DR prediction endpoint."""

import time

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import RATE_LIMIT
from app.schemas.models import PredictionResponse
from app.security import validate_upload
from app.services.image_quality import assess_quality
from app.services.inference import is_model_loaded, predict
from app.services.preprocessing import pil_to_numpy

router = APIRouter(prefix="/api/v1", tags=["prediction"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit(RATE_LIMIT)
async def predict_dr(request: Request, image: UploadFile = File(...)):
    """Classify a fundus image for diabetic retinopathy severity.

    Accepts JPEG/PNG fundus images up to 15 MB.
    Returns ICDR severity level (0-4), confidence scores,
    and image quality assessment.
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the ONNX model file exists.")

    # Validate and sanitize upload
    clean_image = await validate_upload(image)
    image_np = pil_to_numpy(clean_image)

    # Assess image quality
    quality = assess_quality(image_np)
    if not quality["is_gradeable"]:
        raise HTTPException(
            status_code=400,
            detail="Image quality insufficient for reliable screening. "
                   f"Focus: {quality['focus_score']:.2f}, "
                   f"Illumination: {quality['illumination_score']:.2f}, "
                   f"FOV adequate: {quality['fov_adequate']}. "
                   "Please recapture with better lighting and focus.",
        )

    # Run inference
    result = predict(image_np)
    result["quality"] = quality

    return result
