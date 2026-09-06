"""Image quality assessment endpoint."""

from fastapi import APIRouter, Request, UploadFile, File
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import RATE_LIMIT
from app.schemas.models import QualityResult
from app.security import validate_upload
from app.services.image_quality import assess_quality
from app.services.preprocessing import pil_to_numpy

router = APIRouter(prefix="/api/v1", tags=["quality"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/quality", response_model=QualityResult)
@limiter.limit(RATE_LIMIT)
async def check_quality(request: Request, image: UploadFile = File(...)):
    """Assess fundus image quality before classification.

    Returns focus, illumination, and FOV scores.
    Use this to pre-check images before submitting for prediction.
    """
    clean_image = await validate_upload(image)
    image_np = pil_to_numpy(clean_image)
    return assess_quality(image_np)
