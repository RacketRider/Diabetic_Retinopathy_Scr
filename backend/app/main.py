"""FastAPI application — Diabetic Retinopathy Screening API.

Security-hardened backend for local Mode C deployment.
All medical images are processed in-memory only (zero disk persistence).
"""

import logging

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import ALLOWED_ORIGINS, LOG_LEVEL, RATE_LIMIT
from app.routers import predict, gradcam, quality, report
from app.schemas.models import HealthResponse
from app.services.inference import is_model_loaded, load_model

# Logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# App — disable Swagger/ReDoc in production for security
app = FastAPI(
    title="DR-Screen AI API",
    description="Explainable AI for Diabetic Retinopathy Screening",
    version="1.0.0",
    docs_url="/docs",      # enabled for local dev; set to None in production
    redoc_url=None,
)

# --- Middleware stack (order matters: outermost first) ---

# 1. CORS — tight allowlist for Mode C (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    max_age=600,
)

# 2. Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# 3. Security headers — applied to every response
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# --- Routers ---
app.include_router(predict.router)
app.include_router(gradcam.router)
app.include_router(quality.router)
app.include_router(report.router)


# --- Startup / Health ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting DR-Screen AI backend...")
    if load_model():
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️  Model not found — /predict will return 503 until model is available")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Liveness and readiness probe."""
    return {
        "status": "healthy" if is_model_loaded() else "degraded",
        "model_loaded": is_model_loaded(),
        "model_name": "V3_E1_HighResFOV448",
        "version": "1.0.0",
    }
