from pydantic import BaseModel


class GradingResult(BaseModel):
    level: int
    label: str
    description: str
    is_referable: bool


class ConfidenceResult(BaseModel):
    calibrated_score: float
    class_probabilities: dict[str, float]
    referable_probability: float
    threshold_used: float


class QualityResult(BaseModel):
    is_gradeable: bool
    focus_score: float
    illumination_score: float
    fov_adequate: bool
    enhancement_applied: bool


class PredictionResponse(BaseModel):
    grading: GradingResult
    confidence: ConfidenceResult
    quality: QualityResult
    processing_time_ms: float


class AttentionRegion(BaseModel):
    type: str
    bbox: list[int]
    confidence: float


class GradCAMResult(BaseModel):
    overlay_base64: str
    attention_regions: list[AttentionRegion]
    clinical_evidence: str


class GradCAMResponse(BaseModel):
    grading: GradingResult
    confidence: ConfidenceResult
    gradcam: GradCAMResult
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
