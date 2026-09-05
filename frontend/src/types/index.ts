export interface GradingResult {
  level: number;
  label: string;
  description: string;
  is_referable: boolean;
}

export interface ConfidenceResult {
  calibrated_score: number;
  class_probabilities: Record<string, number>;
  referable_probability: number;
  threshold_used: number;
}

export interface QualityResult {
  is_gradeable: boolean;
  focus_score: number;
  illumination_score: number;
  fov_adequate: boolean;
  enhancement_applied: boolean;
}

export interface PredictionResponse {
  grading: GradingResult;
  confidence: ConfidenceResult;
  quality: QualityResult;
  processing_time_ms: number;
}

export interface AttentionRegion {
  type: string;
  bbox: number[];
  confidence: number;
}

export interface GradCAMResult {
  overlay_base64: string;
  attention_regions: AttentionRegion[];
  clinical_evidence: string;
}

export interface GradCAMResponse {
  grading: GradingResult;
  confidence: ConfidenceResult;
  gradcam: GradCAMResult;
  processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string;
  version: string;
}

export type AppView = 'upload' | 'loading' | 'results';

export const SEVERITY_COLORS: Record<string, string> = {
  No_DR: '#22c55e',
  Mild: '#84cc16',
  Moderate: '#f59e0b',
  Severe: '#f97316',
  Proliferate_DR: '#ef4444',
};

export const SEVERITY_LABELS: Record<string, string> = {
  No_DR: 'No DR',
  Mild: 'Mild NPDR',
  Moderate: 'Moderate NPDR',
  Severe: 'Severe NPDR',
  Proliferate_DR: 'Proliferative DR',
};
