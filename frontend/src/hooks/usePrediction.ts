import { useState, useCallback } from 'react';
import { getGradCAM } from '@/api/client';
import type { GradCAMResponse } from '@/types';

export function usePrediction() {
  const [result, setResult] = useState<GradCAMResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Use gradcam endpoint for full analysis (includes prediction + explainability)
      const data: GradCAMResponse = await getGradCAM(file);
      setResult(data);
    } catch (err: any) {
      const message = err?.message || 'Analysis failed. Is the backend running?';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, isLoading, error, analyze, reset };
}
