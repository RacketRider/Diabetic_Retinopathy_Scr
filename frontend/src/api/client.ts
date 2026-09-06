/**
 * API client for the DR-Screen AI backend.
 * Mode C: calls http://localhost:8000 via Vite proxy.
 * All requests have a 30-second timeout.
 */

const API_BASE = '/api/v1';
const TIMEOUT_MS = 30_000;

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function fetchWithTimeout(url: string, options: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      throw new ApiError(body.detail || 'Request failed', response.status);
    }
    return response;
  } finally {
    clearTimeout(timeout);
  }
}

export async function checkHealth() {
  const resp = await fetchWithTimeout(`${API_BASE}/health`, { method: 'GET' });
  return resp.json();
}

export async function predictDR(file: File) {
  const form = new FormData();
  form.append('image', file);
  const resp = await fetchWithTimeout(`${API_BASE}/predict`, {
    method: 'POST',
    body: form,
  });
  return resp.json();
}

export async function getGradCAM(file: File) {
  const form = new FormData();
  form.append('image', file);
  const resp = await fetchWithTimeout(`${API_BASE}/gradcam`, {
    method: 'POST',
    body: form,
  });
  return resp.json();
}

export async function downloadReport(file: File): Promise<Blob> {
  const form = new FormData();
  form.append('image', file);
  const resp = await fetchWithTimeout(`${API_BASE}/report`, {
    method: 'POST',
    body: form,
  });
  return resp.blob();
}

export { ApiError };
