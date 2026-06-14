// Thin REST client - all endpoints match the FastAPI backend at /api/*
// Add ?demo to the URL to use mock data instead of the real backend.

import { mockApi } from './mock';

export const DEMO_MODE = new URLSearchParams(window.location.search).has('demo');

const BASE = '/api';

async function req(path, options = {}) {
  const { body, ...rest } = options;
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...rest.headers },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...rest,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw Object.assign(new Error(err.error || `HTTP ${res.status}`), { code: err.code, status: res.status });
  }
  // For file-stream endpoints, return the raw response
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res;
}

// Build ExportRequest body from flat UI params
function exportBody(params = {}) {
  const status = [];
  if (params.include_reviewed !== false) status.push('reviewed', 'applied');
  if (params.include_rejected !== false) status.push('rejected');
  if (params.include_pending)            status.push('pending');
  return {
    format: params.format || 'jsonl',
    filters: {
      status:   status.length ? [...new Set(status)] : ['reviewed', 'applied'],
      date_from: params.date_from || null,
      date_to:   params.date_to   || null,
      min_iqs:   params.min_iqs_improvement > 0 ? params.min_iqs_improvement : null,
      max_iqs:   null,
      flags:     [],
      agents:    params.agent ? [params.agent] : [],
    },
  };
}

const liveApi = {
  // ─── Queue ──────────────────────────────────────────────────────────
  getQueue:           (params = {})    => req(`/queue?${new URLSearchParams(params)}`),

  // ─── Records ────────────────────────────────────────────────────────
  getRecord:          (id)             => req(`/records/${id}`),
  submitCorrection:   (id, correction) => req(`/records/${id}/review`,             { method: 'POST', body: { correction, category: 'manual' } }),
  rejectRecord:       (id, reason)     => req(`/records/${id}/reject`,             { method: 'POST', body: { reason } }),
  generateCorrection: (id)             => req(`/records/${id}/generate-correction`, { method: 'POST' }),
  deleteCorrection:   (id)             => req(`/records/${id}/correction`,          { method: 'DELETE' }),

  // ─── Analytics ──────────────────────────────────────────────────────
  getAnalytics: (range = '30d') => req(`/analytics?range=${range}`),

  // ─── Export ─────────────────────────────────────────────────────────
  getExportPreview: (params = {}) => req('/export/preview',  { method: 'POST', body: exportBody(params) }),
  // Returns raw response for streaming download
  downloadExport:   (params = {}) => req('/export/download', { method: 'POST', body: exportBody(params) }),

  // ─── Settings ───────────────────────────────────────────────────────
  getSettings:    ()         => req('/settings'),
  updateSettings: (patch)    => req('/settings', { method: 'PUT', body: patch }),
  testConnection: ()         => req('/settings/test-connection', { method: 'POST' }),

  // ─── Pipeline ───────────────────────────────────────────────────────
  startPipelineRun:  (config) => req('/pipeline/run',             { method: 'POST',   body: config }),
  getPipelineStatus: (runId)  => req(`/pipeline/${runId}/status`),
  pausePipelineRun:  (runId)  => req(`/pipeline/${runId}/pause`,  { method: 'POST' }),
  resumePipelineRun: (runId)  => req(`/pipeline/${runId}/resume`, { method: 'POST' }),
  cancelPipelineRun: (runId)  => req(`/pipeline/${runId}`,        { method: 'DELETE' }),

  // ─── Corrector ──────────────────────────────────────────────────────
  getCorrectorRuntime:     ()        => req('/corrector/runtime'),
  getCorrectorModels:      ()        => req('/corrector/models'),
  downloadModel:           (modelId) => req(`/corrector/models/${modelId}/download`, { method: 'POST' }),
  getModelDownloadStatus:  (modelId) => req(`/corrector/models/${modelId}/download-status`),
  deleteModel:             (modelId) => req(`/corrector/models/${modelId}`, { method: 'DELETE' }),
  testCorrector:           ()        => req('/corrector/test', { method: 'POST' }),

  // ─── Health ─────────────────────────────────────────────────────────
  getHealth: () => req('/health'),

  // ─── Guardrails ─────────────────────────────────────────────────────
  getGuardrailStats: () => req('/guardrails/stats'),
};

export const api = DEMO_MODE ? mockApi : liveApi;
