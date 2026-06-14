// Canonical IQS status logic - import this everywhere; never inline.

export function getIQSStatus(iqs, threshold = 0.70) {
  if (iqs >= threshold) return 'pass';
  if (iqs >= threshold * 0.7) return 'warn';
  return 'fail';
}

export const STATUS_COLORS = {
  pass: {
    text:   'text-green-700',
    bg:     'bg-green-50',
    border: 'border-green-200',
    dot:    'bg-green-500',
    hex:    '#059669',
  },
  warn: {
    text:   'text-amber-700',
    bg:     'bg-amber-50',
    border: 'border-amber-200',
    dot:    'bg-amber-500',
    hex:    '#D97706',
  },
  fail: {
    text:   'text-red-700',
    bg:     'bg-red-50',
    border: 'border-red-200',
    dot:    'bg-red-500',
    hex:    '#DC2626',
  },
};

// Per-metric fill colors for MetricBar and charts
export const METRIC_COLORS = {
  groundedness: '#4F46E5',
  completeness: '#818CF8',
  relevance:    '#6366F1',
  consistency:  '#A5B4FC',
  confidence:   '#C7D2FE',
};

export const METRIC_LABELS = {
  groundedness: 'Groundedness',
  completeness: 'Completeness',
  relevance:    'Relevance',
  consistency:  'Consistency',
  confidence:   'Confidence',
};

export const ALL_METRICS = ['groundedness', 'completeness', 'relevance', 'consistency', 'confidence'];

// Format IQS as 2-decimal string - always use this for display
export const fmtIQS = (v) => (typeof v === 'number' ? v.toFixed(2) : '—');
