import { AlertTriangle } from 'lucide-react';
import { METRIC_LABELS } from '../../utils/iqs';

/**
 * Compact chip for a flagged metric.
 * Always red - indicates a metric below threshold.
 */
export function FlagChip({ metric, value }) {
  // Flags may be stored as 'low_groundedness' - strip prefix for display
  const normalized = metric.replace(/^low_/, '');
  const label = METRIC_LABELS[normalized] || normalized;

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border
                 bg-red-50 text-red-700 border-red-200 text-[10px] font-medium"
      title={value !== undefined ? `${label}: ${value.toFixed(2)}` : label}
    >
      <AlertTriangle size={9} strokeWidth={2.5} className="shrink-0" />
      {label}
    </span>
  );
}

/**
 * Overflow chip - shows "+N more" when flag list is truncated.
 */
export function FlagOverflow({ count }) {
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full border
                 bg-indigo-50 text-indigo-500 border-indigo-100 text-[10px] font-medium"
    >
      +{count}
    </span>
  );
}
