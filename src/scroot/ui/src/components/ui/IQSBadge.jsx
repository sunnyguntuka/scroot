import { getIQSStatus, STATUS_COLORS, fmtIQS } from '../../utils/iqs';

const SIZE_CLASSES = {
  sm: 'text-[11px] px-2 py-0.5',
  md: 'text-sm px-2.5 py-1',
  lg: 'text-2xl px-3 py-1.5 font-semibold',
};

/**
 * IQS score pill - color by pass/warn/fail status.
 * Score always in JetBrains Mono per spec.
 */
export function IQSBadge({ score, size = 'md', threshold = 0.70, showLabel = false, metricCount = 5 }) {
  const status = getIQSStatus(score, threshold);
  const c = STATUS_COLORS[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-mono-score font-medium
        ${c.bg} ${c.text} ${c.border} ${SIZE_CLASSES[size]}`}
    >
      {fmtIQS(score)}
      {metricCount < 5 && (
        <span
          className="text-[9px] opacity-70"
          title={`IQS computed from ${metricCount} of 5 metrics - groundedness not scored (no context). Add context for a complete score.`}
        >
          ({metricCount}/5)
        </span>
      )}
      {showLabel && (
        <span className="text-[10px] uppercase tracking-wider opacity-60">{status}</span>
      )}
    </span>
  );
}
