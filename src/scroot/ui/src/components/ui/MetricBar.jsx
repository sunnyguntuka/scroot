import { useEffect, useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { METRIC_COLORS, METRIC_LABELS } from '../../utils/iqs';

/**
 * Horizontal progress bar for a single metric score.
 * Width animates only on first mount (not re-renders).
 */
export function MetricBar({ metric, value, showLabel = true, showValue = true, explanation = null }) {
  const [width, setWidth] = useState(0);
  const mounted = useRef(false);
  const color = METRIC_COLORS[metric] || '#4F46E5';

  // Inapplicable metric (e.g. groundedness with no context): show a neutral
  // hatched bar and an em-dash, not a red 0.0. A genuine 0.0 still renders
  // as a real (empty) bar below.
  if (value === null || value === undefined) {
    return (
      <div className="flex items-center gap-3" title="not scored - no context provided">
        {showLabel && (
          <span className="w-28 shrink-0 text-[11px] uppercase tracking-wider text-indigo-400 font-medium">
            {METRIC_LABELS[metric] || metric}
          </span>
        )}
        <div
          className="flex-1 h-[5px] rounded-full"
          style={{ background: 'repeating-linear-gradient(90deg, #E0E7FF 0 4px, transparent 4px 8px)' }}
        />
        {showValue && (
          <span className="w-9 text-right text-[12px] font-mono-score tabular-nums text-indigo-300">—</span>
        )}
      </div>
    );
  }

  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      // requestAnimationFrame ensures the transition fires after first paint
      requestAnimationFrame(() => setWidth(Math.min(value, 1) * 100));
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex items-center gap-3">
      {showLabel && (
        <span className="w-28 shrink-0 flex items-center gap-1 text-[11px] uppercase tracking-wider text-indigo-500 font-medium">
          {METRIC_LABELS[metric] || metric}
          {explanation && (
            <span className="relative group inline-flex">
              <AlertTriangle size={11} className="text-amber-500 cursor-help shrink-0" />
              <span className="absolute bottom-5 left-0 w-56 p-2.5 bg-indigo-950 text-white text-[11px]
                              normal-case tracking-normal font-normal leading-snug rounded-lg
                              opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 shadow-xl">
                {explanation}
              </span>
            </span>
          )}
        </span>
      )}
      <div className="flex-1 h-[5px] bg-indigo-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${width}%`,
            backgroundColor: color,
            transition: 'width 400ms ease-out',
          }}
        />
      </div>
      {showValue && (
        <span className="w-9 text-right text-[12px] font-mono-score tabular-nums" style={{ color }}>
          {value.toFixed(2)}
        </span>
      )}
    </div>
  );
}
