import { getIQSStatus, STATUS_COLORS, fmtIQS } from '../../utils/iqs';

/**
 * 52px top bar - page title (left) + ambient IQS health pill (right).
 * Health pill always visible: shows avg score when available, - otherwise.
 */
export function TopBar({ title, avgIqs, threshold = 0.70, actions }) {
  const hasIqs = typeof avgIqs === 'number';
  const status = hasIqs ? getIQSStatus(avgIqs, threshold) : null;
  const c = status ? STATUS_COLORS[status] : null;

  return (
    <header className="h-[52px] bg-white border-b border-indigo-100 flex items-center justify-between px-6 shrink-0 z-40">
      {/* Left: page title */}
      <h1 className="text-[15px] font-medium text-indigo-950 tracking-[-0.01em]">
        {title}
      </h1>

      {/* Right: custom actions + IQS health pill */}
      <div className="flex items-center gap-3">
        {actions}
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-mono-score transition-colors
            ${hasIqs
              ? `${c.bg} ${c.text} ${c.border}`
              : 'bg-indigo-25 text-indigo-300 border-indigo-100'
            }`}
          title="Today's average IQS"
        >
          {hasIqs && <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />}
          {hasIqs ? `avg ${fmtIQS(avgIqs)}` : '—'}
        </span>
      </div>
    </header>
  );
}
