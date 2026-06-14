import { useEffect, useState } from 'react';
import { getIQSStatus, STATUS_COLORS, fmtIQS } from '../../utils/iqs';

/**
 * SVG donut ring showing IQS score.
 * Inner text always in JetBrains Mono 600.
 */
export function ScoreRing({ score, threshold = 0.70, size = 72, strokeWidth = 6 }) {
  const [animated, setAnimated] = useState(false);
  const status = getIQSStatus(score, threshold);
  const c = STATUS_COLORS[status];

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = animated ? circumference * (1 - Math.min(score, 1)) : circumference;

  useEffect(() => {
    requestAnimationFrame(() => setAnimated(true));
  }, []);

  return (
    <div
      className="relative inline-flex items-center justify-center"
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        {/* Track */}
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="#E0E7FF" strokeWidth={strokeWidth}
        />
        {/* Fill */}
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none"
          stroke={c.hex}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 600ms ease-out' }}
        />
      </svg>
      <span
        className="absolute font-mono-score font-semibold leading-none"
        style={{
          fontSize: size < 64 ? 11 : 13,
          color: c.hex,
        }}
        aria-label={`IQS ${fmtIQS(score)}`}
      >
        {fmtIQS(score)}
      </span>
    </div>
  );
}
