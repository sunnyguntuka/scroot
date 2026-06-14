import { useEffect, useState } from 'react';

export function ScoreBar({ label, value }) {
  const [width, setWidth] = useState(0);
  const color = value >= 0.8 ? 'var(--green)'
    : value >= 0.6 ? 'var(--yellow)'
    : 'var(--red)';

  // Animate from 0 on mount
  useEffect(() => {
    const t = requestAnimationFrame(() => setWidth(value * 100));
    return () => cancelAnimationFrame(t);
  }, [value]);

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          {label}
        </span>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
          color,
          fontWeight: 700,
        }}>
          {typeof value === 'number' ? value.toFixed(2) : '—'}
        </span>
      </div>
      <div style={{
        height: 4,
        background: 'var(--bg-border)',
        borderRadius: 2,
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${width}%`,
          background: color,
          borderRadius: 2,
          transition: 'width var(--transition-metric)',
        }} />
      </div>
    </div>
  );
}
