import { useEffect, useState } from 'react'

interface MetricBarProps {
  label: string
  value: number
  color: string
  labelColor?: string
}

// Compact metric bar matching .mbar-c in mockup.
export function MetricBar({ label, value, color, labelColor }: MetricBarProps) {
  const [width, setWidth] = useState(0)
  const pct = Math.round(value * 100)

  useEffect(() => {
    const reduced =
      typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced) {
      setWidth(pct)
      return
    }
    const id = requestAnimationFrame(() =>
      requestAnimationFrame(() => setWidth(pct))
    )
    return () => cancelAnimationFrame(id)
  }, [pct])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 0' }}>
      <span
        style={{ fontSize: 11, color: labelColor ?? 'var(--text-2)', width: 88, flexShrink: 0 }}
      >
        {label}
      </span>
      <div
        style={{
          flex: 1,
          height: 4,
          background: 'rgba(255,255,255,0.04)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            borderRadius: 2,
            background: color,
            width: `${width}%`,
            transition: 'width 1s cubic-bezier(0.22,1,0.36,1)',
          }}
        />
      </div>
      <span
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 11,
          fontWeight: 500,
          width: 32,
          textAlign: 'right',
          flexShrink: 0,
          color,
        }}
      >
        {value.toFixed(2)}
      </span>
    </div>
  )
}
