import { useEffect, useState } from 'react'

interface DistributionBin {
  range: string
  count: number
  color: string
}

interface DistributionChartProps {
  bins: DistributionBin[]
}

// Plain-div bar chart matching mockup .dist. count is treated as a percentage height (0-100).
export function DistributionChart({ bins }: DistributionChartProps) {
  const [animated, setAnimated] = useState(false)

  useEffect(() => {
    const reduced =
      typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduced) {
      setAnimated(true)
      return
    }
    const t = setTimeout(() => setAnimated(true), 100)
    return () => clearTimeout(t)
  }, [])

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 90 }}>
        {bins.map((b, i) => (
          <div
            key={b.range}
            title={`${b.range}: ${b.count}%`}
            style={{
              flex: 1,
              borderRadius: '3px 3px 0 0',
              background: b.color,
              minHeight: 4,
              height: animated ? `${b.count}%` : 2,
              transition: `height 0.8s cubic-bezier(0.22,1,0.36,1) ${i * 80}ms, opacity 0.15s`,
              cursor: 'pointer',
            }}
          />
        ))}
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 6,
          fontSize: 9,
          color: 'var(--text-4)',
          fontFamily: 'var(--mono)',
        }}
      >
        {bins.map((b) => (
          <span key={b.range}>{b.range}</span>
        ))}
      </div>
    </div>
  )
}
