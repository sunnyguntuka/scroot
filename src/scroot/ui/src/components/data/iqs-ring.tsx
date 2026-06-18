import { useEffect, useState } from 'react'

interface IqsRingProps {
  score: number
  size?: number
}

export function IqsRing({ score, size = 96 }: IqsRingProps) {
  const r = (size / 96) * 42
  const stroke = 4
  const circumference = 2 * Math.PI * r
  const [offset, setOffset] = useState(circumference)

  useEffect(() => {
    const reduced =
      typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const target = circumference * (1 - score)
    if (reduced) {
      setOffset(target)
      return
    }
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setOffset(target)))
    return () => cancelAnimationFrame(id)
  }, [score, circumference])

  return (
    <div style={{ textAlign: 'center', padding: '8px 0 18px' }}>
      <div style={{ position: 'relative', display: 'inline-block', width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--ring-track)" strokeWidth={stroke} />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.22,1,0.36,1)' }}
          />
        </svg>
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%,-50%)',
            fontFamily: 'var(--mono)',
            fontSize: 26,
            fontWeight: 600,
          }}
        >
          {score.toFixed(2)}
        </div>
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 6 }}>composite IQS</div>
    </div>
  )
}
