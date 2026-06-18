interface ScoreBadgeProps {
  score: number
  size?: 'sm' | 'md'
}

function tone(score: number): { bg: string; color: string } {
  if (score >= 0.8) return { bg: 'var(--green-dim)', color: 'var(--green)' }
  if (score >= 0.5) return { bg: 'var(--amber-dim)', color: 'var(--amber)' }
  return { bg: 'var(--red-dim)', color: 'var(--red)' }
}

export function ScoreBadge({ score, size = 'sm' }: ScoreBadgeProps) {
  const t = tone(score)
  const md = size === 'md'
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        fontFamily: 'var(--mono)',
        fontSize: md ? 13 : 11,
        fontWeight: 500,
        padding: md ? '5px 12px' : '3px 10px',
        borderRadius: 5,
        background: t.bg,
        color: t.color,
      }}
    >
      {score.toFixed(2)}
    </span>
  )
}
