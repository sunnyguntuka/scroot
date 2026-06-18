interface StatusPillProps {
  status: 'pending' | 'corrected' | 'passed' | 'failing'
}

const MAP = {
  pending: { bg: 'var(--amber-dim)', color: 'var(--amber)' },
  corrected: { bg: 'var(--accent-dim)', color: 'var(--accent)' },
  passed: { bg: 'var(--green-dim)', color: 'var(--green)' },
  failing: { bg: 'var(--red-dim)', color: 'var(--red)' },
} as const

export function StatusPill({ status }: StatusPillProps) {
  const t = MAP[status]
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        fontSize: 10,
        fontWeight: 500,
        padding: '3px 10px',
        borderRadius: 20,
        background: t.bg,
        color: t.color,
      }}
    >
      {status}
    </span>
  )
}
