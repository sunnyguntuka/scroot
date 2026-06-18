interface FlagPillProps {
  flag: string
}

const RED_FLAGS = new Set(['hallucination_risk', 'ungrounded'])

export function FlagPill({ flag }: FlagPillProps) {
  const isRed = RED_FLAGS.has(flag)
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        fontSize: 10,
        fontWeight: 500,
        padding: '3px 10px',
        borderRadius: 20,
        background: isRed ? 'var(--red-dim)' : 'var(--amber-dim)',
        color: isRed ? 'var(--red)' : 'var(--amber)',
      }}
    >
      {flag}
    </span>
  )
}
