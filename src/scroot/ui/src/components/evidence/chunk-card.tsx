interface ChunkCardProps {
  source: string
  text: string
  matchCount?: number
}

// Context source card (right-column .ctx in mockup).
export function ChunkCard({ source, text, matchCount }: ChunkCardProps) {
  return (
    <div
      style={{
        padding: '11px 14px',
        background: 'rgba(255,255,255,0.015)',
        border: '0.5px solid var(--border)',
        borderRadius: 8,
        marginBottom: 8,
      }}
    >
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          color: 'var(--text-4)',
          marginBottom: 5,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>{source}</span>
        {matchCount != null && (
          <span
            style={{
              fontSize: 8,
              padding: '1px 7px',
              borderRadius: 3,
              background: 'var(--green-dim)',
              color: 'rgba(52,211,153,0.6)',
            }}
          >
            {matchCount} matches
          </span>
        )}
      </div>
      <div style={{ fontSize: 11.5, lineHeight: 1.65, color: 'var(--text-3)' }}>{text}</div>
    </div>
  )
}
