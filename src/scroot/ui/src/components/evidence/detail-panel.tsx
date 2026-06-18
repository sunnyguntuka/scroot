interface DetailMetric {
  label: string
  value: number
  color: string
}

interface NumericClaim {
  claim: string
  verdict: string
  found?: boolean
  color?: string
}

interface DetailPanelProps {
  sentence: string
  index: number
  dims: DetailMetric[]
  note?: string
  numericClaims?: NumericClaim[]
}

export function DetailPanel({ sentence, index, dims, note, numericClaims }: DetailPanelProps) {
  return (
    <div
      style={{
        background: 'var(--bg-2)',
        border: '0.5px solid var(--border)',
        borderRadius: 12,
        padding: '20px 22px',
        marginTop: 8,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 12,
          fontWeight: 500,
          color: 'var(--text-2)',
          marginBottom: 16,
        }}
      >
        <span style={{ color: 'var(--amber)', fontSize: 15 }}>⚠</span>
        sentence {index} — {sentence}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4,1fr)',
          gap: 10,
          marginBottom: 16,
        }}
      >
        {dims.map((d) => (
          <div key={d.label} style={{ padding: '12px 14px', background: 'var(--bg-3)', borderRadius: 8 }}>
            <div style={{ fontSize: 10, color: 'var(--text-3)', marginBottom: 5 }}>{d.label}</div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 18, fontWeight: 500, color: d.color }}>
              {d.value.toFixed(2)}
            </div>
          </div>
        ))}
      </div>

      {note && (
        <div
          style={{
            fontSize: 12,
            color: 'var(--text-2)',
            lineHeight: 1.7,
            padding: '14px 16px',
            background: 'rgba(251,191,36,0.025)',
            borderRadius: 8,
            borderLeft: '2px solid rgba(251,191,36,0.2)',
            marginBottom: 12,
          }}
        >
          {note}
        </div>
      )}

      {numericClaims && numericClaims.length > 0 && (
        <div
          style={{
            padding: '14px 16px',
            background: 'rgba(91,140,255,0.025)',
            borderRadius: 8,
            borderLeft: '2px solid rgba(91,140,255,0.2)',
          }}
        >
          <div
            style={{
              fontSize: 10,
              color: 'rgba(91,140,255,0.6)',
              fontWeight: 500,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              marginBottom: 10,
            }}
          >
            numeric grounding verification
          </div>
          {numericClaims.map((c, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: 12,
                padding: '4px 0',
              }}
            >
              <span style={{ color: 'var(--text-2)' }}>{c.claim}</span>
              <span style={{ fontFamily: 'var(--mono)', fontWeight: 500, color: c.color ?? 'var(--text-2)' }}>
                {c.verdict}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
