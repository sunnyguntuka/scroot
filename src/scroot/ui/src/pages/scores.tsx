import { PageWrapper } from '../components/layout/page-wrapper'
import { Sparkline } from '../components/data/sparkline'
import { MOCK_ALL_SCORES } from '../lib/mock-data'

const PILL_CLASS = { r: 'pill pill-r', y: 'pill pill-y' } as const

const HEADERS = [
  'query',
  'IQS',
  'trend',
  'groundedness',
  'completeness',
  'relevance',
  'consistency',
  'numeric',
  'flags',
  'time',
]

const tdStyle: React.CSSProperties = {
  padding: 12,
  borderBottom: '0.5px solid var(--border)',
  fontSize: 12.5,
  transition: 'background 0.15s',
}

const mono = (color?: string): React.CSSProperties => ({
  ...tdStyle,
  fontFamily: 'var(--mono)',
  fontWeight: 500,
  color,
})

export function Scores() {
  return (
    <PageWrapper>
      <div className="card">
        <div
          className="card-h"
          style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
        >
          all scores
          <div style={{ display: 'flex', gap: 6 }}>
            <button className="btn" style={{ fontSize: 10 }}>
              filter
            </button>
            <button className="btn" style={{ fontSize: 10 }}>
              export CSV
            </button>
          </div>
        </div>
        <table className="tbl" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {HEADERS.map((h) => (
                <th
                  key={h}
                  style={{
                    textAlign: 'left',
                    fontWeight: 500,
                    fontSize: 10,
                    color: 'var(--text-3)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    padding: '10px 12px',
                    borderBottom: '0.5px solid var(--border)',
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOCK_ALL_SCORES.map((r, i) => (
              <tr key={i} style={{ cursor: 'pointer' }}>
                <td style={tdStyle}>
                  <span
                    style={{
                      maxWidth: 300,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      color: 'var(--text-2)',
                      display: 'block',
                    }}
                  >
                    {r.query}
                  </span>
                </td>
                <td style={mono(r.iqsColor)}>{r.iqs.toFixed(2)}</td>
                <td style={tdStyle}>
                  <Sparkline data={r.spark} declining={r.sparkDeclining} />
                </td>
                <td style={mono(r.groundednessColor)}>{r.groundedness.toFixed(2)}</td>
                <td style={mono(r.completenessColor)}>{r.completeness.toFixed(2)}</td>
                <td style={mono()}>{r.relevance.toFixed(2)}</td>
                <td style={mono()}>{r.consistency.toFixed(2)}</td>
                <td style={mono(r.numericColor)}>{r.numeric.toFixed(2)}</td>
                <td style={tdStyle}>
                  {r.flag ? <span className={PILL_CLASS[r.flag.pill]}>{r.flag.label}</span> : '—'}
                </td>
                <td style={{ ...tdStyle, color: 'var(--text-3)', fontSize: 11 }}>{r.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageWrapper>
  )
}
