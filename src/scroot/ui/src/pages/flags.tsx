import { BarChart, Bar, XAxis, ResponsiveContainer, Cell } from 'recharts'
import { PageWrapper } from '../components/layout/page-wrapper'
import { StatRow } from '../components/data/stat-row'
import {
  MOCK_FLAG_VOLUME,
  FLAG_DAYS,
  FLAG_SERIES,
  MOCK_FLAG_STATS,
  MOCK_FLAGS_TABLE,
} from '../lib/mock-data'

const PILL_CLASS = { r: 'pill pill-r', y: 'pill pill-y' } as const

const tdStyle: React.CSSProperties = {
  padding: 12,
  borderBottom: '0.5px solid var(--border)',
  fontSize: 12.5,
  transition: 'background 0.15s',
}

// Resolve series colors to concrete hex (recharts needs computed values, not CSS vars).
const SERIES_HEX = ['#F87171', '#F87171', '#FBBF24', '#FBBF24', '#FBBF24']
const SERIES_OPACITY = [0.3, 0.45, 0.6, 0.75, 0.9]

const chartData = MOCK_FLAG_VOLUME.map((day, i) => {
  const row: Record<string, number | string> = { day: FLAG_DAYS[i] }
  FLAG_SERIES.forEach((s, si) => {
    row[s.key] = day[si]
  })
  return row
})

export function Flags() {
  return (
    <PageWrapper>
      {/* Row 1 — chart + stats */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 14, marginBottom: 20 }}>
        <div className="card">
          <div className="card-h">flag volume — last 7 days</div>
          <div style={{ height: 120, paddingTop: 10 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                <XAxis dataKey="day" hide />
                {FLAG_SERIES.map((s, si) => (
                  <Bar key={s.key} dataKey={s.key} stackId="a" radius={si === 0 ? [0, 0, 2, 2] : [2, 2, 0, 0]}>
                    {chartData.map((_, ci) => (
                      <Cell key={ci} fill={SERIES_HEX[si]} fillOpacity={SERIES_OPACITY[si]} />
                    ))}
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: 8,
              fontSize: 10,
              color: 'var(--text-4)',
            }}
          >
            {FLAG_DAYS.map((d) => (
              <span key={d}>{d}</span>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-h">flag stats</div>
          {MOCK_FLAG_STATS.map((s, i) => (
            <StatRow
              key={s.label}
              label={s.label}
              border={i !== MOCK_FLAG_STATS.length - 1}
              value={<span style={{ color: s.color }}>{s.value}</span>}
            />
          ))}
        </div>
      </div>

      {/* Row 2 — flagged scores table */}
      <div className="card">
        <div className="card-h">flagged scores</div>
        <table className="tbl" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['query', 'IQS', 'flag', 'trigger metric', 'time'].map((h) => (
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
            {MOCK_FLAGS_TABLE.map((r, i) => (
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
                <td style={{ ...tdStyle, fontFamily: 'var(--mono)', fontWeight: 500, color: r.iqsColor }}>
                  {r.iqs.toFixed(2)}
                </td>
                <td style={tdStyle}>
                  <span className={PILL_CLASS[r.flag.pill]}>{r.flag.label}</span>
                </td>
                <td style={{ ...tdStyle, fontFamily: 'var(--mono)', fontWeight: 500, color: r.triggerColor }}>
                  {r.trigger}
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
