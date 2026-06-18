import { PageWrapper } from '../components/layout/page-wrapper'
import { MetricCard } from '../components/data/metric-card'
import { MetricBar } from '../components/data/metric-bar'
import { Heatmap } from '../components/data/heatmap'
import { DistributionChart } from '../components/data/distribution-chart'
import { Sparkline } from '../components/data/sparkline'
import { StatRow } from '../components/data/stat-row'
import {
  MOCK_METRICS,
  MOCK_FLAGS,
  MOCK_METRIC_BARS,
  MOCK_DISTRIBUTION,
  MOCK_DISTRIBUTION_SUMMARY,
  MOCK_CALIBRATION,
  MOCK_RECENT_SCORES,
} from '../lib/mock-data'

function CardHeader({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return <div className="card-h" style={style}>{children}</div>
}

const PILL_CLASS = { g: 'pill pill-g', y: 'pill pill-y', r: 'pill pill-r', b: 'pill pill-b' } as const

export function Overview() {
  return (
    <PageWrapper>
      {/* Row 1 — metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 20 }}>
        {MOCK_METRICS.map((m) => (
          <MetricCard
            key={m.label}
            label={m.label}
            value={m.value}
            delta={m.delta}
            deltaDirection={m.deltaDirection}
            decimals={m.decimals}
            suffix={m.suffix}
          />
        ))}
      </div>

      {/* Row 2 — heatmap + flag breakdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: 14, marginBottom: 20 }}>
        <div className="card">
          <CardHeader>IQS distribution — last 7 days</CardHeader>
          <Heatmap />
        </div>
        <div className="card">
          <CardHeader>flag breakdown</CardHeader>
          {MOCK_FLAGS.map((f, i) => (
            <div
              key={f.name}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 0',
                borderBottom: i === MOCK_FLAGS.length - 1 ? 'none' : '0.5px solid var(--border)',
              }}
            >
              <div style={{ width: 7, height: 7, borderRadius: '50%', flexShrink: 0, background: f.color }} />
              <div style={{ fontSize: 12.5, flex: 1, fontFamily: 'var(--mono)' }}>{f.name}</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 500, width: 40, textAlign: 'right' }}>
                {f.count}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', width: 44, textAlign: 'right' }}>{f.pct}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Row 3 — per-metric averages + distribution + calibration */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginBottom: 20 }}>
        <div className="card">
          <CardHeader>per-metric averages</CardHeader>
          {MOCK_METRIC_BARS.map((b) => (
            <div
              key={b.label}
              style={
                b.accent
                  ? { marginTop: 4, paddingTop: 6, borderTop: '0.5px solid var(--border)' }
                  : undefined
              }
            >
              <MetricBar
                label={b.label}
                value={b.value}
                color={b.color}
                labelColor={b.accent ? 'var(--accent)' : undefined}
              />
            </div>
          ))}
        </div>

        <div className="card">
          <CardHeader>score distribution</CardHeader>
          <DistributionChart bins={MOCK_DISTRIBUTION} />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 11 }}>
            <div>
              <span style={{ color: 'var(--text-3)' }}>fail</span>{' '}
              <span style={{ color: 'var(--red)', fontFamily: 'var(--mono)', fontWeight: 500 }}>
                {MOCK_DISTRIBUTION_SUMMARY.fail}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-3)' }}>warn</span>{' '}
              <span style={{ color: 'var(--amber)', fontFamily: 'var(--mono)', fontWeight: 500 }}>
                {MOCK_DISTRIBUTION_SUMMARY.warn}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-3)' }}>pass</span>{' '}
              <span style={{ color: 'var(--green)', fontFamily: 'var(--mono)', fontWeight: 500 }}>
                {MOCK_DISTRIBUTION_SUMMARY.pass}
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <CardHeader>calibration</CardHeader>
          {MOCK_CALIBRATION.map((c, i) => (
            <StatRow
              key={c.label}
              label={c.label}
              border={i !== MOCK_CALIBRATION.length - 1}
              value={
                c.pill ? (
                  <span className={PILL_CLASS[c.pill]}>{c.value}</span>
                ) : (
                  <span style={{ color: c.color }}>{c.value}</span>
                )
              }
            />
          ))}
        </div>
      </div>

      {/* Row 4 — recent scores */}
      <div className="card">
        <CardHeader style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          recent scores
          <button className="btn" style={{ fontSize: 10, padding: '4px 10px' }}>
            view all →
          </button>
        </CardHeader>
        <table className="tbl" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['query', 'IQS', 'trend', 'groundedness', 'completeness', 'flags', 'time'].map((h, i) => (
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
                    width: i === 2 ? 90 : i === 6 ? 70 : undefined,
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOCK_RECENT_SCORES.map((r, i) => (
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
                  <Sparkline data={r.spark} declining={r.sparkDeclining} />
                </td>
                <td style={{ ...tdStyle, fontFamily: 'var(--mono)', fontWeight: 500, color: r.groundednessColor }}>
                  {r.groundedness.toFixed(2)}
                </td>
                <td style={{ ...tdStyle, fontFamily: 'var(--mono)', fontWeight: 500, color: r.completenessColor }}>
                  {r.completeness.toFixed(2)}
                </td>
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

const tdStyle: React.CSSProperties = {
  padding: 12,
  borderBottom: '0.5px solid var(--border)',
  fontSize: 12.5,
  transition: 'background 0.15s',
}
