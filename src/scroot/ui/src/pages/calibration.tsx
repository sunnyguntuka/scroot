import { PageWrapper } from '../components/layout/page-wrapper'
import { MOCK_CALIBRATION_HISTORY } from '../lib/mock-data'

const PILL_CLASS = { g: 'pill pill-g', y: 'pill pill-y', b: 'pill pill-b' } as const

const tdStyle: React.CSSProperties = {
  padding: 12,
  borderBottom: '0.5px solid var(--border)',
  fontSize: 12.5,
  transition: 'background 0.15s',
}

export function Calibration() {
  return (
    <PageWrapper>
      {/* Row 1 — 3 stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginBottom: 20 }}>
        <div className="card" style={{ textAlign: 'center', padding: 24 }}>
          <div className="card-h" style={{ textAlign: 'left' }}>method</div>
          <div style={{ fontSize: 20, fontWeight: 500, marginBottom: 6 }}>
            <span className="pill pill-g" style={{ fontSize: 12, padding: '4px 14px' }}>
              labeled
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-3)' }}>human-verified ground truth</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 24 }}>
          <div className="card-h" style={{ textAlign: 'left' }}>labeled samples</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 32, fontWeight: 500, marginBottom: 6 }}>248</div>
          <div style={{ fontSize: 11, color: 'var(--text-3)' }}>target: 500 for high confidence</div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 24 }}>
          <div className="card-h" style={{ textAlign: 'left' }}>last calibrated</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 32, fontWeight: 500, marginBottom: 6 }}>2d</div>
          <div style={{ fontSize: 11, color: 'var(--text-3)' }}>Jun 16, 2026</div>
        </div>
      </div>

      {/* Row 2 — threshold viz */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-h">entailment threshold</div>
        <div style={{ position: 'relative', height: 80, margin: '20px 0 10px' }}>
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              top: 36,
              height: 6,
              borderRadius: 3,
              background:
                'linear-gradient(90deg, rgba(248,113,113,0.15) 0%, rgba(248,113,113,0.15) 52%, rgba(52,211,153,0.15) 52%, rgba(52,211,153,0.15) 100%)',
            }}
          />
          <div
            style={{
              position: 'absolute',
              left: '52%',
              top: 20,
              transform: 'translateX(-50%)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <div
              style={{
                fontFamily: 'var(--mono)',
                fontSize: 14,
                fontWeight: 500,
                color: 'var(--accent)',
                marginBottom: 4,
              }}
            >
              0.52
            </div>
            <div style={{ width: 2, height: 32, background: 'var(--accent)', borderRadius: 1 }} />
            <div style={{ fontSize: 9, color: 'var(--text-3)', marginTop: 4 }}>threshold</div>
          </div>
          <div style={{ position: 'absolute', left: 8, top: 24, fontSize: 10, color: 'var(--red)' }}>reject</div>
          <div style={{ position: 'absolute', right: 8, top: 24, fontSize: 10, color: 'var(--green)' }}>accept</div>
          <div
            style={{ position: 'absolute', left: 0, bottom: 0, fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-4)' }}
          >
            0.0
          </div>
          <div
            style={{ position: 'absolute', right: 0, bottom: 0, fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-4)' }}
          >
            1.0
          </div>
        </div>
      </div>

      {/* Row 3 — history table */}
      <div className="card">
        <div className="card-h">calibration history</div>
        <table className="tbl" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['date', 'method', 'samples', 'threshold', 'status'].map((h) => (
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
            {MOCK_CALIBRATION_HISTORY.map((r, i) => (
              <tr key={i} style={{ cursor: 'pointer' }}>
                <td style={{ ...tdStyle, fontFamily: 'var(--mono)', fontWeight: 500 }}>{r.date}</td>
                <td style={tdStyle}>
                  {r.method.pill ? (
                    <span className={PILL_CLASS[r.method.pill]}>{r.method.label}</span>
                  ) : (
                    <span style={{ color: 'var(--text-3)', fontSize: 11 }}>{r.method.label}</span>
                  )}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    fontFamily: r.samples.muted ? 'var(--sans)' : 'var(--mono)',
                    fontWeight: r.samples.muted ? 400 : 500,
                    color: r.samples.muted ? 'var(--text-3)' : r.samples.color,
                    fontSize: r.samples.muted ? 11 : 12.5,
                  }}
                >
                  {r.samples.label}
                </td>
                <td style={{ ...tdStyle, fontFamily: 'var(--mono)', fontWeight: 500 }}>{r.threshold}</td>
                <td style={tdStyle}>
                  {r.status.pill ? (
                    <span className={PILL_CLASS[r.status.pill]}>{r.status.label}</span>
                  ) : (
                    <span style={{ color: 'var(--text-3)', fontSize: 11 }}>{r.status.label}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PageWrapper>
  )
}
