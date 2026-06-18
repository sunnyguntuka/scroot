import { useState } from 'react'
import { PageWrapper } from '../components/layout/page-wrapper'
import { RadioCard } from '../components/forms/radio-card'
import { ThresholdSlider } from '../components/forms/threshold-slider'

const RECORD_CHECKS = [
  { id: 'corrected', label: 'Reviewed + corrected', default: true },
  { id: 'rejected', label: 'Reviewed + rejected', default: true },
  { id: 'pending', label: 'Pending (unreviewed)', default: false },
]

const PREVIEW = `{
  "id": "uuid-0042",
  "query": "Summarize Q3 earnings report",
  "original_response": "Q3 was decent...",
  "corrected_response": "Q3 revenue grew 12% YoY...",
  "iqs_before": 0.54,
  "iqs_after": 0.81,
  "flags": ["consistency", "completeness"],
  "model": "gpt-4o",
  "agent_id": "support-bot-v2"
}`

export function Export() {
  const [format, setFormat] = useState('jsonl')
  const [checks, setChecks] = useState<Record<string, boolean>>({ corrected: true, rejected: true })
  const [improvement, setImprovement] = useState(0)

  return (
    <PageWrapper>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* LEFT */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">format</div>
            <RadioCard
              label="JSONL"
              badge={<span style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 400 }}>fine-tuning format</span>}
              selected={format === 'jsonl'}
              onChange={() => setFormat('jsonl')}
            />
            <RadioCard
              label="CSV"
              badge={<span style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 400 }}>flat, all fields</span>}
              selected={format === 'csv'}
              onChange={() => setFormat('csv')}
            />
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">records to include</div>
            {RECORD_CHECKS.map((c, i) => (
              <label
                key={c.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '10px 0',
                  borderBottom: i === RECORD_CHECKS.length - 1 ? 'none' : '0.5px solid var(--border)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                <input
                  type="checkbox"
                  checked={!!checks[c.id]}
                  onChange={() => setChecks((s) => ({ ...s, [c.id]: !s[c.id] }))}
                  style={{ accentColor: 'var(--accent)' }}
                />
                <span style={{ fontSize: 12 }}>{c.label}</span>
              </label>
            ))}
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">IQS improvement filter</div>
            <ThresholdSlider
              label="Only records where IQS improved by at least:"
              value={improvement}
              onChange={setImprovement}
              min={0}
              max={1}
              step={0.01}
            />
          </div>

          <div className="card">
            <div className="card-h">date range</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 4 }}>From</div>
                <input type="date" className="date-input" aria-label="From date" />
              </div>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 4 }}>To</div>
                <input type="date" className="date-input" aria-label="To date" />
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 500 }}>248</span>
              <span style={{ fontSize: 12, color: 'var(--text-2)' }}>records matched</span>
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: 'var(--text-2)' }}>Fine-tuning readiness</span>
                <span style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>248 / 500</span>
              </div>
              <div style={{ height: 4, background: 'var(--bg-3)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: '49.6%', background: 'var(--accent)', borderRadius: 2 }} />
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 4 }}>
                252 more corrected records recommended
              </div>
            </div>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">preview (first 3 records)</div>
            <pre className="code-preview">{PREVIEW}</pre>
          </div>

          <button className="btn-run">⤓ Download {format === 'jsonl' ? 'JSONL' : 'CSV'}</button>
          <div
            style={{
              fontSize: 10,
              color: 'var(--text-4)',
              textAlign: 'center',
              marginTop: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 4,
            }}
          >
            <span>⚡</span> Push to S3/GCS and compliance exports available in{' '}
            <a href="#" style={{ color: 'var(--accent)', textDecoration: 'none' }}>
              Scroot Cloud →
            </a>
          </div>
        </div>
      </div>
    </PageWrapper>
  )
}
