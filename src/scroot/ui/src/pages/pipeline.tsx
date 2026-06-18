import { useState } from 'react'
import { PageWrapper } from '../components/layout/page-wrapper'
import { RadioCard } from '../components/forms/radio-card'
import { ThresholdSlider } from '../components/forms/threshold-slider'
import { StatRow } from '../components/data/stat-row'

const CHECKS = [
  { id: 'pending', title: 'Pending review (23)', desc: 'Records flagged for human review', default: true },
  { id: 'hardfails', title: 'Hard fails only', desc: 'Records below the fail floor — correction most needed', default: false },
  { id: 'all', title: 'All records', desc: 'Every record in the store, regardless of status', default: false },
]

export function Pipeline() {
  const [mode, setMode] = useState('auto')
  const [checks, setChecks] = useState<Record<string, boolean>>({ pending: true })
  const [minImprovement, setMinImprovement] = useState(0.15)
  const [minAbsolute, setMinAbsolute] = useState(0.7)

  return (
    <PageWrapper>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* LEFT */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">run mode</div>
            <RadioCard
              label="Generate drafts only"
              badge={<span className="pill pill-b">human review</span>}
              description="LLM fills every record's correction field. You review each one before committing."
              selected={mode === 'drafts'}
              onChange={() => setMode('drafts')}
            />
            <RadioCard
              label="Auto-commit if NLI passes"
              badge={<span className="pill pill-g">NLI gated</span>}
              description="Corrections auto-committed when NLI improvement ≥ threshold. Failures stay in queue for review."
              selected={mode === 'auto'}
              onChange={() => setMode('auto')}
            />
            <RadioCard
              label="Fully autonomous"
              badge={
                <span className="pill" style={{ background: 'rgba(255,255,255,0.04)', color: 'var(--text-3)' }}>
                  enterprise only
                </span>
              }
              description="Requires audit trail and approval policy. Available in Scroot Cloud with RBAC."
              selected={false}
              disabled
              onChange={() => {}}
            />
          </div>

          <div className="card">
            <div className="card-h">which records to process</div>
            {CHECKS.map((c, i) => (
              <label
                key={c.id}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 10,
                  padding: '10px 0',
                  borderBottom: i === CHECKS.length - 1 ? 'none' : '0.5px solid var(--border)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
              >
                <input
                  type="checkbox"
                  checked={!!checks[c.id]}
                  onChange={() => setChecks((s) => ({ ...s, [c.id]: !s[c.id] }))}
                  style={{ accentColor: 'var(--accent)', marginTop: 2 }}
                />
                <div>
                  <div style={{ fontSize: 12, fontWeight: 500 }}>{c.title}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-3)' }}>{c.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* RIGHT */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">auto-commit thresholds</div>
            <div style={{ marginBottom: 18 }}>
              <ThresholdSlider
                label="Min IQS improvement to auto-commit"
                value={minImprovement}
                onChange={setMinImprovement}
                min={0}
                max={0.5}
                step={0.01}
                showSign
                hint="Below this delta, record goes back to review queue."
              />
            </div>
            <ThresholdSlider
              label="Min absolute IQS after correction"
              value={minAbsolute}
              onChange={setMinAbsolute}
              min={0}
              max={1}
              step={0.01}
              hint="Even if delta is met, this floor must be reached."
            />
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h">preview</div>
            <StatRow label="Records to process" value="23" />
            <StatRow label="Est. LLM calls" value="23" />
            <StatRow label="Est. duration" value="~92s (4s/record)" />
            <StatRow
              label="Provider"
              border={false}
              value={<span style={{ fontSize: 11 }}>API (meta-llama/llama-3)</span>}
            />
          </div>

          <button className="btn-run">▶ Run pipeline on 23 records</button>
          <div style={{ fontSize: 10, color: 'var(--text-4)', textAlign: 'center', marginTop: 8 }}>
            NLI re-scores every correction before any commit. The LLM is the intern, NLI is the senior reviewer.
          </div>
        </div>
      </div>
    </PageWrapper>
  )
}
