import { useState } from 'react'
import { PageWrapper } from '../components/layout/page-wrapper'
import { RadioCard } from '../components/forms/radio-card'
import { ThresholdSlider } from '../components/forms/threshold-slider'
import { useTheme } from '../stores/theme'

const DEFAULT_WEIGHTS = {
  Groundedness: 0.4,
  Completeness: 0.2,
  Relevance: 0.2,
  Consistency: 0.1,
  Confidence: 0.1,
}

export function Settings() {
  const { theme, toggle } = useTheme()
  const [iqsThreshold, setIqsThreshold] = useState(0.8)
  const [weights, setWeights] = useState<Record<string, number>>({ ...DEFAULT_WEIGHTS })
  const [llm, setLlm] = useState('api')

  const total = Object.values(weights).reduce((a, b) => a + b, 0)
  const totalOk = Math.abs(total - 1) < 0.001

  return (
    <PageWrapper>
      <div style={{ maxWidth: 680 }}>
        {/* Scoring */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 18 }}>Scoring</div>
          <div style={{ marginBottom: 20 }}>
            <ThresholdSlider
              label="IQS threshold"
              value={iqsThreshold}
              onChange={setIqsThreshold}
              min={0}
              max={1}
              step={0.01}
              hint="Below this score, records are flagged for review."
            />
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
              <div className="card-h" style={{ margin: 0 }}>
                metric weights
              </div>
              <span
                style={{
                  fontSize: 11,
                  color: totalOk ? 'var(--green)' : 'var(--amber)',
                  fontFamily: 'var(--mono)',
                }}
              >
                Total: {total.toFixed(2)} {totalOk ? '✓' : '!'}
              </span>
            </div>
            {Object.keys(DEFAULT_WEIGHTS).map((k) => (
              <div
                key={k}
                style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '6px 0', fontSize: 12, color: 'var(--text-2)' }}
              >
                <span style={{ width: 110, flexShrink: 0 }}>{k}</span>
                <input
                  type="range"
                  className="slider"
                  min={0}
                  max={1}
                  step={0.05}
                  value={weights[k]}
                  aria-label={`${k} weight`}
                  onChange={(e) => setWeights((w) => ({ ...w, [k]: parseFloat(e.target.value) }))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontFamily: 'var(--mono)', fontWeight: 500, width: 36, textAlign: 'right', flexShrink: 0 }}>
                  {weights[k].toFixed(2)}
                </span>
              </div>
            ))}
            <button
              className="btn"
              style={{ marginTop: 10, fontSize: 10 }}
              onClick={() => setWeights({ ...DEFAULT_WEIGHTS })}
            >
              Reset to defaults
            </button>
          </div>
        </div>

        {/* LLM corrector */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 18 }}>LLM corrector</div>
          <RadioCard
            label="Disabled"
            description="No LLM correction. NLI scoring only."
            selected={llm === 'disabled'}
            onChange={() => setLlm('disabled')}
          />
          <RadioCard
            label="Local LLM"
            description="Runs on your machine. Data never leaves your environment. No API key."
            selected={llm === 'local'}
            onChange={() => setLlm('local')}
          />
          <RadioCard
            label="LLM via API"
            description="Send corrections to an external provider. Requires an API key."
            selected={llm === 'api'}
            onChange={() => setLlm('api')}
          />
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 4 }}>API key</div>
            <input type="password" defaultValue="sk-••••••••••••••••" className="text-input" aria-label="API key" />
            <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 4 }}>
              Works with OpenAI, Anthropic, Google Gemini, and OpenRouter. Provider is detected automatically.
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 4 }}>Model</div>
            <input type="text" defaultValue="meta-llama/llama-3" className="text-input" aria-label="Model" />
          </div>
        </div>

        {/* Appearance */}
        <div className="card">
          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 18 }}>Appearance</div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 500 }}>Theme</div>
              <div style={{ fontSize: 11, color: 'var(--text-3)' }}>Switch between dark and light mode</div>
            </div>
            <button className="btn" style={{ gap: 6 }} onClick={toggle}>
              {theme === 'dark' ? '☀ Light mode' : '☾ Dark mode'}
            </button>
          </div>
        </div>
      </div>
    </PageWrapper>
  )
}
