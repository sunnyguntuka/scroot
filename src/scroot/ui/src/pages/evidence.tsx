import { PageWrapper } from '../components/layout/page-wrapper'
import { SentenceRow } from '../components/evidence/sentence-row'
import { DetailPanel } from '../components/evidence/detail-panel'
import { ChunkCard } from '../components/evidence/chunk-card'
import { IqsRing } from '../components/data/iqs-ring'
import { MetricBar } from '../components/data/metric-bar'
import { StatRow } from '../components/data/stat-row'
import { MOCK_EVIDENCE_RECORD } from '../lib/mock-data'

const PILL_CLASS = { y: 'pill pill-y', g: 'pill pill-g', r: 'pill pill-r' } as const

export function Evidence() {
  const rec = MOCK_EVIDENCE_RECORD
  const weakest = rec.entries[rec.detail.sentence - 1]

  return (
    <PageWrapper>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 20 }}>
        {/* LEFT */}
        <div>
          <div
            style={{
              background: 'rgba(91,140,255,0.03)',
              borderLeft: '2.5px solid var(--accent)',
              borderRadius: '0 8px 8px 0',
              padding: '14px 20px',
              marginBottom: 18,
            }}
          >
            <div
              style={{
                fontSize: 10,
                color: 'rgba(91,140,255,0.6)',
                textTransform: 'uppercase',
                letterSpacing: '0.8px',
                fontWeight: 500,
                marginBottom: 6,
              }}
            >
              query
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.7, color: 'var(--text-2)' }}>{rec.query}</div>
          </div>

          <div className="card" style={{ padding: '20px 22px', marginBottom: 16 }}>
            <div className="card-h">response — per-sentence evidence</div>
            {rec.entries.map((entry, i) => (
              <SentenceRow key={i} entry={entry} index={i} />
            ))}
          </div>

          <DetailPanel
            sentence="weak grounding detected"
            index={rec.detail.sentence}
            dims={rec.detail.dims}
            note={rec.detail.note}
            numericClaims={rec.detail.numericClaims}
          />
        </div>

        {/* RIGHT */}
        <div>
          <div className="card" style={{ marginBottom: 14 }}>
            <div className="card-h">score summary</div>
            <IqsRing score={rec.compositeIqs} />
            {rec.summary.map((s, i) => (
              <StatRow
                key={s.label + i}
                label={s.label}
                border={i !== rec.summary.length - 1}
                value={
                  s.pill ? (
                    <span className={PILL_CLASS[s.pill.kind]}>{s.pill.label}</span>
                  ) : (
                    <span style={{ color: s.color }}>{s.value}</span>
                  )
                }
              />
            ))}
          </div>

          <div className="card" style={{ marginBottom: 14 }}>
            <div className="card-h">metric breakdown</div>
            {rec.metricBreakdown.map((m) => (
              <MetricBar
                key={m.label}
                label={m.label}
                value={m.value}
                color={m.color}
                labelColor={m.accent ? 'var(--accent)' : undefined}
              />
            ))}
          </div>

          <div className="card">
            <div className="card-h">context sources</div>
            {rec.contextSources.map((c) => (
              <ChunkCard
                key={c.source}
                source={c.source}
                text={c.text}
                matchCount={c.matchTag ? parseInt(c.matchTag) : undefined}
              />
            ))}
          </div>
        </div>
      </div>
      {/* weakest reference retained for future deep-link */}
      <span hidden>{weakest?.chunk_source}</span>
    </PageWrapper>
  )
}
