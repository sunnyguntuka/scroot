import type { EvidenceEntry } from '../../lib/types'

const CIRCLED = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']

type Tone = 'hi' | 'mid' | 'lo'

interface SentenceRowProps {
  entry: EvidenceEntry & { tone?: Tone; numFlag?: string; chunkTag?: string }
  index: number
}

function toneFromScore(score: number | null | undefined): Tone {
  if (score == null) return 'lo'
  if (score >= 0.8) return 'hi'
  if (score >= 0.5) return 'mid'
  return 'lo'
}

const TONE_STYLES: Record<Tone, { border: string; bg: string }> = {
  hi: { border: 'rgba(52,211,153,0.35)', bg: 'rgba(52,211,153,0.015)' },
  mid: { border: 'rgba(251,191,36,0.35)', bg: 'rgba(251,191,36,0.015)' },
  lo: { border: 'transparent', bg: 'transparent' },
}

const BADGE_STYLES: Record<Tone, { bg: string; color: string }> = {
  hi: { bg: 'var(--green-dim)', color: 'var(--green)' },
  mid: { bg: 'var(--amber-dim)', color: 'var(--amber)' },
  lo: { bg: 'var(--red-dim)', color: 'var(--red)' },
}

function chunkBorder(tone: Tone): string {
  if (tone === 'hi') return 'rgba(52,211,153,0.3)'
  if (tone === 'mid') return 'rgba(248,113,113,0.3)'
  return 'rgba(248,113,113,0.3)'
}

export function SentenceRow({ entry, index }: SentenceRowProps) {
  const tone: Tone = entry.tone ?? toneFromScore(entry.entailment_score)
  const ts = TONE_STYLES[tone]
  const badge = BADGE_STYLES[tone]
  const score = entry.entailment_score ?? entry.mini_iqs ?? 0
  const dims = entry.mini_dims ?? {}

  // render sentence with optional numeric-flag highlight
  let sentenceNode: React.ReactNode = entry.response_sentence
  if (entry.numFlag) {
    const parts = entry.response_sentence.split(entry.numFlag)
    if (parts.length === 2) {
      sentenceNode = (
        <>
          {parts[0]}
          <span style={{ background: 'var(--red-dim)', padding: '1px 5px', borderRadius: 3, color: 'var(--red)' }}>
            {entry.numFlag}
          </span>
          {parts[1]}
        </>
      )
    }
  }

  return (
    <div
      className="sent"
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 50px 280px',
        gap: 16,
        alignItems: 'start',
        padding: '14px 16px',
        borderRadius: 8,
        marginBottom: 4,
        cursor: 'pointer',
        borderLeft: `2.5px solid ${ts.border}`,
        background: ts.bg,
        outline: tone === 'mid' ? '1px solid rgba(251,191,36,0.12)' : undefined,
      }}
    >
      <div style={{ fontSize: 13, lineHeight: 1.75, color: 'rgba(244,241,236,0.75)' }}>
        <span style={{ color: 'var(--text-3)', marginRight: 6, fontFamily: 'var(--mono)', fontSize: 11 }}>
          {CIRCLED[index] ?? index + 1}
        </span>
        {sentenceNode}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, paddingTop: 4 }}>
        <div
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 11,
            fontWeight: 500,
            padding: '3px 10px',
            borderRadius: 5,
            background: badge.bg,
            color: badge.color,
          }}
        >
          {score.toFixed(2)}
        </div>
        <div
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'var(--text-4)',
            textAlign: 'center',
            lineHeight: 1.4,
          }}
        >
          {Object.entries(dims).map(([k, v]) => `${k}${v.toFixed(2).replace(/^0/, '')}`).join(' ')}
        </div>
      </div>

      <div
        style={{
          fontSize: 11.5,
          lineHeight: 1.65,
          color: 'var(--text-3)',
          padding: '10px 14px',
          background: 'var(--bg-2)',
          borderRadius: 6,
          borderLeft: `2px solid ${chunkBorder(tone)}`,
        }}
      >
        <div
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'var(--text-4)',
            marginBottom: 5,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          {entry.chunk_source}
          {entry.chunkTag && (
            <span
              style={{
                fontSize: 8,
                padding: '1px 7px',
                borderRadius: 3,
                background: 'var(--green-dim)',
                color: 'rgba(52,211,153,0.6)',
              }}
            >
              {entry.chunkTag}
            </span>
          )}
        </div>
        {entry.best_matching_chunk}
      </div>
    </div>
  )
}
