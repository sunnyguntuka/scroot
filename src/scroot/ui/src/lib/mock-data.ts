import type { EvidenceEntry } from './types'

export interface MetricDef {
  label: string
  value: number
  display?: string
  decimals?: number
  suffix?: string
  delta: string
  deltaDirection: 'up' | 'down' | 'neutral'
}

// ─── Overview metric cards (5) ───
export const MOCK_METRICS: MetricDef[] = [
  { label: 'total scores', value: 12847, delta: '↑ 14.2% vs last week', deltaDirection: 'up' },
  { label: 'avg IQS', value: 0.91, decimals: 2, delta: '↑ 0.03 vs baseline', deltaDirection: 'up' },
  { label: 'flag rate', value: 8.4, decimals: 1, suffix: '%', delta: '↑ 1.2% vs last week', deltaDirection: 'down' },
  { label: 'numeric issues', value: 34, delta: '0.26% of scores', deltaDirection: 'neutral' },
  { label: 'p50 latency', value: 840, suffix: 'ms', delta: '↓ 120ms vs last week', deltaDirection: 'up' },
]

// ─── Flag breakdown (overview right card) ───
export interface FlagBreakdown {
  name: string
  count: number
  pct: string
  color: string
}
export const MOCK_FLAGS: FlagBreakdown[] = [
  { name: 'hallucination_risk', count: 142, pct: '1.1%', color: 'var(--red)' },
  { name: 'ungrounded', count: 98, pct: '0.8%', color: 'var(--red)' },
  { name: 'incomplete', count: 412, pct: '3.2%', color: 'var(--amber)' },
  { name: 'off_topic', count: 189, pct: '1.5%', color: 'var(--amber)' },
  { name: 'self_contradictory', count: 67, pct: '0.5%', color: 'var(--amber)' },
]

// ─── Per-metric averages (6 bars) ───
export interface MetricBarDef {
  label: string
  value: number
  color: string
  accent?: boolean
}
export const MOCK_METRIC_BARS: MetricBarDef[] = [
  { label: 'groundedness', value: 0.89, color: 'var(--green)' },
  { label: 'completeness', value: 0.84, color: 'var(--green)' },
  { label: 'relevance', value: 0.93, color: 'var(--green)' },
  { label: 'consistency', value: 0.91, color: 'var(--green)' },
  { label: 'confidence', value: 0.72, color: 'var(--amber)' },
  { label: 'numeric', value: 0.96, color: 'var(--accent)', accent: true },
]

// ─── Score distribution (5 bins) ───
export interface DistBin {
  range: string
  count: number
  color: string
}
export const MOCK_DISTRIBUTION: DistBin[] = [
  { range: '0.0–0.2', count: 4, color: 'var(--red)' },
  { range: '0.2–0.4', count: 8, color: 'var(--red)' },
  { range: '0.4–0.6', count: 12, color: 'var(--amber)' },
  { range: '0.6–0.8', count: 28, color: 'var(--green)' },
  { range: '0.8–1.0', count: 48, color: 'var(--green)' },
]
export const MOCK_DISTRIBUTION_SUMMARY = { fail: '882', warn: '2,118', pass: '9,847' }

// ─── Calibration (overview card stat rows) ───
export interface CalRow {
  label: string
  value: string
  pill?: 'g' | 'y' | 'r' | 'b'
  color?: string
  topBorder?: boolean
}
export const MOCK_CALIBRATION: CalRow[] = [
  { label: 'entailment threshold', value: '0.52' },
  { label: 'method', value: 'labeled', pill: 'g' },
  { label: 'labeled samples', value: '248' },
  { label: 'last calibrated', value: '2d ago', color: 'var(--text-2)' },
  { label: 'regression check', value: 'passing', pill: 'g', topBorder: true },
  { label: 'baseline delta', value: '+0.03', color: 'var(--green)' },
]

// ─── Recent scores (overview, 5 rows) ───
export interface RecentScore {
  query: string
  iqs: number
  iqsColor?: string
  spark: number[]
  sparkDeclining?: boolean
  groundedness: number
  groundednessColor?: string
  completeness: number
  completenessColor?: string
  flag: { label: string; pill: 'r' | 'y' } | null
  time: string
}
export const MOCK_RECENT_SCORES: RecentScore[] = [
  { query: 'What are the key provisions of the merger agreement?', iqs: 0.96, spark: [60, 70, 65, 80, 90, 95], groundedness: 0.94, completeness: 0.91, flag: null, time: '2m ago' },
  { query: 'Summarize Q2 revenue by region', iqs: 0.88, spark: [80, 75, 70, 85, 88, 85], groundedness: 0.91, completeness: 0.78, completenessColor: 'var(--amber)', flag: { label: 'incomplete', pill: 'y' }, time: '5m ago' },
  { query: 'How many employees joined in 2024?', iqs: 0.41, iqsColor: 'var(--red)', spark: [85, 80, 50, 30, 35, 40], sparkDeclining: true, groundedness: 0.32, groundednessColor: 'var(--red)', completeness: 0.61, flag: { label: 'hallucination_risk', pill: 'r' }, time: '8m ago' },
  { query: 'What is our refund policy for enterprise clients?', iqs: 0.94, spark: [90, 92, 88, 93, 94, 94], groundedness: 0.97, completeness: 0.89, flag: null, time: '12m ago' },
  { query: 'Compare cloud infrastructure costs between Q1 and Q2', iqs: 0.92, spark: [88, 90, 91, 89, 92, 92], groundedness: 0.95, completeness: 0.87, flag: null, time: '18m ago' },
]

// ─── All scores (scores page, 6 rows) ───
export interface AllScore {
  query: string
  iqs: number
  iqsColor?: string
  spark: number[]
  sparkDeclining?: boolean
  groundedness: number
  groundednessColor?: string
  completeness: number
  completenessColor?: string
  relevance: number
  consistency: number
  numeric: number
  numericColor?: string
  flag: { label: string; pill: 'r' | 'y' } | null
  time: string
}
export const MOCK_ALL_SCORES: AllScore[] = [
  { query: 'What are the key provisions of the merger agreement?', iqs: 0.96, spark: [60, 70, 90, 95], groundedness: 0.94, completeness: 0.91, relevance: 0.97, consistency: 0.98, numeric: 1.0, numericColor: 'var(--accent)', flag: null, time: '2m' },
  { query: 'Summarize Q2 revenue by region', iqs: 0.88, spark: [75, 85, 88, 85], groundedness: 0.91, completeness: 0.78, completenessColor: 'var(--amber)', relevance: 0.86, consistency: 0.93, numeric: 0.75, numericColor: 'var(--amber)', flag: { label: 'incomplete', pill: 'y' }, time: '5m' },
  { query: 'How many employees joined in 2024?', iqs: 0.41, iqsColor: 'var(--red)', spark: [50, 30, 35, 40], sparkDeclining: true, groundedness: 0.32, groundednessColor: 'var(--red)', completeness: 0.61, relevance: 0.78, consistency: 0.88, numeric: 0.2, numericColor: 'var(--red)', flag: { label: 'hallucination_risk', pill: 'r' }, time: '8m' },
  { query: 'What is our refund policy for enterprise clients?', iqs: 0.94, spark: [93, 94, 94, 94], groundedness: 0.97, completeness: 0.89, relevance: 0.95, consistency: 0.96, numeric: 1.0, numericColor: 'var(--accent)', flag: null, time: '12m' },
  { query: 'Compare cloud infrastructure costs between Q1 and Q2', iqs: 0.92, spark: [91, 89, 92, 92], groundedness: 0.95, completeness: 0.87, relevance: 0.91, consistency: 0.94, numeric: 0.98, numericColor: 'var(--accent)', flag: null, time: '18m' },
  { query: 'Explain the change in deferred revenue recognition', iqs: 0.9, spark: [88, 90, 89, 90], groundedness: 0.92, completeness: 0.85, relevance: 0.89, consistency: 0.95, numeric: 0.92, numericColor: 'var(--accent)', flag: null, time: '25m' },
]

// ─── Evidence record (full evidence page) ───
export interface EvidenceChunkSource {
  source: string
  text: string
  matchTag?: string
}
export interface EvidenceRecord {
  query: string
  compositeIqs: number
  entries: (EvidenceEntry & { tone: 'hi' | 'mid' | 'lo'; numFlag?: string; chunkTag?: string })[]
  summary: { label: string; value: string; color?: string; pill?: { label: string; kind: 'y' | 'g' | 'r' } }[]
  metricBreakdown: { label: string; value: number; color: string; accent?: boolean }[]
  contextSources: EvidenceChunkSource[]
  detail: {
    sentence: number
    note: string
    dims: { label: string; value: number; color: string }[]
    numericClaims: { claim: string; verdict: string; color: string }[]
  }
}

export const MOCK_EVIDENCE_RECORD: EvidenceRecord = {
  query: 'What were the key revenue drivers in Q2 2024 and how did they compare to Q1 performance?',
  compositeIqs: 0.88,
  entries: [
    {
      response_sentence:
        'The primary revenue driver in Q2 2024 was the enterprise segment, which grew 34% year-over-year to reach $142M in quarterly revenue.',
      best_matching_chunk:
        'Enterprise segment revenue reached $142M in Q2, representing 34% YoY growth driven by 12 new logos.',
      entailment_score: 0.96,
      supported: true,
      contradiction_detected: false,
      no_grounding_found: false,
      chunk_source: 'chunk 1 · annual_report.pdf',
      chunk_index: 1,
      mini_iqs: 0.96,
      mini_dims: { G: 0.97, R: 0.94, C: 0.96 },
      tone: 'hi',
      chunkTag: 'matched',
    },
    {
      response_sentence:
        'Self-serve ARR crossed the $50M threshold for the first time, driven primarily by the new usage-based pricing tier launched in March.',
      best_matching_chunk:
        'Self-serve ARR hit $50M milestone following the March launch of usage-based pricing.',
      entailment_score: 0.91,
      supported: true,
      contradiction_detected: false,
      no_grounding_found: false,
      chunk_source: 'chunk 2 · earnings_call.txt',
      chunk_index: 2,
      mini_iqs: 0.91,
      mini_dims: { G: 0.93, R: 0.88, C: 0.91 },
      tone: 'hi',
      chunkTag: 'matched',
    },
    {
      response_sentence:
        'Compared to Q1, overall revenue increased by approximately 23%, with the APAC region contributing the largest incremental gains.',
      best_matching_chunk:
        'Q1-to-Q2 growth was reported at 18% in the board summary. APAC regional data was not broken out.',
      entailment_score: 0.58,
      supported: false,
      contradiction_detected: true,
      no_grounding_found: false,
      chunk_source: 'chunk 3 · quarterly_review.pdf',
      chunk_index: 3,
      mini_iqs: 0.58,
      mini_dims: { G: 0.42, R: 0.81, C: 0.74 },
      tone: 'mid',
      numFlag: '23%',
    },
    {
      response_sentence:
        'The company maintained healthy gross margins of 78%, consistent with prior quarters despite increased infrastructure investment.',
      best_matching_chunk:
        'Gross margin held steady at 78% in Q2, in line with 77.8% reported in Q1.',
      entailment_score: 0.93,
      supported: true,
      contradiction_detected: false,
      no_grounding_found: false,
      chunk_source: 'chunk 1 · annual_report.pdf',
      chunk_index: 1,
      mini_iqs: 0.93,
      mini_dims: { G: 0.95, R: 0.86, C: 0.97 },
      tone: 'hi',
      chunkTag: 'matched',
    },
    {
      response_sentence:
        'Net retention rate remained above 120%, indicating strong expansion revenue within the existing customer base.',
      best_matching_chunk:
        'NRR of 122% in Q2, up from 119% in Q1, driven by seat expansion in mid-market.',
      entailment_score: 0.89,
      supported: true,
      contradiction_detected: false,
      no_grounding_found: false,
      chunk_source: 'chunk 2 · earnings_call.txt',
      chunk_index: 2,
      mini_iqs: 0.89,
      mini_dims: { G: 0.91, R: 0.82, C: 0.93 },
      tone: 'hi',
      chunkTag: 'matched',
    },
  ],
  summary: [
    { label: 'sentences', value: '5' },
    { label: 'grounded', value: '4 / 5', color: 'var(--green)' },
    { label: 'contradictions', value: '1', color: 'var(--red)' },
    { label: 'weakest', value: '#3 (0.58)', color: 'var(--amber)' },
    { label: 'flags', value: '', pill: { label: 'incomplete', kind: 'y' } },
    { label: 'numeric claims', value: '4 found' },
    { label: 'numeric verified', value: '3 / 4', color: 'var(--amber)' },
  ],
  metricBreakdown: [
    { label: 'groundedness', value: 0.84, color: 'var(--green)' },
    { label: 'completeness', value: 0.78, color: 'var(--amber)' },
    { label: 'relevance', value: 0.86, color: 'var(--green)' },
    { label: 'consistency', value: 0.9, color: 'var(--green)' },
    { label: 'numeric grounding', value: 0.75, color: 'var(--accent)', accent: true },
  ],
  contextSources: [
    {
      source: 'chunk 1 · annual_report.pdf',
      text: 'Enterprise segment revenue reached $142M in Q2, representing 34% YoY growth. Gross margin held steady at 78%...',
      matchTag: '2 matches',
    },
    {
      source: 'chunk 2 · earnings_call.txt',
      text: 'Self-serve ARR hit $50M milestone. NRR of 122% in Q2, up from 119% in Q1...',
      matchTag: '2 matches',
    },
    {
      source: 'chunk 3 · quarterly_review.pdf',
      text: 'Q1-to-Q2 growth was reported at 18%. APAC regional data was not broken out...',
    },
  ],
  detail: {
    sentence: 3,
    note: 'Claim "23% revenue increase" contradicts context which states 18% Q1-to-Q2 growth. APAC regional attribution not found in any provided context chunk.',
    dims: [
      { label: 'groundedness', value: 0.42, color: 'var(--red)' },
      { label: 'relevance', value: 0.81, color: 'var(--green)' },
      { label: 'consistency', value: 0.74, color: 'var(--amber)' },
      { label: 'mini IQS', value: 0.58, color: 'var(--amber)' },
    ],
    numericClaims: [
      { claim: '"23%" — Q1-to-Q2 growth rate', verdict: '✗ contradiction (context: 18%)', color: 'var(--red)' },
      { claim: '"APAC" — largest regional contributor', verdict: '? not found in context', color: 'var(--amber)' },
    ],
  },
}

// ─── Inbox records (5 cards) ───
export interface InboxRecord {
  id: string
  iqs: number
  tone: 'hi' | 'mid' | 'lo'
  query: string
  meta: string
  flag: { label: string; pill: 'r' | 'y' } | null
  status: { label: string; kind: 'pending' | 'corrected' | 'passed' }
}
export const MOCK_INBOX_RECORDS: InboxRecord[] = [
  { id: 'rec-0041', iqs: 0.41, tone: 'lo', query: 'How many employees joined in 2024?', meta: 'gpt-4o · support-bot-v2 · 8 min ago', flag: { label: 'hallucination_risk', pill: 'r' }, status: { label: 'pending', kind: 'pending' } },
  { id: 'rec-0067', iqs: 0.67, tone: 'mid', query: 'What caused the drop in NPS scores last quarter?', meta: 'claude-sonnet · analyst-agent · 22 min ago', flag: { label: 'off_topic', pill: 'y' }, status: { label: 'pending', kind: 'pending' } },
  { id: 'rec-0088', iqs: 0.88, tone: 'mid', query: 'Summarize Q2 revenue by region', meta: 'gpt-4o · support-bot-v2 · 5 min ago', flag: { label: 'incomplete', pill: 'y' }, status: { label: 'corrected', kind: 'corrected' } },
  { id: 'rec-0096', iqs: 0.96, tone: 'hi', query: 'What are the key provisions of the merger agreement?', meta: 'claude-opus · legal-review · 2 min ago', flag: null, status: { label: 'passed', kind: 'passed' } },
  { id: 'rec-0038', iqs: 0.38, tone: 'lo', query: 'Explain the warranty terms for Plan B customers', meta: 'gpt-4o-mini · support-bot-v2 · 1h ago', flag: { label: 'ungrounded', pill: 'r' }, status: { label: 'pending', kind: 'pending' } },
]

// ─── Flags table (5 rows) ───
export interface FlagsTableRow {
  query: string
  iqs: number
  iqsColor?: string
  flag: { label: string; pill: 'r' | 'y' }
  trigger: string
  triggerColor: string
  time: string
}
export const MOCK_FLAGS_TABLE: FlagsTableRow[] = [
  { query: 'How many employees joined in 2024?', iqs: 0.41, iqsColor: 'var(--red)', flag: { label: 'hallucination_risk', pill: 'r' }, trigger: 'groundedness: 0.32', triggerColor: 'var(--red)', time: '8m ago' },
  { query: 'Summarize Q2 revenue by region', iqs: 0.88, flag: { label: 'incomplete', pill: 'y' }, trigger: 'completeness: 0.78', triggerColor: 'var(--amber)', time: '5m ago' },
  { query: 'What caused the drop in NPS scores last quarter?', iqs: 0.67, iqsColor: 'var(--amber)', flag: { label: 'off_topic', pill: 'y' }, trigger: 'relevance: 0.41', triggerColor: 'var(--amber)', time: '22m ago' },
  { query: 'Draft a response about data privacy compliance', iqs: 0.79, flag: { label: 'self_contradictory', pill: 'y' }, trigger: 'consistency: 0.52', triggerColor: 'var(--amber)', time: '35m ago' },
  { query: 'Explain the warranty terms for Plan B customers', iqs: 0.38, iqsColor: 'var(--red)', flag: { label: 'ungrounded', pill: 'r' }, trigger: 'groundedness: 0.21', triggerColor: 'var(--red)', time: '1h ago' },
]

// ─── Flag volume chart (7 days × 5 flag types) ───
export const MOCK_FLAG_VOLUME: number[][] = [
  [18, 8, 52, 22, 7],
  [22, 12, 61, 28, 9],
  [15, 6, 48, 18, 11],
  [28, 14, 72, 35, 8],
  [20, 10, 58, 26, 10],
  [12, 5, 38, 15, 6],
  [24, 11, 65, 30, 12],
]
export const FLAG_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
export const FLAG_SERIES = [
  { key: 'hallucination', color: 'var(--red)' },
  { key: 'ungrounded', color: 'var(--red)' },
  { key: 'incomplete', color: 'var(--amber)' },
  { key: 'off_topic', color: 'var(--amber)' },
  { key: 'self_contradictory', color: 'var(--amber)' },
]
export const MOCK_FLAG_STATS = [
  { label: 'total this week', value: '908' },
  { label: 'most common', value: 'incomplete (45%)', color: 'var(--amber)' },
  { label: 'most severe', value: 'hallucination (16%)', color: 'var(--red)' },
  { label: 'trending up', value: 'off_topic +22%', color: 'var(--red)' },
  { label: 'trending down', value: 'self_contradictory -8%', color: 'var(--green)' },
]

// ─── Calibration history (3 rows) ───
export interface CalHistoryRow {
  date: string
  method: { label: string; pill?: 'g' | 'y'; muted?: boolean }
  samples: { label: string; color?: string; muted?: boolean }
  threshold: string
  status: { label: string; pill?: 'b'; muted?: boolean }
}
export const MOCK_CALIBRATION_HISTORY: CalHistoryRow[] = [
  { date: 'Jun 16', method: { label: 'labeled', pill: 'g' }, samples: { label: '248' }, threshold: '0.52', status: { label: 'active', pill: 'b' } },
  { date: 'Jun 10', method: { label: 'bootstrap', pill: 'y' }, samples: { label: '0', color: 'var(--text-3)' }, threshold: '0.48', status: { label: 'superseded', muted: true } },
  { date: 'Jun 3', method: { label: 'default', muted: true }, samples: { label: '—', muted: true }, threshold: '0.50', status: { label: 'initial', muted: true } },
]

// ─── Heatmap (7 rows × 24 cols of quality 0–1) ───
function buildHeatmap(): number[][] {
  // Deterministic pseudo-random so render is stable; mirrors mockup distribution buckets.
  const grid: number[][] = []
  let seed = 1337
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
  for (let row = 0; row < 7; row++) {
    const cols: number[] = []
    for (let col = 0; col < 24; col++) {
      const r = rand()
      // bucket into ranges matching mockup: <0.04 low, <0.12 mid, <0.42 good, else excellent
      let v: number
      if (r < 0.04) v = 0.1 + rand() * 0.15
      else if (r < 0.12) v = 0.35 + rand() * 0.12
      else if (r < 0.42) v = 0.55 + rand() * 0.2
      else v = 0.82 + rand() * 0.18
      cols.push(parseFloat(v.toFixed(2)))
    }
    grid.push(cols)
  }
  return grid
}
export const MOCK_HEATMAP: number[][] = buildHeatmap()
