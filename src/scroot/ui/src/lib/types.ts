export interface EntailmentResult {
  groundedness: number | null
  completeness: number
  relevance: number
  consistency: number
  confidence: number
  iqs: number
  flags: string[]
  details: Record<string, unknown>
  evidence_map: EvidenceMap | null
  effective_weights: Record<string, number> | null
  context_used: boolean | null
  iqs_metric_count: number | null
}

export interface EvidenceMap {
  entries: EvidenceEntry[]
  supported_count: number
  unsupported_count: number
  contradiction_count: number
  coverage_ratio: number
  weakest_sentence: string | null
}

export interface EvidenceEntry {
  response_sentence: string
  best_matching_chunk: string | null
  entailment_score: number | null
  supported: boolean
  contradiction_detected: boolean
  no_grounding_found: boolean
  chunk_source: string | null
  chunk_index: number | null
  mini_iqs?: number | null
  mini_dims?: Record<string, number> | null
}

export interface ScoreRecord {
  id: string
  query: string
  response: string
  context: string[] | null
  result: EntailmentResult
  model: string | null
  agent_id: string | null
  status: 'pending' | 'reviewed' | 'corrected' | 'passed'
  created_at: string
}

export type Theme = 'dark' | 'light'
