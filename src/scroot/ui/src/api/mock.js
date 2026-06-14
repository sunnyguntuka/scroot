// Demo mode mock data - activated via ?demo in the URL
// All functions return promises that resolve after a short simulated delay.

const delay = (ms = 120) => new Promise(r => setTimeout(r, ms));

// Mirrors src/scroot/result.py _METRIC_FLAG_EXPLANATIONS, for demo-mode parity.
const METRIC_FLAG_EXPLANATIONS = {
  groundedness: 'The response makes claims that are not supported by the provided context.',
  completeness: 'The response does not address all parts of the query.',
  relevance: 'The response drifts from the topic of the query.',
  consistency: 'The response contradicts itself.',
  confidence: 'The response expresses inappropriate certainty relative to the evidence.',
};

function metricExplanationsFor(flags = []) {
  const explanations = {};
  for (const flag of flags) {
    const metric = flag.replace(/^low_/, '');
    if (METRIC_FLAG_EXPLANATIONS[metric]) explanations[metric] = METRIC_FLAG_EXPLANATIONS[metric];
  }
  return explanations;
}

function weakestMetric(metrics = {}) {
  const entries = Object.entries(metrics);
  if (!entries.length) return null;
  return entries.reduce((min, cur) => (cur[1] < min[1] ? cur : min))[0];
}

function scoreVariance(metrics = {}) {
  const values = Object.values(metrics);
  if (!values.length) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.round(Math.sqrt(variance) * 10000) / 10000;
}

function iqsExplanationFor(iqs, metrics, threshold = 0.70) {
  if (iqs >= threshold) return `IQS ${iqs.toFixed(2)} - all metrics above threshold.`;
  const weakest = weakestMetric(metrics);
  const weakestScore = metrics[weakest];
  return `IQS ${iqs.toFixed(2)} - primary driver: ${weakest} (${weakestScore.toFixed(2)}). ${METRIC_FLAG_EXPLANATIONS[weakest] || ''}`;
}

// Sentence-level evidence attribution (EntailmentResult.evidence_map.to_dict()
// shape) for a couple of records, to exercise the Evidence Map panel.
const EVIDENCE_MAPS = {
  rec_001: {
    supported: 2,
    unsupported: 1,
    contradictions: 0,
    coverage_ratio: 0.667,
    weakest_sentence: 'It lowers blood sugar.',
    entries: [
      {
        response_sentence: 'Metformin can cause stomach upset.',
        best_matching_chunk: 'Long-term use is associated with vitamin B12 deficiency in approximately 10-30% of patients, lactic acidosis (rare), and gastrointestinal effects including nausea and diarrhea, particularly at initiation.',
        entailment_score: 0.81,
        supported: true,
        contradiction_detected: false,
        no_grounding_found: false,
        chunk_source: 'retrieval',
        chunk_index: 0,
      },
      {
        response_sentence: 'Some people feel nauseous.',
        best_matching_chunk: 'Long-term use is associated with vitamin B12 deficiency in approximately 10-30% of patients, lactic acidosis (rare), and gastrointestinal effects including nausea and diarrhea, particularly at initiation.',
        entailment_score: 0.77,
        supported: true,
        contradiction_detected: false,
        no_grounding_found: false,
        chunk_source: 'retrieval',
        chunk_index: 0,
      },
      {
        response_sentence: 'It lowers blood sugar.',
        best_matching_chunk: 'Metformin is a biguanide antihyperglycemic agent used in the management of type 2 diabetes.',
        entailment_score: 0.22,
        supported: false,
        contradiction_detected: false,
        no_grounding_found: true,
        chunk_source: 'retrieval',
        chunk_index: 0,
      },
    ],
  },
  rec_004: {
    supported: 0,
    unsupported: 0,
    contradictions: 1,
    coverage_ratio: 0,
    weakest_sentence: 'Aspirin prevents heart attacks by thinning the blood.',
    entries: [
      {
        response_sentence: 'Aspirin prevents heart attacks by thinning the blood.',
        best_matching_chunk: 'Aspirin irreversibly inhibits cyclooxygenase-1 (COX-1) and COX-2 enzymes, blocking thromboxane A2 synthesis in platelets.',
        entailment_score: 0.18,
        supported: false,
        contradiction_detected: true,
        no_grounding_found: false,
        chunk_source: 'retrieval',
        chunk_index: 0,
      },
    ],
  },
};

// ─── Shared fake records ──────────────────────────────────────────────────────

const RECORDS = [
  {
    id: 'rec_001',
    query: 'What are the side effects of metformin when taken long-term?',
    context: 'Metformin is a biguanide antihyperglycemic agent used in the management of type 2 diabetes. Long-term use is associated with vitamin B12 deficiency in approximately 10-30% of patients, lactic acidosis (rare), and gastrointestinal effects including nausea and diarrhea, particularly at initiation.',
    response: 'Metformin can cause stomach upset. Some people feel nauseous. It lowers blood sugar.',
    corrected_response: 'Long-term metformin use is associated with vitamin B12 deficiency (affecting 10–30% of patients), gastrointestinal effects such as nausea and diarrhea (especially at initiation), and rarely lactic acidosis. Regular B12 monitoring is recommended for long-term users.',
    model: 'gpt-4o-mini',
    agent: 'clinical-qa-v2',
    status: 'pending_review',
    flags: ['low_groundedness', 'low_completeness'],
    iqs: 0.41,
    metrics: { groundedness: 0.32, completeness: 0.28, relevance: 0.72, consistency: 0.65, confidence: 0.58 },
    created_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_002',
    query: 'Explain the mechanism of action of SSRIs in treating depression.',
    context: 'Selective serotonin reuptake inhibitors (SSRIs) work by blocking the reuptake of serotonin into the presynaptic neuron, increasing serotonin availability in the synaptic cleft. This enhanced serotonergic transmission is thought to improve mood, though the exact therapeutic mechanism involves downstream neuroplasticity changes that occur over weeks.',
    response: 'SSRIs block serotonin reuptake, making more serotonin available in the synapse. This improves mood over several weeks as neuroplasticity changes occur in response to enhanced serotonergic signaling.',
    corrected_response: null,
    model: 'claude-haiku-4-5-20251001',
    agent: 'clinical-qa-v2',
    status: 'pending_review',
    flags: ['low_confidence'],
    iqs: 0.61,
    metrics: { groundedness: 0.74, completeness: 0.68, relevance: 0.81, consistency: 0.70, confidence: 0.42 },
    created_at: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_003',
    query: 'What is the recommended first-line treatment for community-acquired pneumonia in outpatients?',
    context: 'For outpatient treatment of community-acquired pneumonia (CAP) in adults without comorbidities, amoxicillin 1g three times daily is the preferred first-line agent per NICE and ATS/IDSA guidelines. In patients with comorbidities or where atypical pathogens are suspected, a respiratory fluoroquinolone or beta-lactam plus macrolide combination is recommended.',
    response: 'The first-line treatment for CAP in outpatients without comorbidities is amoxicillin 1g three times daily. For patients with comorbidities or suspected atypical pathogens, a respiratory fluoroquinolone or a beta-lactam combined with a macrolide is recommended.',
    corrected_response: null,
    model: 'gpt-4o',
    agent: 'clinical-qa-v2',
    status: 'reviewed',
    flags: [],
    iqs: 0.88,
    metrics: { groundedness: 0.91, completeness: 0.87, relevance: 0.93, consistency: 0.85, confidence: 0.84 },
    created_at: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_004',
    query: 'How does aspirin prevent cardiovascular events?',
    context: 'Aspirin irreversibly inhibits cyclooxygenase-1 (COX-1) and COX-2 enzymes, blocking thromboxane A2 synthesis in platelets. This reduces platelet aggregation for the platelet lifespan (7-10 days). At low doses (75-100mg), the antiplatelet effect predominates, reducing the risk of arterial thrombosis and secondary cardiovascular events.',
    response: 'Aspirin prevents heart attacks by thinning the blood.',
    corrected_response: 'Aspirin irreversibly inhibits COX-1 and COX-2 enzymes, blocking thromboxane A2 synthesis and thereby reducing platelet aggregation. At low doses (75–100 mg), this antiplatelet effect persists for the platelet lifespan (7–10 days), reducing the risk of arterial thrombosis and secondary cardiovascular events.',
    model: 'llama3.2',
    agent: 'general-qa-v1',
    status: 'pending_review',
    flags: ['low_groundedness', 'low_completeness', 'low_consistency'],
    iqs: 0.29,
    metrics: { groundedness: 0.18, completeness: 0.21, relevance: 0.68, consistency: 0.22, confidence: 0.45 },
    created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_005',
    query: 'What are the contraindications to thrombolytic therapy in acute ischemic stroke?',
    context: 'Absolute contraindications to IV tPA in acute ischemic stroke include: prior intracranial hemorrhage, significant head trauma or stroke in the past 3 months, symptoms suggestive of subarachnoid hemorrhage, active internal bleeding, intracranial neoplasm, and current anticoagulant use with INR >1.7. Time window is 4.5 hours from symptom onset.',
    response: 'Contraindications to thrombolytics in stroke include recent head trauma, prior brain bleed, active bleeding, anticoagulant use with elevated INR (>1.7), intracranial tumor, and subarachnoid hemorrhage. The treatment window is 4.5 hours from symptom onset.',
    corrected_response: null,
    model: 'gpt-4o',
    agent: 'clinical-qa-v2',
    status: 'reviewed',
    flags: [],
    iqs: 0.82,
    metrics: { groundedness: 0.85, completeness: 0.81, relevance: 0.90, consistency: 0.82, confidence: 0.77 },
    created_at: new Date(Date.now() - 36 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_006',
    query: 'What is the Glasgow Coma Scale and how is it scored?',
    context: 'The Glasgow Coma Scale (GCS) assesses consciousness across three domains: eye opening (1-4), verbal response (1-5), and motor response (1-6), giving a total score of 3-15. A score of 8 or below generally indicates severe neurological impairment.',
    response: 'The GCS has three components: eyes (1–4), verbal (1–5), motor (1–6), for a total of 3–15. A score ≤8 indicates severe impairment.',
    corrected_response: null,
    model: 'claude-haiku-4-5-20251001',
    agent: 'general-qa-v1',
    status: 'reviewed',
    flags: [],
    iqs: 0.79,
    metrics: { groundedness: 0.82, completeness: 0.76, relevance: 0.88, consistency: 0.80, confidence: 0.73 },
    created_at: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_007',
    query: 'Describe the pathophysiology of septic shock.',
    context: 'Septic shock results from a dysregulated host response to infection leading to life-threatening organ dysfunction. The inflammatory cascade causes vasodilation, increased vascular permeability, myocardial depression, and microvascular dysfunction. This produces distributive shock with inappropriately low systemic vascular resistance despite elevated cardiac output.',
    response: 'Septic shock happens when infection causes inflammation. The body releases cytokines and blood pressure drops.',
    corrected_response: 'Septic shock is caused by a dysregulated host response to infection that triggers widespread vasodilation, increased vascular permeability, and myocardial depression through inflammatory mediators. This results in distributive shock - characterized by low systemic vascular resistance despite elevated cardiac output - leading to tissue hypoperfusion and organ dysfunction.',
    model: 'llama3.2',
    agent: 'clinical-qa-v2',
    status: 'pending_review',
    flags: ['low_groundedness', 'low_completeness'],
    iqs: 0.38,
    metrics: { groundedness: 0.27, completeness: 0.31, relevance: 0.71, consistency: 0.55, confidence: 0.49 },
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_008',
    query: 'What is the mechanism behind ACE inhibitor-induced cough?',
    context: 'ACE inhibitors block the conversion of angiotensin I to angiotensin II and also prevent the breakdown of bradykinin. Bradykinin accumulates in the lungs, stimulating sensory C-fibers and producing a dry, persistent cough in 5-20% of patients, more common in women and people of East Asian descent.',
    response: 'ACE inhibitor cough is caused by bradykinin accumulation in the lungs due to reduced breakdown by ACE. This stimulates pulmonary sensory C-fibers, causing a dry persistent cough. The incidence is 5–20%, with higher rates in women and East Asian patients.',
    corrected_response: null,
    model: 'gpt-4o-mini',
    agent: 'clinical-qa-v2',
    status: 'reviewed',
    flags: [],
    iqs: 0.85,
    metrics: { groundedness: 0.88, completeness: 0.84, relevance: 0.91, consistency: 0.83, confidence: 0.79 },
    created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_009',
    query: 'How do beta blockers work in heart failure?',
    context: 'In heart failure with reduced ejection fraction (HFrEF), chronic sympathetic activation is initially compensatory but ultimately harmful, causing myocardial remodeling, arrhythmias, and receptor downregulation. Beta blockers counteract this by reducing heart rate, decreasing oxygen demand, preventing arrhythmias, and over time reversing maladaptive remodeling - improving ejection fraction and survival.',
    response: 'Beta blockers are used in heart failure. They slow the heart rate and protect the heart.',
    corrected_response: 'In HFrEF, chronic sympathetic activation causes harmful myocardial remodeling and receptor downregulation. Beta blockers counteract this by reducing heart rate and oxygen demand, preventing arrhythmias, and over time reversing maladaptive remodeling - resulting in improved ejection fraction and reduced mortality.',
    model: 'llama3.2',
    agent: 'general-qa-v1',
    status: 'pending_review',
    flags: ['low_groundedness', 'low_completeness'],
    iqs: 0.44,
    metrics: { groundedness: 0.35, completeness: 0.38, relevance: 0.75, consistency: 0.62, confidence: 0.51 },
    created_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'rec_010',
    query: 'What is herd immunity and how is the threshold calculated?',
    context: 'Herd immunity occurs when enough of a population is immune - through vaccination or prior infection - to make disease spread unlikely. The herd immunity threshold (HIT) is calculated as 1 − 1/R0, where R0 is the basic reproduction number. For measles (R0 ≈ 15), the HIT is approximately 93%. For COVID-19 original strain (R0 ≈ 2.5), HIT was approximately 60%.',
    response: 'Herd immunity is when enough people in a population are immune that the disease stops spreading easily. It can be calculated.',
    corrected_response: 'Herd immunity occurs when sufficient population immunity (via vaccination or prior infection) reduces disease spread. The herd immunity threshold (HIT) is calculated as 1 − 1/R₀. For measles (R₀ ≈ 15), HIT ≈ 93%; for the original COVID-19 strain (R₀ ≈ 2.5), HIT ≈ 60%.',
    model: 'gpt-4o-mini',
    agent: 'general-qa-v1',
    status: 'pending_review',
    flags: ['low_completeness', 'low_groundedness'],
    iqs: 0.47,
    metrics: { groundedness: 0.40, completeness: 0.35, relevance: 0.78, consistency: 0.66, confidence: 0.57 },
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

// ─── Analytics trend (30 days) ──────────────────────────────────────────────

function trendData() {
  const points = [];
  let iqs = 0.58;
  for (let i = 29; i >= 0; i--) {
    const d = new Date(Date.now() - i * 24 * 60 * 60 * 1000);
    iqs = Math.min(0.95, Math.max(0.30, iqs + (Math.random() - 0.45) * 0.04));
    points.push({ date: d.toISOString().slice(0, 10), avg_iqs: +iqs.toFixed(3) });
  }
  return points;
}

// ─── Pipeline results ────────────────────────────────────────────────────────

const PIPELINE_RESULTS = RECORDS.slice(0, 8).map((r, i) => ({
  record_id: r.id,
  query_preview: r.query.slice(0, 72) + (r.query.length > 72 ? '…' : ''),
  iqs_before: r.iqs,
  iqs_after: i < 3 ? +(r.iqs + 0.28).toFixed(2) : i < 5 ? +(r.iqs + 0.14).toFixed(2) : null,
  delta: i < 3 ? 0.28 : i < 5 ? 0.14 : null,
  outcome: i < 3 ? 'committed' : i < 5 ? 'review_queue' : i < 7 ? 'draft_ready' : 'skipped',
}));

// ─── Mock API ────────────────────────────────────────────────────────────────

export const mockApi = {
  getQueue: async ({ status, search, sort } = {}) => {
    await delay();
    let records = RECORDS.map(r => ({
      id: r.id,
      query: r.query,
      model: r.model,
      agent_id: r.agent,
      status: r.status,
      flags: r.flags,
      iqs: r.iqs,
      created_at: r.created_at,
    }));
    if (status && status !== 'all') {
      if (['pass', 'warn', 'fail'].includes(status)) {
        const threshold = 0.70;
        records = records.filter(r => {
          if (status === 'pass') return r.iqs >= threshold;
          if (status === 'warn') return r.iqs >= threshold * 0.7 && r.iqs < threshold;
          return r.iqs < threshold * 0.7;
        });
      } else {
        records = records.filter(r => r.status === status);
      }
    }
    if (search) records = records.filter(r => r.query.toLowerCase().includes(search.toLowerCase()));
    return { records, total: records.length, page: 1 };
  },

  getRecord: async (id) => {
    await delay(80);
    const r = RECORDS.find(r => r.id === id) || RECORDS[0];
    return {
      ...r,
      agent_id: r.agent,
      context: r.context,
      correction: r.corrected_response,
      rejection_reason: null,
      corrected_by: null,
      weakest_metric: weakestMetric(r.metrics),
      score_variance: scoreVariance(r.metrics),
      iqs_explanation: iqsExplanationFor(r.iqs, r.metrics),
      metric_explanations: metricExplanationsFor(r.flags),
      guardrail_applied_count: r.corrected_response ? 3 : 0,
      evidence_map: EVIDENCE_MAPS[r.id] || null,
    };
  },

  submitCorrection: async (id, correction) => {
    await delay(300);
    return { success: true, new_iqs: 0.84, delta: 0.31 };
  },

  rejectRecord: async (id, reason) => {
    await delay(200);
    return { success: true };
  },

  generateCorrection: async (id) => {
    await delay(1200);
    const r = RECORDS.find(r => r.id === id) || RECORDS[0];
    return { draft: r.corrected_response || 'Generated correction would appear here.' };
  },

  deleteCorrection: async (id) => {
    await delay(150);
    return { success: true };
  },

  getAnalytics: async (period = 'all') => {
    await delay(200);
    return {
      total_scored: 847,
      avg_iqs: 0.71,
      avg_iqs_delta: 0.04,
      pending_review: 23,
      pass_count: 541,
      warn_count: 198,
      fail_count: 108,
      iqs_trend: trendData(),
      flag_frequency: {
        groundedness: 94,
        completeness: 71,
        confidence:   58,
        relevance:    32,
        consistency:  19,
      },
      iqs_distribution: [
        { bucket: '0.0–0.2', count: 18 },
        { bucket: '0.2–0.4', count: 90 },
        { bucket: '0.4–0.6', count: 185 },
        { bucket: '0.6–0.8', count: 312 },
        { bucket: '0.8–1.0', count: 242 },
      ],
      per_agent: [
        { agent_id: 'llama3.2',       avg_iqs: 0.58, count: 213 },
        { agent_id: 'general-qa-v1',  avg_iqs: 0.67, count: 298 },
        { agent_id: 'clinical-qa-v2', avg_iqs: 0.79, count: 336 },
      ],
    };
  },

  getExportPreview: async (params = {}) => {
    await delay(100);
    return {
      count: 41,
      corrected_count: 38,
      size_bytes: 187_400,
      agents: ['llama3.2', 'general-qa-v1', 'clinical-qa-v2'],
    };
  },

  downloadExport: async () => {
    await delay(300);
    throw new Error('Download not available in demo mode');
  },

  getSettings: async () => {
    await delay(80);
    return {
      iqs_threshold: 0.70,
      metric_weights: { groundedness: 0.35, completeness: 0.25, relevance: 0.20, consistency: 0.15, confidence: 0.05 },
      corrector: {
        mode: 'disabled',
        local: { model_id: 'phi4-mini', model_name: 'Phi-4-mini-instruct', model_downloaded: false, model_size_gb: 2.4, model_path: null },
        api: { api_key_set: false, api_key_prefix: '', base_url: '', model: 'gpt-4o-mini' },
      },
      llm_corrector: { provider: 'none', api_key: '', base_url: '', model: '' },
      store_path: '~/.scroot/feedback.jsonl',
      record_count: 847,
      store_size: '2.3 MB',
    };
  },

  updateSettings: async (patch) => {
    await delay(150);
    return { success: true };
  },

  testConnection: async () => {
    await delay(800);
    return { ok: true, latency_ms: 214 };
  },

  startPipelineRun: async (config) => {
    await delay(400);
    return {
      run_id: 'run_demo_001',
      status: 'completed',
      mode: config.mode,
      total_records: 8,
      processed_count: 8,
      committed_count: 3,
      review_queue_count: 2,
      skipped_count: 2,
      failed_count: 1,
      log: [
        '[00:00] Starting pipeline - 8 records, mode: auto_commit',
        '[00:03] rec_001 → NLI: 0.41 → 0.69  Δ+0.28 ✓ committed',
        '[00:06] rec_004 → NLI: 0.29 → 0.57  Δ+0.28 ✓ committed',
        '[00:09] rec_007 → NLI: 0.38 → 0.66  Δ+0.28 ✓ committed',
        '[00:12] rec_002 → NLI: 0.61 → 0.72  Δ+0.11 ↷ below threshold → review queue',
        '[00:15] rec_009 → NLI: 0.44 → 0.55  Δ+0.11 ↷ below threshold → review queue',
        '[00:18] rec_003 → already above threshold, skipped',
        '[00:21] rec_005 → already above threshold, skipped',
        '[00:24] rec_010 → LLM call failed: rate limit ✗ failed',
        '[00:24] Pipeline complete - 3 committed, 2 queued, 2 skipped, 1 failed',
      ],
      results: PIPELINE_RESULTS,
      summary: { avg_delta: 0.19, committed_rate: 0.375 },
    };
  },

  getPipelineStatus: async (runId) => {
    await delay(100);
    return { run_id: runId, status: 'completed' };
  },

  pausePipelineRun:  async () => { await delay(100); return { status: 'paused' }; },
  resumePipelineRun: async () => { await delay(100); return { status: 'running' }; },
  cancelPipelineRun: async () => { await delay(100); return { status: 'cancelled' }; },

  getHealth: async () => {
    await delay(50);
    return { status: 'ok', version: '0.1.2', pending_count: 23, avg_iqs_today: 0.71 };
  },

  getGuardrailStats: async () => {
    await delay(50);
    const records = RECORDS.filter(r => r.corrected_response);
    return {
      active_guardrails: records.length,
      total_applications: records.length * 3,
      records: records.map(r => ({ id: r.id, guardrail_applied_count: 3 })),
    };
  },

  getCorrectorModels: async () => {
    await delay(80);
    return {
      models: [
        {
          id: 'phi4-mini', name: 'Phi-4-mini-instruct', is_default: true,
          description: "Microsoft's efficient 3.8B model. Best instruction following in its class. Default choice.",
          size_gb: 2.4, min_ram_gb: 4, rec_ram_gb: 8, context_window: 128000,
          license: 'MIT', downloaded: false, path: null,
        },
        {
          id: 'smollm3', name: 'SmolLM3', is_default: false,
          description: "HuggingFace's efficient 3B model. Faster on CPU, slightly lower quality than Phi-4-mini.",
          size_gb: 1.8, min_ram_gb: 4, rec_ram_gb: 8, context_window: 128000,
          license: 'Apache 2.0', downloaded: false, path: null,
        },
      ],
    };
  },

  downloadModel: async (modelId) => {
    await delay(200);
    return { model_id: modelId, status: 'downloading' };
  },

  getModelDownloadStatus: async (modelId) => {
    await delay(50);
    return {
      model_id: modelId, status: 'downloading',
      progress_bytes: 600_000_000, total_bytes: 2_400_000_000,
      progress_pct: 25, eta_seconds: 90, error: null,
    };
  },

  deleteModel: async (modelId) => {
    await delay(150);
    return { model_id: modelId, deleted: true, freed_gb: 2.4 };
  },

  testCorrector: async () => {
    await delay(500);
    return {
      mode: 'disabled', model: null, latency_ms: 0,
      sample_output: null, tokens_generated: 0,
      tok_per_sec: null, error: 'Corrector is disabled',
    };
  },
};
