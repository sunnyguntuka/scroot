import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Play, Pause, StopCircle, RotateCcw, Inbox, Download,
  ShieldCheck, AlertCircle, Info, Wand2, CheckCircle2,
} from 'lucide-react';
import { TopBar } from '../components/layout/TopBar';
import { Button } from '../components/ui/Button';
import { IQSBadge } from '../components/ui/IQSBadge';
import { useToast } from '../components/ui/Toast';
import { api } from '../api/client';
import { usePipelineStore } from '../stores/pipelineStore';
import { fmtIQS } from '../utils/iqs';
import { fmtDuration as fmtDur } from '../utils/time';

// ─── Section label ─────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return <p className="section-label mb-3">{children}</p>;
}

// ─── Mode card ─────────────────────────────────────────────────────────
function ModeCard({ value, selected, onSelect, title, description, chip, chipColor, disabled }) {
  return (
    <div
      onClick={!disabled ? onSelect : undefined}
      className={`card p-4 cursor-pointer transition-all duration-150
        ${disabled ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}
        ${selected && !disabled ? 'border-indigo-400 bg-indigo-50 ring-1 ring-indigo-300' : 'hover:border-indigo-200'}
      `}
    >
      <div className="flex items-start gap-3">
        <input
          type="radio" name="pipeline-mode" value={value}
          checked={selected} disabled={disabled}
          onChange={!disabled ? onSelect : undefined}
          className="accent-indigo-600 mt-0.5 shrink-0"
        />
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-indigo-950">{title}</span>
            {chip && (
              <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${chipColor}`}>
                {chip}
              </span>
            )}
          </div>
          <p className="text-xs text-indigo-500 leading-relaxed">{description}</p>
        </div>
      </div>
    </div>
  );
}

// ─── Progress bar ─────────────────────────────────────────────────────
function ProgressBar({ value, total }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div className="w-full">
      <div className="h-2 bg-indigo-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-600 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between mt-1 text-[11px] font-mono-score text-indigo-400">
        <span>{value} / {total}</span>
        <span>{pct}%</span>
      </div>
    </div>
  );
}

// ─── Log panel ─────────────────────────────────────────────────────────
function LogPanel({ lines = [] }) {
  const ref = useRef(null);
  const userScrolled = useRef(false);

  useEffect(() => {
    if (!userScrolled.current && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [lines]);

  const lineColor = (line) => {
    if (line.includes('✓') || line.includes('committed')) return 'text-green-700';
    if (line.includes('↷') || line.includes('review')) return 'text-amber-700';
    if (line.includes('✗') || line.includes('failed')) return 'text-red-600';
    return 'text-indigo-300';
  };

  return (
    <div
      ref={ref}
      onScroll={() => {
        if (!ref.current) return;
        const { scrollTop, scrollHeight, clientHeight } = ref.current;
        userScrolled.current = scrollTop + clientHeight < scrollHeight - 10;
      }}
      className="bg-white border border-indigo-100 rounded-card font-mono-score text-[11px] leading-relaxed
                 overflow-y-auto p-4 space-y-0.5"
      style={{ maxHeight: 220 }}
    >
      {lines.length === 0
        ? <span className="text-indigo-200">Waiting for log output…</span>
        : lines.map((line, i) => (
            <div key={i} className={lineColor(line)}>{line}</div>
          ))
      }
    </div>
  );
}

// ─── Outcome chip ─────────────────────────────────────────────────────
const OUTCOME_STYLE = {
  committed:    'bg-green-50 text-green-700 border-green-200',
  review_queue: 'bg-amber-50 text-amber-700 border-amber-200',
  skipped:      'bg-gray-50 text-gray-500 border-gray-200',
  draft_ready:  'bg-indigo-50 text-indigo-600 border-indigo-200',
  failed:       'bg-red-50 text-red-600 border-red-200',
};
const OUTCOME_DOT = {
  committed:    'bg-green-500',
  review_queue: 'bg-amber-500',
  skipped:      'bg-gray-400',
  draft_ready:  'bg-indigo-500',
  failed:       'bg-red-500',
};
const OUTCOME_LABEL = {
  committed:    'committed',
  review_queue: 'review queue',
  skipped:      'skipped',
  draft_ready:  'draft ready',
  failed:       'failed',
};

// ─── Main page ─────────────────────────────────────────────────────────
// Per-record latency estimates (seconds) by provider type
const PROVIDER_LATENCY = {
  'local_phi4-mini': 15,
  'local_smollm3': 10,
  api_fast: 2,
  api_standard: 4,
};

function getProviderLabel(corrector) {
  if (!corrector || corrector.mode === 'disabled') return null;
  if (corrector.mode === 'local') {
    const name = corrector.local?.model_name || 'Local LLM';
    return `Local (${name})`;
  }
  return `API (${corrector.api?.model || 'gpt-4o-mini'})`;
}

function getPerRecordSecs(corrector) {
  if (!corrector || corrector.mode === 'disabled') return 3;
  if (corrector.mode === 'local') {
    const modelId = corrector.local?.model_id || 'phi4-mini';
    return PROVIDER_LATENCY[`local_${modelId}`] || 15;
  }
  const m = corrector.api?.model || '';
  if (m.includes('mini') || m.includes('haiku') || m.includes('flash')) return PROVIDER_LATENCY.api_fast;
  return PROVIDER_LATENCY.api_standard;
}

export function Pipeline({ pendingCount = 0, avgIqs, threshold = 0.70, hasCorrector = false, corrector = null }) {
  const navigate = useNavigate();
  const addToast = useToast();
  const { run, config, setRun, setConfig, resetRun } = usePipelineStore();

  const [cancelConfirm, setCancelConfirm] = useState(false);
  const pollRef = useRef(null);

  // Determine which sub-view to show
  const view =
    !run || run.status === null                    ? 'configure'  :
    run.status === 'running' || run.status === 'paused' ? 'running'    :
    'results';

  // Polling for running/paused states
  useEffect(() => {
    if (view !== 'running') { clearInterval(pollRef.current); return; }
    pollRef.current = setInterval(async () => {
      try {
        const updated = await api.getPipelineStatus(run.run_id);
        setRun(updated);
        if (['completed','cancelled','failed'].includes(updated.status)) {
          clearInterval(pollRef.current);
        }
      } catch {}
    }, 1500);
    return () => clearInterval(pollRef.current);
  }, [view, run?.run_id]);

  // ─── Configure view ─────────────────────────────────────────────────
  const handleStartRun = async () => {
    if (!hasCorrector && config.mode !== 'drafts_only') {
      addToast('Configure an LLM corrector in Settings first.', 'error');
      return;
    }
    try {
      const newRun = await api.startPipelineRun(config);
      setRun(newRun);
      addToast('Pipeline started', 'info');
    } catch (e) {
      if (e.code === 'ENTERPRISE_ONLY') {
        addToast('Fully autonomous mode requires Scroot Cloud.', 'error');
      } else {
        addToast(e.message || 'Failed to start pipeline', 'error');
      }
    }
  };

  const recordsToProcess =
    config.record_filter.include_all ? (pendingCount + 50) :
    config.record_filter.include_fail_only ? Math.floor(pendingCount * 0.3) :
    pendingCount;

  const perRecordSecs = getPerRecordSecs(corrector);
  const estDuration = recordsToProcess * perRecordSecs;
  const providerLabel = getProviderLabel(corrector);

  // ─── Running view helpers ───────────────────────────────────────────
  const handlePause = async () => {
    try {
      const updated = await api.pausePipelineRun(run.run_id);
      setRun(updated);
    } catch { addToast('Failed to pause', 'error'); }
  };

  const handleResume = async () => {
    try {
      const updated = await api.resumePipelineRun(run.run_id);
      setRun(updated);
    } catch { addToast('Failed to resume', 'error'); }
  };

  const handleCancel = async () => {
    try {
      const updated = await api.cancelPipelineRun(run.run_id);
      setRun(updated);
      setCancelConfirm(false);
    } catch { addToast('Failed to cancel', 'error'); }
  };

  // Sort results: committed → review_queue → draft_ready → skipped → failed
  const ORDER = ['committed','draft_ready','review_queue','skipped','failed'];
  const sortedResults = run?.results
    ? [...run.results].sort((a, b) =>
        ORDER.indexOf(a.outcome) - ORDER.indexOf(b.outcome) ||
        (b.delta ?? -Infinity) - (a.delta ?? -Infinity)
      )
    : [];

  return (
    <div className="flex flex-col h-full page-enter">
      <TopBar title="Pipeline" avgIqs={avgIqs} threshold={threshold} />

      <div className="flex-1 overflow-y-auto p-6">

        {/* ─── CONFIGURE ─────────────────────────────────────────────── */}
        {view === 'configure' && (
          <div className="grid grid-cols-[2fr_3fr] gap-6 max-w-5xl">

            {/* Left */}
            <div className="space-y-4">
              {/* Run mode */}
              <div className="card p-5">
                <SectionLabel>Run mode</SectionLabel>
                <div className="space-y-3">
                  <ModeCard
                    value="drafts_only"
                    selected={config.mode === 'drafts_only'}
                    onSelect={() => setConfig({ mode: 'drafts_only' })}
                    title="Generate drafts only"
                    description="LLM fills every record's correction field. You review each one before committing."
                    chip="human reviews"
                    chipColor="bg-green-100 text-green-700"
                  />
                  <ModeCard
                    value="auto_commit"
                    selected={config.mode === 'auto_commit'}
                    onSelect={() => setConfig({ mode: 'auto_commit' })}
                    title="Auto-commit if NLI passes"
                    description="Corrections auto-committed when NLI improvement ≥ threshold. Failures stay in queue for review."
                    chip="NLI gated"
                    chipColor="bg-indigo-50 text-indigo-500 border border-indigo-200"
                  />
                  <ModeCard
                    value="fully_autonomous"
                    selected={false}
                    disabled
                    title="Fully autonomous"
                    description="All corrections auto-committed after NLI re-scoring. Requires audit trail and approval policy. Available in Scroot Cloud with RBAC and compliance export."
                    chip="enterprise only"
                    chipColor="bg-amber-100 text-amber-700"
                  />
                </div>
              </div>

              {/* Record filter */}
              <div className="card p-5">
                <SectionLabel>Which records to process</SectionLabel>
                <div className="space-y-2">
                  {[
                    { key: 'include_pending',   label: `Pending review (${pendingCount})`, hint: 'Records flagged for human review' },
                    { key: 'include_fail_only', label: 'Hard fails only',                  hint: 'Records that scored below the fail floor - correction most needed' },
                    { key: 'include_all',       label: 'All records',                      hint: 'Every record in the store, regardless of status' },
                  ].map(({ key, label, hint }) => (
                    <label key={key} className="flex items-start gap-2.5 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={config.record_filter[key]}
                        onChange={e => {
                          const checked = e.target.checked;
                          const rf = { ...config.record_filter };
                          if (key === 'include_all' && checked) {
                            rf.include_pending = false;
                            rf.include_fail_only = false;
                          } else if (checked) {
                            rf.include_all = false;
                          }
                          rf[key] = checked;
                          setConfig({ record_filter: rf });
                        }}
                        className="accent-indigo-600 mt-0.5 shrink-0"
                      />
                      <div>
                        <span className="text-sm text-indigo-700 group-hover:text-indigo-900 transition-colors leading-none">
                          {label}
                        </span>
                        <p className="text-[11px] text-indigo-400 mt-0.5 leading-snug">{hint}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Right */}
            <div className="space-y-4">
              {/* Thresholds */}
              <div className={`card p-5 transition-opacity ${config.mode === 'drafts_only' ? 'opacity-50 pointer-events-none' : ''}`}>
                <SectionLabel>Auto-commit thresholds</SectionLabel>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs text-indigo-500 mb-1.5">
                      <span>Min IQS improvement to auto-commit</span>
                      <span className="font-mono-score">+{config.auto_commit_thresholds.min_iqs_delta.toFixed(2)}</span>
                    </div>
                    <input type="range" min={0.05} max={0.50} step={0.01}
                      value={config.auto_commit_thresholds.min_iqs_delta}
                      onChange={e => setConfig({ auto_commit_thresholds: { ...config.auto_commit_thresholds, min_iqs_delta: parseFloat(e.target.value) } })}
                      className="w-full accent-indigo-600"
                    />
                    <p className="text-xs text-indigo-400 mt-1">Below this delta, record goes back to review queue.</p>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-indigo-500 mb-1.5">
                      <span>Min absolute IQS after correction</span>
                      <span className="font-mono-score">{config.auto_commit_thresholds.min_iqs_floor.toFixed(2)}</span>
                    </div>
                    <input type="range" min={0.50} max={0.95} step={0.01}
                      value={config.auto_commit_thresholds.min_iqs_floor}
                      onChange={e => setConfig({ auto_commit_thresholds: { ...config.auto_commit_thresholds, min_iqs_floor: parseFloat(e.target.value) } })}
                      className="w-full accent-indigo-600"
                    />
                    <p className="text-xs text-indigo-400 mt-1">Even if delta is met, this floor must be reached.</p>
                  </div>
                </div>
              </div>

              {/* Preview or no-corrector warning */}
              {hasCorrector ? (
                <div className="card p-5 bg-indigo-25">
                  <SectionLabel>Preview</SectionLabel>
                  <dl className="space-y-2 text-sm">
                    {[
                      ['Records to process', recordsToProcess],
                      ['Est. LLM calls',     recordsToProcess],
                      ['Est. duration',      `${fmtDur(estDuration)}${providerLabel ? ` (${perRecordSecs}s/record)` : ''}`],
                      ...(providerLabel ? [['Provider', providerLabel]] : []),
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <dt className="text-indigo-500">{k}</dt>
                        <dd className="font-mono-score text-indigo-700 tabular-nums">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ) : (
                <div className="card p-5 border-amber-200 bg-amber-50">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle size={16} className="text-amber-500 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-amber-800 font-medium mb-1">LLM corrector not configured</p>
                      <p className="text-xs text-amber-700 mb-3">Configure a provider in Settings → LLM corrector before running the pipeline.</p>
                      <Button variant="secondary" size="sm" onClick={() => navigate('/settings#corrector')}>
                        Go to Settings →
                      </Button>
                    </div>
                  </div>
                </div>
              )}

              {/* Run button */}
              <Button
                variant={config.mode === 'fully_autonomous' ? 'secondary' : 'primary'}
                size="lg"
                fullWidth
                disabled={!hasCorrector || config.mode === 'fully_autonomous' || recordsToProcess === 0}
                icon={<Play size={15} />}
                onClick={handleStartRun}
              >
                {config.mode === 'fully_autonomous'
                  ? 'Enterprise feature - upgrade to Scroot Cloud'
                  : `Run pipeline on ${recordsToProcess} records`
                }
              </Button>

              {/* NLI guarantee strip */}
              <div className="flex items-start gap-1.5 text-[11px] text-indigo-400 px-1">
                <ShieldCheck size={12} className="text-indigo-300 shrink-0 mt-0.5" />
                <span>NLI re-scores every correction before any commit - regardless of mode. The LLM is the intern, NLI is the senior reviewer.</span>
              </div>
            </div>
          </div>
        )}

        {/* ─── RUNNING ───────────────────────────────────────────────── */}
        {view === 'running' && run && (
          <div className="max-w-2xl space-y-4">
            <div className="card p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin shrink-0" />
                <span className="text-sm font-medium text-indigo-950">
                  {run.status === 'paused' ? 'Paused' : `Processing ${run.total_records} records`}
                </span>
                <span className="font-mono-score text-sm text-indigo-500 ml-auto">
                  {run.processed_count} / {run.total_records}
                </span>
              </div>

              <ProgressBar value={run.processed_count} total={run.total_records} />

              <div className="flex gap-4 text-xs text-indigo-500 mt-3 font-mono-score">
                <span className="text-green-600">{run.committed_count} committed</span>
                <span className="text-amber-600">{run.review_queue_count} needs review</span>
                <span className="text-gray-500">{run.skipped_count} skipped</span>
                {run.failed_count > 0 && <span className="text-red-600">{run.failed_count} failed</span>}
              </div>
            </div>

            {/* Live log */}
            <div className="card p-5">
              <SectionLabel>Live log</SectionLabel>
              <LogPanel lines={run.log || []} />
            </div>

            {/* Controls */}
            <div className="flex items-center gap-3">
              {run.status === 'paused' ? (
                <Button variant="primary" size="sm" icon={<Play size={14} />} onClick={handleResume}>Resume</Button>
              ) : (
                <Button variant="secondary" size="sm" icon={<Pause size={14} />} onClick={handlePause}>Pause</Button>
              )}

              {cancelConfirm ? (
                <div className="flex items-center gap-2 animate-fade-in">
                  <span className="text-xs text-red-600">Cancel this run?</span>
                  <Button variant="danger" size="sm" onClick={handleCancel}>Confirm cancel</Button>
                  <Button variant="ghost" size="sm" onClick={() => setCancelConfirm(false)}>No</Button>
                </div>
              ) : (
                <Button variant="ghost" size="sm" icon={<StopCircle size={14} />}
                  className="text-red-600 hover:text-red-700 hover:bg-red-50 border-transparent"
                  onClick={() => setCancelConfirm(true)}>
                  Cancel run
                </Button>
              )}
            </div>

            <div className="flex items-start gap-1.5 text-[11px] text-indigo-300 px-1">
              <Info size={11} className="shrink-0 mt-0.5" />
              <span>Committed records cannot be undone from here - use Export to review.</span>
            </div>
          </div>
        )}

        {/* ─── RESULTS ───────────────────────────────────────────────── */}
        {view === 'results' && run && (
          <div className="max-w-4xl space-y-5">
            {/* Failed run */}
            {run.status === 'failed' && (
              <div className="card p-5 border-red-200 bg-red-50">
                <div className="flex items-start gap-3">
                  <AlertCircle size={20} className="text-red-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-800 mb-1">Pipeline failed</p>
                    <pre className="text-xs font-mono-score text-red-700 whitespace-pre-wrap">{run.error}</pre>
                    <div className="flex gap-2 mt-3">
                      <Button variant="secondary" size="sm" onClick={handleStartRun}>Retry</Button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Summary stat row */}
            <div className="grid grid-cols-4 gap-4">
              {[
                { label: 'Processed',      value: run.total_records,        color: '' },
                { label: 'Auto-committed', value: run.committed_count,      color: 'text-green-700' },
                { label: 'Needs review',   value: run.review_queue_count,   color: 'text-amber-700' },
                { label: 'Avg IQS lift',   value: run.summary ? `+${fmtIQS(run.summary.avg_delta)}` : '—', color: 'text-indigo-700', mono: true },
              ].map(({ label, value, color, mono }) => (
                <div key={label} className="card p-4">
                  <p className="section-label mb-2">{label}</p>
                  <p className={`text-2xl font-semibold leading-none ${color} ${mono ? 'font-mono-score' : ''}`}>{value}</p>
                </div>
              ))}
            </div>

            {/* Results table */}
            {sortedResults.length > 0 && (
              <div className="card p-5">
                <SectionLabel>Results ({run.results.length} records)</SectionLabel>
                <div className="space-y-2">
                  {sortedResults.map(r => (
                    <div
                      key={r.record_id}
                      onClick={r.outcome === 'review_queue' ? () => navigate(`/queue/${r.record_id}`) : undefined}
                      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg border border-indigo-50 hover:border-indigo-100 transition-colors
                        ${r.outcome === 'review_queue' ? 'cursor-pointer hover:bg-indigo-25' : ''}`}
                    >
                      <span className={`w-2 h-2 rounded-full shrink-0 ${OUTCOME_DOT[r.outcome] || 'bg-gray-400'}`} />
                      <span className="flex-1 text-sm text-indigo-700 truncate">{r.query_preview}</span>
                      <span className="font-mono-score text-xs text-indigo-400 shrink-0 tabular-nums">
                        {fmtIQS(r.iqs_before)} → {r.iqs_after != null ? fmtIQS(r.iqs_after) : '—'}
                      </span>
                      {/* Delta badge */}
                      <span className={`text-xs font-mono-score px-2 py-0.5 rounded-full border shrink-0 ${OUTCOME_STYLE[r.outcome]}`}>
                        {r.delta != null
                          ? `${r.delta > 0 ? '+' : ''}${r.delta.toFixed(2)} ${r.outcome === 'committed' ? '✓' : r.outcome === 'review_queue' ? '↷' : ''}`
                          : OUTCOME_LABEL[r.outcome]
                        }
                      </span>
                      <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full shrink-0 ${OUTCOME_STYLE[r.outcome]}`}>
                        {OUTCOME_LABEL[r.outcome]}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action row */}
            <div className="flex gap-3">
              <Button
                variant="secondary"
                size="sm"
                icon={<Inbox size={14} />}
                onClick={() => navigate(`/queue?run_id=${run.run_id}`)}
              >
                Review {run.review_queue_count} queued records
              </Button>
              <Button
                variant="secondary"
                size="sm"
                icon={<Download size={14} />}
                onClick={() => navigate(`/export?source=pipeline&run_id=${run.run_id}`)}
              >
                Export committed
              </Button>
              <Button
                variant="ghost"
                size="sm"
                icon={<RotateCcw size={14} />}
                onClick={() => resetRun()}
              >
                Run again
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
