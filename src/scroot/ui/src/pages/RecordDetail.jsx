import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ChevronLeft, Sparkles, X, HelpCircle, Copy, CheckCircle2, AlertCircle } from 'lucide-react';
import { TopBar } from '../components/layout/TopBar';
import { IQSBadge } from '../components/ui/IQSBadge';
import { MetricBar } from '../components/ui/MetricBar';
import { EvidenceMap } from '../components/ui/EvidenceMap';
import { ScoreRing } from '../components/ui/ScoreRing';
import { FlagChip } from '../components/ui/FlagChip';
import { Button } from '../components/ui/Button';
import { useToast } from '../components/ui/Toast';
import { api } from '../api/client';
import { STATUS_COLORS, getIQSStatus, ALL_METRICS, fmtIQS } from '../utils/iqs';
import { fmtDate } from '../utils/time';

const REJECT_REASONS = [
  { value: 'response_is_correct', label: 'Response is actually correct' },
  { value: 'wrong_context',       label: 'Wrong context was provided' },
  { value: 'out_of_scope',        label: 'Out of scope for this agent' },
  { value: 'scoring_error',       label: 'Scoring looks incorrect' },
  { value: 'other',               label: 'Other' },
];

// The ScoreRing + status line already show the IQS number, so strip the
// redundant "IQS 0.29 - " prefix the explanation string carries for standalone
// (log/CLI) use, and capitalize what remains.
function stripIqsPrefix(text) {
  if (!text) return text;
  const stripped = text.replace(/^IQS\s+[\d.]+\s*[-–]\s*/i, '');
  return stripped.charAt(0).toUpperCase() + stripped.slice(1);
}

// ─── Section label ────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return (
    <p className="section-label mb-3">{children}</p>
  );
}

// ─── Panel card ──────────────────────────────────────────────────────
function PanelCard({ label, children, className = '', scrollable = false, maxH }) {
  return (
    <div className={`card p-4 ${className}`}>
      {label && <SectionLabel>{label}</SectionLabel>}
      <div className={scrollable ? 'overflow-y-auto' : ''} style={maxH ? { maxHeight: maxH } : undefined}>
        {children}
      </div>
    </div>
  );
}

// ─── Copy icon button ─────────────────────────────────────────────────
function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button onClick={copy} className="text-indigo-300 hover:text-indigo-600 transition-colors" aria-label="Copy">
      {copied ? <CheckCircle2 size={13} className="text-green-500" /> : <Copy size={13} />}
    </button>
  );
}

// ─── Reject popover ───────────────────────────────────────────────────
function RejectPopover({ onConfirm, onCancel }) {
  const [reason, setReason] = useState('');

  return (
    <div className="mt-2 card p-4 border-red-100 animate-fade-in">
      <p className="text-sm font-medium text-indigo-950 mb-3">Why are you rejecting this?</p>
      <div className="space-y-2 mb-4">
        {REJECT_REASONS.map(r => (
          <label key={r.value} className="flex items-center gap-2.5 cursor-pointer group">
            <input
              type="radio" name="reject-reason" value={r.value}
              checked={reason === r.value}
              onChange={() => setReason(r.value)}
              className="accent-red-600"
            />
            <span className="text-sm text-indigo-700 group-hover:text-indigo-900 transition-colors">
              {r.label}
            </span>
          </label>
        ))}
      </div>
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button>
        <Button
          variant="danger"
          size="sm"
          disabled={!reason}
          onClick={() => onConfirm(reason)}
        >
          Confirm rejection
        </Button>
      </div>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────
export function RecordDetail({ threshold = 0.70, hasLlmCorrector = false, corrector = null }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const addToast = useToast();
  const textareaRef = useRef(null);

  const [record, setRecord] = useState(null);
  const [loading, setLoading] = useState(true);
  const [correction, setCorrection] = useState('');
  const [generating, setGenerating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [showReject, setShowReject] = useState(false);

  // Derive state from record
  const correctionState =
    record?.correction ? 'submitted' :
    record?.rejection_reason ? 'rejected' :
    'unreviewed';

  useEffect(() => {
    api.getRecord(id)
      .then(r => { setRecord(r); setCorrection(r.correction || ''); })
      .catch(() => addToast('Failed to load record', 'error'))
      .finally(() => setLoading(false));
  }, [id]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 400)}px`;
    }
  }, [correction]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); handleSubmit(); }
      if (e.key === 'Escape') navigate('/queue');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  });

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const { draft } = await api.generateCorrection(id);
      setCorrection(draft);
    } catch {
      addToast('Failed to generate correction', 'error');
    } finally {
      setGenerating(false);
    }
  };

  const handleSubmit = async () => {
    if (!correction.trim() || submitting) return;
    setSubmitting(true);
    try {
      const updated = await api.submitCorrection(id, correction.trim());
      setRecord(updated);
      addToast('Correction stored in feedback loop', 'success');
    } catch {
      addToast('Failed to submit correction', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReject = async (reason) => {
    try {
      const updated = await api.rejectRecord(id, reason);
      setRecord(updated);
      setShowReject(false);
      addToast('Record marked as rejected', 'info');
    } catch {
      addToast('Failed to reject record', 'error');
    }
  };

  const handleUndoRejection = async () => {
    try {
      const updated = await api.deleteCorrection(id);
      setRecord(updated);
    } catch {
      addToast('Failed to undo rejection', 'error');
    }
  };

  // ─── Loading ─────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <TopBar title="Loading…" />
        <div className="flex-1 p-6 grid grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="skeleton h-64 rounded-card" />
          ))}
        </div>
      </div>
    );
  }

  if (!record) {
    return (
      <div className="flex flex-col h-full">
        <TopBar title="Record not found" />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <AlertCircle size={32} className="text-red-300 mx-auto mb-3" />
            <p className="text-sm text-indigo-400">No record with ID {id}</p>
            <Button variant="ghost" size="sm" className="mt-3" onClick={() => navigate('/queue')}>
              Back to queue
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const status = getIQSStatus(record.iqs, threshold);
  const c = STATUS_COLORS[status];

  return (
    <div className="flex flex-col h-full page-enter">
      {/* Top bar */}
      <TopBar
        title={
          <span className="flex items-center gap-2">
            <button
              onClick={() => navigate('/queue')}
              className="flex items-center gap-1 text-indigo-400 hover:text-indigo-700 transition-colors text-sm font-normal"
            >
              <ChevronLeft size={15} />Back to queue
            </button>
          </span>
        }
      />

      {/* Header strip */}
      <div className="bg-white border-b border-indigo-100 px-6 py-3 flex items-center justify-between shrink-0">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <span className="text-sm font-mono-score text-indigo-400">#{record.id?.slice(-4) || id}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full border ${c.bg} ${c.text} ${c.border}`}>
              {status}
            </span>
          </div>
          <div className="text-[11px] text-indigo-400">
            {[record.model, record.agent_id].filter(Boolean).join(' · ')}
            {record.created_at && ` · ${fmtDate(record.created_at)}`}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <ScoreRing score={record.iqs} threshold={threshold} size={64} />
          <div>
            <div className="text-[28px] font-mono-score font-semibold leading-none" style={{ color: c.hex }}>
              {fmtIQS(record.iqs)}
            </div>
            <div className="text-[10px] uppercase tracking-wider text-indigo-400 mt-0.5">
              IQS
              {record.context_used === false && (
                <span
                  className="ml-1 normal-case tracking-normal text-indigo-300"
                  title="Computed from 4 of 5 metrics — groundedness not scored (no context). Add context for a complete score."
                >
                  ({record.iqs_metric_count ?? 4}/5)
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 3-column body */}
      <div className="flex-1 overflow-hidden grid grid-cols-3 divide-x divide-indigo-100">

        {/* Column 1 - Context */}
        <div className="overflow-y-auto p-5 space-y-3">
          <SectionLabel>Context</SectionLabel>

          <PanelCard label="Query">
            <p className="text-sm text-indigo-950 leading-relaxed select-text">{record.query}</p>
          </PanelCard>

          <PanelCard label="Context / grounding docs" scrollable maxH="280px">
            {record.context ? (
              <p className="text-sm text-indigo-700 leading-relaxed select-text whitespace-pre-wrap">
                {record.context}
              </p>
            ) : (
              <p className="text-sm text-indigo-300 italic">No context provided</p>
            )}
          </PanelCard>

          <PanelCard label="Metadata">
            <dl className="space-y-1.5 text-sm">
              {[
                ['Model',     record.model    || '—'],
                ['Agent',     record.agent_id || '—'],
                ['Scored at', fmtDate(record.created_at)],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="text-indigo-400 shrink-0">{k}</dt>
                  <dd className="text-indigo-700 text-right truncate">{v}</dd>
                </div>
              ))}
              <div className="flex justify-between gap-2 pt-1 border-t border-indigo-50">
                <dt className="text-indigo-400 text-xs font-mono-score">Record ID</dt>
                <dd className="flex items-center gap-1">
                  <span className="text-xs font-mono-score text-indigo-500">{record.id}</span>
                  <CopyBtn text={record.id} />
                </dd>
              </div>
            </dl>
          </PanelCard>
        </div>

        {/* Column 2 - Response + Scores */}
        <div className="overflow-y-auto p-5 space-y-3">
          <SectionLabel>Response</SectionLabel>

          <PanelCard scrollable maxH="320px">
            <p className="text-sm text-indigo-950 leading-relaxed select-text whitespace-pre-wrap">
              {record.response}
            </p>
          </PanelCard>

          <PanelCard label="Information quality score">
            {/* IQS summary - the number lives in the ring; don't repeat it */}
            <div className="flex items-center gap-3 mb-4">
              <ScoreRing score={record.iqs} threshold={threshold} size={48} />
              <div>
                <div className={`text-sm font-medium ${c.text}`}>
                  {status === 'pass' ? `Above threshold (${threshold})` : `Below threshold (${threshold})`}
                </div>
                {record.iqs_explanation && (
                  <div className="text-[11px] text-indigo-400 mt-0.5 flex items-center gap-1.5">
                    <span>{stripIqsPrefix(record.iqs_explanation)}</span>
                    {record.score_variance > 0.30 && (
                      <span className="px-1.5 py-0.5 rounded-full bg-amber-50 text-amber-600 text-[10px] font-medium shrink-0">
                        high spread
                      </span>
                    )}
                  </div>
                )}
                {record.context_used === false && (
                  <div className="text-[10px] text-indigo-300 mt-0.5">
                    Scored from {record.iqs_metric_count ?? 4} of 5 metrics · no context provided
                  </div>
                )}
              </div>
            </div>

            <div className="border-t border-indigo-50 pt-3 space-y-2.5">
              {ALL_METRICS.map(m => (
                <MetricBar key={m} metric={m} value={record.metrics?.[m] ?? null} explanation={record.metric_explanations?.[m]} />
              ))}
            </div>

            <div className="border-t border-indigo-50 mt-3 pt-2.5 flex items-center gap-1.5">
              <span className="text-[11px] text-indigo-400">IQS = weighted harmonic mean</span>
              <div className="relative group">
                <HelpCircle size={12} className="text-indigo-300 cursor-help" />
                <div className="absolute bottom-5 left-0 w-60 p-2.5 bg-indigo-950 text-white text-xs rounded-lg
                                opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 shadow-xl">
                  IQS is the weighted harmonic mean of all metric scores.
                  A score of 0 on any metric reduces IQS to near 0.
                </div>
              </div>
            </div>

            {/* Flags */}
            {(record.flags || []).length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {record.flags.map(f => <FlagChip key={f} metric={f} value={record.metrics?.[f]} />)}
              </div>
            )}
          </PanelCard>

          <PanelCard label="Evidence map" scrollable maxH="280px">
            <EvidenceMap evidenceMap={record.evidence_map} />
          </PanelCard>
        </div>

        {/* Column 3 - Correction */}
        <div className="overflow-y-auto p-5">
          <SectionLabel>Correction</SectionLabel>

          {/* State A: Unreviewed */}
          {correctionState === 'unreviewed' && (
            <div className="space-y-3">
              <textarea
                ref={textareaRef}
                value={correction}
                onChange={e => setCorrection(e.target.value)}
                placeholder="Write a corrected response…"
                className="w-full min-h-[200px] input resize-none leading-relaxed text-sm"
                style={{ fontFamily: 'inherit' }}
              />

              <div className="flex gap-2">
                {hasLlmCorrector ? (
                  <div className="flex flex-col gap-1">
                    <Button
                      variant="secondary"
                      size="sm"
                      loading={generating}
                      icon={!generating && <Sparkles size={14} />}
                      onClick={handleGenerate}
                      disabled={generating}
                    >
                      {generating
                        ? (corrector?.mode === 'local'
                            ? `Generating with ${corrector?.local?.model_name || 'local model'} (~10–15s)…`
                            : 'Generating…')
                        : 'Generate'}
                    </Button>
                    {!generating && corrector?.mode === 'local' && (
                      <p className="text-[10px] text-indigo-400">~10–15s on CPU with local LLM</p>
                    )}
                  </div>
                ) : corrector?.mode === 'local' && !corrector?.local?.model_downloaded ? (
                  <button
                    onClick={() => navigate('/settings#corrector')}
                    className="text-[11px] text-indigo-400 hover:text-indigo-600 underline"
                  >
                    Download local model →
                  </button>
                ) : (
                  <button
                    onClick={() => navigate('/settings#corrector')}
                    className="text-[11px] text-indigo-400 hover:text-indigo-600 underline"
                  >
                    Configure LLM corrector →
                  </button>
                )}

                <Button
                  variant="primary"
                  size="sm"
                  loading={submitting}
                  disabled={!correction.trim()}
                  onClick={handleSubmit}
                  className="flex-1"
                >
                  Submit correction
                </Button>
              </div>

              <div className="relative">
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<X size={14} />}
                  onClick={() => setShowReject(!showReject)}
                  className="text-red-600 hover:text-red-700 hover:bg-red-50 border-transparent"
                >
                  Reject
                </Button>
                {showReject && (
                  <RejectPopover
                    onConfirm={handleReject}
                    onCancel={() => setShowReject(false)}
                  />
                )}
              </div>

              {/* Keyboard shortcuts hint */}
              <div className="pt-2 border-t border-indigo-50 text-[10px] text-indigo-300 space-y-0.5">
                <div>⌘ Enter - Submit · ⌘ K - Generate · Esc - Back</div>
              </div>
            </div>
          )}

          {/* State B: Correction submitted */}
          {correctionState === 'submitted' && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 px-3 py-2.5 bg-green-50 border border-green-200 rounded-lg">
                <CheckCircle2 size={15} className="text-green-600 shrink-0" />
                <span className="text-sm text-green-700 font-medium">Correction stored in feedback loop</span>
              </div>
              <div className="bg-green-50 border border-green-100 rounded-card p-3">
                <p className="text-sm text-green-800 leading-relaxed whitespace-pre-wrap select-text">
                  {record.correction}
                </p>
              </div>
              <p className="text-[11px] text-indigo-400">
                {record.guardrail_applied_count > 0
                  ? `Guardrail status: Active - applied to ${record.guardrail_applied_count} subsequent prompt${record.guardrail_applied_count === 1 ? '' : 's'}`
                  : 'Guardrail status: Not yet applied'}
              </p>
              <div className="flex gap-2">
                <Button variant="secondary" size="sm"
                  onClick={() => { setRecord(r => ({ ...r, correction: null })); setCorrection(record.correction || ''); }}>
                  Edit correction
                </Button>
                <Button variant="ghost" size="sm"
                  onClick={() => api.deleteCorrection(id).then(r => { setRecord(r); setCorrection(''); }).catch(() => addToast('Failed', 'error'))}>
                  Remove
                </Button>
              </div>
            </div>
          )}

          {/* State C: Rejected */}
          {correctionState === 'rejected' && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 px-3 py-2.5 bg-red-50 border border-red-200 rounded-lg">
                <X size={15} className="text-red-600 shrink-0" />
                <span className="text-sm text-red-700">
                  Marked as rejected · {REJECT_REASONS.find(r => r.value === record.rejection_reason)?.label || record.rejection_reason}
                </span>
              </div>
              <Button variant="secondary" size="sm" onClick={handleUndoRejection}>
                Undo rejection
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
