import { useContext, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { TopBar } from '../components/TopBar';
import { FlagChip } from '../components/FlagChip';
import { ScoreBar } from '../components/ScoreBar';
import { EmptyState } from '../components/EmptyState';
import { DashboardContext } from '../context/DashboardContext';

const CATEGORIES = ['factual_error', 'missing_info', 'wrong_tone', 'policy_violation', 'hallucination', 'other'];
const REJECT_REASONS = ['actually_correct', 'poor_question', 'duplicate', 'out_of_scope', 'other'];

export function RecordReview() {
  const { id } = useParams();
  const navigate = useNavigate();
  const ctx = useContext(DashboardContext);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [correction, setCorrection] = useState('');
  const [category, setCategory] = useState('factual_error');
  const [notes, setNotes] = useState('');
  const [generating, setGenerating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [showReject, setShowReject] = useState(false);
  const [rejectReason, setRejectReason] = useState('actually_correct');
  const [rejectNotes, setRejectNotes] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    fetch(`/api/records/${id}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));

    // Release claim on unmount
    return () => fetch(`/api/queue/claim/${id}`, { method: 'DELETE' }).catch(() => {});
  }, [id]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [correction]);

  const generateCorrection = async () => {
    // SPEC: Never auto-populate - user must click ✨ Generate explicitly
    setGenerating(true);
    let full = '';
    const res = await fetch(`/api/records/${id}/generate-correction`, { method: 'POST' });
    if (!res.ok) { setGenerating(false); return; }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\n').filter(l => l.startsWith('data:'));
      for (const line of lines) {
        try {
          const payload = JSON.parse(line.slice(5));
          if (payload.error) { ctx?.addToast(payload.error, 'error'); break; }
          if (payload.done) { full = payload.suggestion; }
        } catch {}
      }
    }
    if (full) setCorrection(full);
    setGenerating(false);
  };

  const submit = async () => {
    if (!correction.trim()) return;
    setSubmitting(true);
    const res = await fetch(`/api/records/${id}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ correction, category, notes }),
    });
    setSubmitting(false);
    if (res.ok) {
      ctx?.addToast('Review submitted', 'success');
      ctx?.refreshStats?.();
      navigate('/queue');
    } else {
      ctx?.addToast('Failed to submit', 'error');
    }
  };

  const confirmReject = async () => {
    const res = await fetch(`/api/records/${id}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: rejectReason + (rejectNotes ? ': ' + rejectNotes : '') }),
    });
    if (res.ok) {
      ctx?.addToast('Record rejected', 'info');
      ctx?.refreshStats?.();
      navigate('/queue');
    }
    setShowReject(false);
  };

  if (loading) return (
    <div>
      <TopBar title="LOADING..." />
      <div style={{ padding: 24 }}>
        {[1,2].map(i => <div key={i} className="skeleton" style={{ height: 200, marginBottom: 16, borderRadius: 'var(--radius-md)' }} />)}
      </div>
    </div>
  );

  if (!data) return (
    <div>
      <TopBar title="RECORD NOT FOUND" />
      <EmptyState icon="◇" title="Record not found" subtitle={`No record with ID ${id}`} />
    </div>
  );

  const { record, scores, flags, context_chunks } = data;

  const btnStyle = (primary) => ({
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    fontSize: 13,
    padding: '8px 16px',
    border: primary ? 'none' : '1px solid var(--red)',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    background: primary ? 'var(--accent)' : 'transparent',
    color: primary ? '#0A0C10' : 'var(--red)',
    transition: 'opacity 120ms',
    opacity: primary && !correction.trim() ? 0.4 : 1,
  });

  return (
    <div className="page-enter">
      <TopBar
        title={<span>← <span onClick={() => navigate('/queue')} style={{ cursor: 'pointer', textDecoration: 'underline', textDecorationColor: 'var(--text-muted)' }}>BACK TO QUEUE</span></span>}
        actions={
          <>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>{id}</span>
            <button onClick={() => setShowReject(true)} style={btnStyle(false)}>Reject ✕</button>
            <button onClick={submit} disabled={!correction.trim() || submitting} style={btnStyle(true)}>
              {submitting ? 'Saving...' : 'Submit ✓'}
            </button>
          </>
        }
      />

      {/* 3-column layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', height: 'calc(100vh - var(--topbar-height))', overflow: 'hidden' }}>

        {/* Column 1: Context */}
        <div style={{ borderRight: '1px solid var(--bg-border)', overflowY: 'auto', padding: 20 }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>CONTEXT</div>

          {/* Query block */}
          <div style={{ background: 'var(--accent-dim)', borderLeft: '3px solid var(--accent)', padding: 12, borderRadius: 'var(--radius-sm)', marginBottom: 16 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>QUERY</div>
            <div style={{ fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.6 }}>{record.query}</div>
          </div>

          {/* Source chunks */}
          {context_chunks?.length > 0 ? (
            context_chunks.map((chunk, i) => (
              <div key={i} style={{ background: 'var(--bg-surface)', border: '1px solid var(--bg-border)', borderRadius: 'var(--radius-md)', padding: 12, marginBottom: 8 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', marginBottom: 6 }}>
                  {String(i + 1).padStart(2, '0')}
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{chunk}</div>
              </div>
            ))
          ) : (
            <div style={{ fontSize: 13, color: 'var(--text-muted)', fontStyle: 'italic' }}>No source context attached to this record</div>
          )}
        </div>

        {/* Column 2: Response + Scores */}
        <div style={{ borderRight: '1px solid var(--bg-border)', overflowY: 'auto', padding: 20 }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>RESPONSE</div>

          <div style={{ background: 'var(--bg-elevated)', borderRadius: 'var(--radius-md)', padding: 16, fontSize: 14, lineHeight: 1.6, color: 'var(--text-primary)', marginBottom: 20 }}>
            {record.response}
          </div>

          {/* IQS composite */}
          <div style={{ textAlign: 'center', marginBottom: 20 }}>
            <div style={{
              fontFamily: 'var(--font-mono)', fontSize: 32, fontWeight: 700,
              color: (scores?.iqs ?? 0) >= 0.8 ? 'var(--green)' : (scores?.iqs ?? 0) >= 0.6 ? 'var(--yellow)' : 'var(--red)',
            }}>
              IQS {(scores?.iqs ?? 0).toFixed(2)}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Information Quality Score
            </div>
          </div>

          {/* Score bars */}
          <div style={{ marginBottom: 20 }}>
            {[
              ['Groundedness', 'groundedness'],
              ['Completeness', 'completeness'],
              ['Relevance', 'relevance'],
              ['Consistency', 'consistency'],
              ['Confidence', 'confidence'],
            ].map(([label, key]) => (
              <ScoreBar key={key} label={label} value={scores?.[key] ?? 0} />
            ))}
          </div>

          {/* Flags */}
          {flags?.length > 0 && (
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>FLAGS</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {flags.map(f => <FlagChip key={f} flag={f} />)}
              </div>
            </div>
          )}
        </div>

        {/* Column 3: Correction editor */}
        <div style={{ overflowY: 'auto', padding: 20 }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>CORRECTION</div>

          <textarea
            ref={textareaRef}
            value={correction}
            onChange={e => setCorrection(e.target.value)}
            placeholder="Write the correct response..."
            style={{
              width: '100%',
              minHeight: 180,
              background: 'var(--bg-surface)',
              border: `1px solid ${correction ? 'var(--accent)' : 'var(--bg-border)'}`,
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: 14,
              lineHeight: 1.6,
              padding: 12,
              resize: 'none',
              outline: 'none',
              transition: 'border-color 150ms',
              marginBottom: 12,
              boxSizing: 'border-box',
            }}
          />

          {/* ✨ Generate - NEVER auto-populates, user must click explicitly */}
          <button
            onClick={generateCorrection}
            disabled={generating}
            style={{
              width: '100%',
              background: 'transparent',
              border: '1px solid var(--accent)',
              color: generating ? 'var(--text-muted)' : 'var(--accent)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              fontSize: 13,
              padding: '8px 16px',
              borderRadius: 'var(--radius-sm)',
              cursor: generating ? 'not-allowed' : 'pointer',
              marginBottom: 16,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              transition: 'all 120ms',
            }}
          >
            {generating ? '⟳ Generating...' : '✨ Generate Suggestion'}
          </button>

          {/* Category */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>CATEGORY</div>
            <select
              value={category}
              onChange={e => setCategory(e.target.value)}
              style={{
                width: '100%',
                background: 'var(--bg-surface)',
                border: '1px solid var(--bg-border)',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontSize: 13,
                padding: '6px 10px',
                outline: 'none',
                cursor: 'pointer',
              }}
            >
              {CATEGORIES.map(c => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
            </select>
          </div>

          {/* Notes */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>NOTES (OPTIONAL)</div>
            <textarea
              value={notes}
              onChange={e => setNotes(e.target.value)}
              placeholder="Optional notes for context..."
              rows={3}
              style={{
                width: '100%',
                background: 'var(--bg-surface)',
                border: '1px solid var(--bg-border)',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontSize: 13,
                lineHeight: 1.5,
                padding: 10,
                resize: 'vertical',
                outline: 'none',
                boxSizing: 'border-box',
              }}
            />
          </div>

          {/* Submit */}
          <button
            onClick={submit}
            disabled={!correction.trim() || submitting}
            style={{
              width: '100%',
              background: correction.trim() ? 'var(--accent)' : 'var(--bg-border)',
              color: correction.trim() ? '#0A0C10' : 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              fontSize: 14,
              padding: '12px',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: correction.trim() ? 'pointer' : 'not-allowed',
              transition: 'all 150ms',
            }}
          >
            {submitting ? 'Saving...' : 'Submit Review ✓'}
          </button>
        </div>
      </div>

      {/* Reject modal */}
      {showReject && (
        <div className="modal-backdrop" onClick={() => setShowReject(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>
              Reject this record
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>REASON</div>
              {REJECT_REASONS.map(r => (
                <label key={r} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, cursor: 'pointer' }}>
                  <input type="radio" name="reject" value={r} checked={rejectReason === r}
                    onChange={() => setRejectReason(r)} style={{ accentColor: 'var(--accent)' }} />
                  <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{r.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>
            <textarea
              placeholder="Additional notes..."
              value={rejectNotes}
              onChange={e => setRejectNotes(e.target.value)}
              rows={2}
              style={{ width: '100%', background: 'var(--bg-base)', border: '1px solid var(--bg-border)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontFamily: 'var(--font-sans)', fontSize: 13, padding: 8, resize: 'none', outline: 'none', boxSizing: 'border-box', marginBottom: 16 }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setShowReject(false)} style={{ background: 'transparent', border: '1px solid var(--bg-border)', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)', fontSize: 13, padding: '7px 14px', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }}>
                Cancel
              </button>
              <button onClick={confirmReject} style={{ background: 'var(--red)', color: '#fff', fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 13, padding: '7px 14px', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer' }}>
                Confirm Reject
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
