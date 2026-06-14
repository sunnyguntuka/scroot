import { useState } from 'react';
import { CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';

const STATUS_STYLES = {
  supported: {
    row: 'bg-green-50 border-l-2 border-green-400',
    icon: CheckCircle2,
    iconClass: 'text-green-500',
    label: 'Supported',
    labelClass: 'text-green-600',
  },
  contradiction: {
    row: 'bg-red-50 border-l-2 border-red-500',
    icon: XCircle,
    iconClass: 'text-red-500',
    label: 'Contradicts context',
    labelClass: 'text-red-600',
  },
  unsupported: {
    row: 'bg-amber-50 border-l-2 border-amber-300',
    icon: AlertTriangle,
    iconClass: 'text-amber-500',
    label: 'No grounding',
    labelClass: 'text-amber-600',
  },
};

const CHUNK_TRUNCATE = 130;

function entryStatus(entry) {
  if (entry.contradiction_detected) return 'contradiction';
  if (entry.supported) return 'supported';
  return 'unsupported';
}

/**
 * One response sentence with its evidence shown inline (no hover tooltip):
 * a status label, the NLI entailment score, and the best-matching context
 * chunk (truncated, expandable). Inline rather than a hover popover so it is
 * never clipped by the panel's scroll container and works without a mouse.
 */
function EvidenceEntryRow({ entry }) {
  const [expanded, setExpanded] = useState(false);
  const { row, icon: Icon, iconClass, label, labelClass } = STATUS_STYLES[entryStatus(entry)];

  const hasScore =
    entry.entailment_score !== null && entry.entailment_score !== undefined;
  const chunk = entry.best_matching_chunk;
  const isLong = chunk && chunk.length > CHUNK_TRUNCATE;
  const shownChunk =
    !chunk || expanded || !isLong ? chunk : `${chunk.slice(0, CHUNK_TRUNCATE).trimEnd()}…`;

  return (
    <div className={`flex items-start gap-2 rounded-r px-2.5 py-1.5 ${row}`}>
      <Icon size={13} className={`mt-0.5 shrink-0 ${iconClass}`} />
      <div className="min-w-0 flex-1">
        <p className="text-xs text-indigo-900 leading-snug">{entry.response_sentence}</p>

        <div className="mt-1 flex items-center justify-between gap-2">
          <span className={`text-[10px] font-medium uppercase tracking-wide ${labelClass}`}>
            {label}
          </span>
          {hasScore && (
            <span
              className="shrink-0 font-mono-score text-[10px] text-indigo-500"
              title="NLI entailment probability against the best-matching context chunk (0-1)"
            >
              entailment {entry.entailment_score.toFixed(2)}
            </span>
          )}
        </div>

        {chunk && (
          <div className="mt-1">
            <p className="text-[11px] text-indigo-500 leading-snug">
              <span className="text-indigo-400">matched: </span>
              {shownChunk}
            </p>
            {isLong && (
              <button
                type="button"
                onClick={() => setExpanded((v) => !v)}
                className="mt-0.5 text-[10px] text-indigo-400 hover:text-indigo-600 transition-colors"
              >
                {expanded ? 'show less' : 'show more'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Sentence-level NLI attribution: which response sentences are supported,
 * contradicted, or ungrounded by the retrieved context. Renders the
 * `evidence_map` shape returned by `EntailmentResult.to_dict()`.
 */
export function EvidenceMap({ evidenceMap }) {
  if (!evidenceMap || !evidenceMap.entries || evidenceMap.entries.length === 0) {
    return (
      <div className="text-sm text-indigo-400 leading-relaxed">
        <p>No evidence map available - this response was scored without retrieved context.</p>
        <p className="mt-1.5 text-xs text-indigo-300">
          Pass grounding documents via <code className="font-mono-score">ContextBuilder</code> (see{' '}
          <code className="font-mono-score">docs/context_builder.md</code>) to enable
          sentence-level evidence attribution.
        </p>
      </div>
    );
  }

  const { supported, unsupported, contradictions, entries } = evidenceMap;
  const total = entries.length;

  return (
    <div className="space-y-2">
      <p className="text-[11px] text-indigo-400">
        Coverage: {supported}/{total} sentence{total === 1 ? '' : 's'} grounded
        {contradictions > 0 && ` · ${contradictions} contradiction${contradictions === 1 ? '' : 's'}`}
        {unsupported > 0 && ` · ${unsupported} unsupported`}
      </p>
      <div className="space-y-1.5">
        {entries.map((entry, i) => (
          <EvidenceEntryRow key={i} entry={entry} />
        ))}
      </div>
    </div>
  );
}
