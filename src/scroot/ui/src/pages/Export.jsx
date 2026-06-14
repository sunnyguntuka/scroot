import { useEffect, useState } from 'react';
import { Download, Zap } from 'lucide-react';
import { TopBar } from '../components/layout/TopBar';
import { Button } from '../components/ui/Button';
import { useToast } from '../components/ui/Toast';
import { api } from '../api/client';

const SAMPLE_JSONL = `{
  "id": "uuid-0042",
  "query": "Summarize Q3 earnings report",
  "original_response": "Q3 was decent...",
  "corrected_response": "Q3 revenue grew 12% YoY...",
  "iqs_before": 0.54,
  "iqs_after": 0.81,
  "flags": ["consistency", "completeness"],
  "model": "gpt-4o",
  "agent_id": "support-bot-v2"
}`;

function SectionLabel({ children }) {
  return <p className="section-label mb-3">{children}</p>;
}

function RadioGroup({ name, options, value, onChange }) {
  return (
    <div className="space-y-2">
      {options.map(opt => (
        <label key={opt.value} className="flex items-center gap-2.5 cursor-pointer group">
          <input
            type="radio" name={name} value={opt.value}
            checked={value === opt.value}
            onChange={() => onChange(opt.value)}
            className="accent-indigo-600"
          />
          <span className="text-sm text-indigo-700 group-hover:text-indigo-900 transition-colors">
            {opt.label}
            {opt.sub && <span className="text-indigo-400 text-xs ml-1.5">{opt.sub}</span>}
          </span>
        </label>
      ))}
    </div>
  );
}

function Checkbox({ label, checked, onChange }) {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer group">
      <input
        type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)}
        className="accent-indigo-600 rounded"
      />
      <span className="text-sm text-indigo-700 group-hover:text-indigo-900 transition-colors">
        {label}
      </span>
    </label>
  );
}

export function Export({ avgIqs, threshold = 0.70 }) {
  const addToast = useToast();

  const [format, setFormat]             = useState('jsonl');
  const [inclReviewed, setInclReviewed] = useState(true);
  const [inclRejected, setInclRejected] = useState(true);
  const [inclPending,  setInclPending]  = useState(false);
  const [minImprovement, setMinImprovement] = useState(0);
  const [dateFrom, setDateFrom]         = useState('');
  const [dateTo,   setDateTo]           = useState('');
  const [agentFilter, setAgentFilter]   = useState('');
  const [matched,  setMatched]          = useState(null);
  const [correctedCount, setCorrectedCount] = useState(null);
  const [agents,   setAgents]           = useState([]);
  const [downloading, setDownloading]   = useState(false);

  const FINETUNE_THRESHOLD = 50;

  const exportParams = () => ({
    format,
    include_reviewed: inclReviewed,
    include_rejected: inclRejected,
    include_pending:  inclPending,
    min_iqs_improvement: minImprovement,
    ...(agentFilter && { agent: agentFilter }),
    ...(dateFrom && { date_from: dateFrom }),
    ...(dateTo   && { date_to:   dateTo }),
  });

  useEffect(() => {
    api.getExportPreview(exportParams())
      .then(d => {
        setMatched(d.count ?? d.matched ?? null);
        setCorrectedCount(d.corrected_count ?? null);
        if (d.agents?.length) setAgents(d.agents);
      })
      .catch(() => { setMatched(null); setCorrectedCount(null); });
  }, [format, inclReviewed, inclRejected, inclPending, minImprovement, agentFilter, dateFrom, dateTo]);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const res = await api.downloadExport(exportParams());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `scroot-export.${format}`;
      a.click();
      URL.revokeObjectURL(url);
      addToast(`Export downloaded (${format.toUpperCase()})`, 'success');
    } catch {
      addToast('Download failed. Check API connection.', 'error');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="flex flex-col h-full page-enter">
      <TopBar title="Export" avgIqs={avgIqs} threshold={threshold} />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-[2fr_3fr] gap-6 max-w-5xl">

          {/* Left - Filters */}
          <div className="space-y-5">
            <div className="card p-5">
              <SectionLabel>Format</SectionLabel>
              <RadioGroup
                name="format"
                value={format}
                onChange={setFormat}
                options={[
                  { value: 'jsonl', label: 'JSONL', sub: '— fine-tuning format' },
                  { value: 'csv',   label: 'CSV',   sub: '— flat, all fields' },
                ]}
              />
            </div>

            <div className="card p-5">
              <SectionLabel>Records to include</SectionLabel>
              <div className="space-y-2">
                <Checkbox label="Reviewed + corrected" checked={inclReviewed} onChange={setInclReviewed} />
                <Checkbox label="Reviewed + rejected"  checked={inclRejected} onChange={setInclRejected} />
                <Checkbox label="Pending (unreviewed)" checked={inclPending}  onChange={setInclPending} />
              </div>
            </div>

            <div className="card p-5">
              <SectionLabel>IQS improvement filter</SectionLabel>
              <p className="text-xs text-indigo-400 mb-3">Only records where IQS improved by at least:</p>
              <div className="flex items-center gap-3">
                <input
                  type="range" min={0} max={0.5} step={0.01}
                  value={minImprovement}
                  onChange={e => setMinImprovement(parseFloat(e.target.value))}
                  className="flex-1 accent-indigo-600"
                />
                <span className="text-sm font-mono-score text-indigo-700 w-10 text-right">
                  {minImprovement.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="card p-5">
              <SectionLabel>Date range</SectionLabel>
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-indigo-400 mb-1 block">From</label>
                  <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} className="input text-sm h-9" />
                </div>
                <div>
                  <label className="text-xs text-indigo-400 mb-1 block">To</label>
                  <input type="date" value={dateTo}   onChange={e => setDateTo(e.target.value)}   className="input text-sm h-9" />
                </div>
              </div>
            </div>

            {agents.length > 0 && (
              <div className="card p-5">
                <SectionLabel>Agent</SectionLabel>
                <select
                  value={agentFilter}
                  onChange={e => setAgentFilter(e.target.value)}
                  className="w-full border border-indigo-100 rounded-lg px-3 py-2 text-sm text-indigo-700
                             bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">All agents</option>
                  {agents.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </div>
            )}
          </div>

          {/* Right - Preview + download */}
          <div className="space-y-4">
            <div className="card p-5">
              <div className="flex items-baseline gap-2 mb-4">
                <span className="text-2xl font-semibold text-indigo-950">
                  {matched !== null ? matched : '—'}
                </span>
                <span className="text-sm text-indigo-400">records matched</span>
              </div>

              {/* Fine-tuning readiness indicator */}
              {correctedCount !== null && (
                <div className="mb-5 pb-5 border-b border-indigo-50">
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-indigo-400">Fine-tuning readiness</span>
                    <span className="font-mono-score text-indigo-600 tabular-nums">
                      {correctedCount} / {FINETUNE_THRESHOLD}
                    </span>
                  </div>
                  <div className="h-1.5 bg-indigo-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full transition-all duration-500 ease-out"
                      style={{ width: `${Math.min(100, Math.round((correctedCount / FINETUNE_THRESHOLD) * 100))}%` }}
                    />
                  </div>
                  {correctedCount >= FINETUNE_THRESHOLD ? (
                    <p className="text-xs text-green-600 mt-1.5 font-medium">Ready for fine-tuning</p>
                  ) : (
                    <p className="text-xs text-indigo-300 mt-1.5">
                      {FINETUNE_THRESHOLD - correctedCount} more corrected records recommended
                    </p>
                  )}
                </div>
              )}

              <SectionLabel>Preview (first 3 records)</SectionLabel>
              <pre
                className="bg-indigo-25 border border-indigo-100 rounded-card p-4
                           text-[12px] font-mono-score text-indigo-800 overflow-y-auto leading-relaxed"
                style={{ maxHeight: 260, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}
              >
                {SAMPLE_JSONL}
              </pre>
            </div>

            <Button
              variant="primary"
              size="lg"
              fullWidth
              loading={downloading}
              icon={<Download size={16} />}
              onClick={handleDownload}
              disabled={matched === 0}
            >
              Download {format.toUpperCase()}
            </Button>

            {/* Cloud nudge - footnote, never a banner */}
            <p className="text-[11px] text-indigo-300 flex items-center gap-1.5 px-1">
              <Zap size={11} strokeWidth={2} />
              Push to S3/GCS and compliance exports available in{' '}
              <a
                href="https://scroot.dev/cloud"
                target="_blank"
                rel="noopener noreferrer"
                className="text-indigo-400 underline underline-offset-2 hover:text-indigo-600 transition-colors"
              >
                Scroot Cloud →
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
