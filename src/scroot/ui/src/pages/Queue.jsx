import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Search, Inbox, CheckCircle2, Download, X, Code2 } from 'lucide-react';
import { TopBar } from '../components/layout/TopBar';
import { IQSBadge } from '../components/ui/IQSBadge';
import { FlagChip, FlagOverflow } from '../components/ui/FlagChip';
import { Button } from '../components/ui/Button';
import { useToast } from '../components/ui/Toast';
import { api } from '../api/client';
import { STATUS_COLORS, getIQSStatus, ALL_METRICS } from '../utils/iqs';
import { relativeTime } from '../utils/time';

// ─── Skeleton row ────────────────────────────────────────────────────
function SkeletonRow() {
  return (
    <div className="card px-4 py-3 mb-2 animate-pulse">
      <div className="flex items-start gap-3">
        <div className="w-2 h-2 rounded-full bg-indigo-100 mt-1.5 shrink-0" />
        <div className="flex-1 space-y-2">
          <div className="h-3.5 bg-indigo-50 rounded w-3/4" />
          <div className="h-3 bg-indigo-50 rounded w-1/2" />
          <div className="flex gap-2">
            <div className="h-4 w-14 bg-indigo-50 rounded-full" />
            <div className="h-4 w-16 bg-indigo-50 rounded-full" />
          </div>
        </div>
        <div className="h-6 w-10 bg-indigo-50 rounded-full" />
      </div>
    </div>
  );
}

// ─── Record row card ─────────────────────────────────────────────────
const RecordRow = function RecordRow({ record, selected, onSelect, threshold = 0.70 }) {
  const navigate = useNavigate();
  const status = getIQSStatus(record.iqs, threshold);
  const c = STATUS_COLORS[status];
  const visibleFlags = (record.flags || []).slice(0, 3);
  const overflow = Math.max(0, (record.flags || []).length - 3);

  return (
    <div
      className={`card card-hover px-4 py-3 mb-2 cursor-pointer
        ${status === 'fail' ? 'border-l-[3px] border-l-red-400' : ''}`}
      onClick={() => navigate(`/queue/${record.id}`)}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && navigate(`/queue/${record.id}`)}
    >
      <div className="flex items-start gap-3">
        {/* Selection checkbox */}
        <input
          type="checkbox"
          checked={selected}
          onChange={e => { e.stopPropagation(); onSelect(record.id, e.target.checked); }}
          onClick={e => e.stopPropagation()}
          className="mt-1 accent-indigo-600 shrink-0"
          aria-label={`Select record ${record.id}`}
        />

        {/* Status dot - 10px for readability at list density */}
        <span className={`w-2.5 h-2.5 rounded-full mt-1 shrink-0 ${c.dot}`} aria-hidden />

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <p className="text-[14px] text-indigo-950 leading-snug line-clamp-2 mb-1.5">
            {record.query}
          </p>
          <div className="flex flex-wrap items-center gap-1.5">
            {record.model && (
              <span className="text-[11px] bg-indigo-50 text-indigo-600 rounded px-1.5 py-0.5">
                {record.model}
              </span>
            )}
            {record.agent_id && (
              <span className="text-[11px] bg-indigo-50 text-indigo-600 rounded px-1.5 py-0.5">
                {record.agent_id}
              </span>
            )}
            {visibleFlags.map(f => <FlagChip key={f} metric={f} />)}
            {overflow > 0 && <FlagOverflow count={overflow} />}
          </div>
        </div>

        {/* Right: IQS badge + timestamp */}
        <div className="flex flex-col items-end gap-1 shrink-0">
          <IQSBadge score={record.iqs} size="sm" threshold={threshold} metricCount={record.iqs_metric_count ?? 5} />
          <span className="text-[11px] text-indigo-300 tabular-nums">
            {relativeTime(record.created_at)}
          </span>
        </div>
      </div>
    </div>
  );
};

// ─── Filter select ────────────────────────────────────────────────────
function FilterSelect({ label, value, onChange, options }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="h-9 pl-3 pr-8 text-sm border border-indigo-100 rounded-lg bg-white text-indigo-700
                   focus:outline-none focus:ring-2 focus:ring-indigo-500 appearance-none cursor-pointer"
        aria-label={label}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-indigo-400">
        ▾
      </span>
    </div>
  );
}

// ─── Bulk action bar ──────────────────────────────────────────────────
function BulkBar({ count, onExport, onMarkReviewed, onClear }) {
  if (count === 0) return null;
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50
                    flex items-center gap-3 px-5 py-3 rounded-xl
                    bg-indigo-950 text-white shadow-2xl animate-slide-up">
      <span className="text-sm font-medium">{count} selected</span>
      <div className="w-px h-4 bg-indigo-700" />
      <Button variant="ghost" size="sm" className="text-white hover:text-white hover:bg-indigo-800"
        icon={<Download size={14} />} onClick={onExport}>
        Export
      </Button>
      <Button variant="ghost" size="sm" className="text-white hover:text-white hover:bg-indigo-800"
        icon={<CheckCircle2 size={14} />} onClick={onMarkReviewed}>
        Mark reviewed
      </Button>
      <button
        onClick={onClear}
        className="text-indigo-400 hover:text-white transition-colors"
        aria-label="Clear selection"
      >
        <X size={16} />
      </button>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────
export function Queue({ avgIqs, threshold = 0.70 }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const addToast = useToast();
  const searchDebounce = useRef(null);

  const [records, setRecords] = useState([]);
  const [totalEver, setTotalEver] = useState(null); // null = unknown, 0 = first run
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(new Set());

  // Filter state - synced to URL params
  const search  = searchParams.get('q')      || '';
  const status  = searchParams.get('status') || 'all';
  const sort    = searchParams.get('sort')   || 'newest';
  const dateRange = searchParams.get('date') || 'all';

  const setParam = (key, val) => {
    setSearchParams(p => { const n = new URLSearchParams(p); n.set(key, val); return n; }, { replace: true });
  };

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = { limit: 50, sort };
      if (status !== 'all') params.status = status;
      if (search)           params.search = search;
      if (dateRange !== 'all') params.date_range = dateRange;
      const data = await api.getQueue(params);
      setRecords(data.records || []);
      if (totalEver === null) setTotalEver(data.total ?? data.records?.length ?? 0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [search, status, sort, dateRange]);

  useEffect(() => { load(); }, [load]);

  const handleSearch = (val) => {
    clearTimeout(searchDebounce.current);
    searchDebounce.current = setTimeout(() => setParam('q', val), 200);
  };

  const handleSelect = (id, checked) => {
    setSelected(s => { const n = new Set(s); checked ? n.add(id) : n.delete(id); return n; });
  };

  const pendingCount = records.filter(r => !r.reviewed_at).length;

  return (
    <div className="flex flex-col h-full page-enter">
      <TopBar title="Review queue" avgIqs={avgIqs} threshold={threshold} />

      {/* Filter bar */}
      <div className="bg-white border-b border-indigo-100 px-6 py-3 flex items-center gap-3 shrink-0">
        {/* Search */}
        <div className="relative flex-1 max-w-[280px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-indigo-300 pointer-events-none" />
          <input
            className="input pl-9 h-9 text-sm"
            placeholder="Search queries…"
            defaultValue={search}
            onChange={e => handleSearch(e.target.value)}
            aria-label="Search records"
          />
        </div>

        <FilterSelect
          label="Status"
          value={status}
          onChange={v => setParam('status', v)}
          options={[
            { value: 'all',  label: 'All status' },
            { value: 'pass', label: 'Pass' },
            { value: 'warn', label: 'Warn' },
            { value: 'fail', label: 'Fail' },
          ]}
        />

        <FilterSelect
          label="Date range"
          value={dateRange}
          onChange={v => setParam('date', v)}
          options={[
            { value: 'all',   label: 'All time' },
            { value: '1d',    label: 'Today' },
            { value: '7d',    label: 'Last 7 days' },
            { value: '30d',   label: 'Last 30 days' },
          ]}
        />

        <FilterSelect
          label="Sort"
          value={sort}
          onChange={v => setParam('sort', v)}
          options={[
            { value: 'newest',  label: 'Newest first' },
            { value: 'oldest',  label: 'Oldest first' },
            { value: 'iqs_asc', label: 'IQS ascending' },
            { value: 'iqs_desc',label: 'IQS descending' },
          ]}
        />

        <div className="ml-auto text-[12px] text-indigo-400 tabular-nums">
          {loading ? '…' : `${records.length} records`}
        </div>
      </div>

      {/* Record list */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {loading ? (
          Array.from({ length: 5 }, (_, i) => <SkeletonRow key={i} />)
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <p className="text-sm text-red-600">{error}</p>
            <Button variant="secondary" size="sm" onClick={load}>Retry</Button>
          </div>
        ) : records.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            {search || status !== 'all' ? (
              <>
                <Inbox size={32} className="text-indigo-200" />
                <p className="text-sm text-indigo-400">No records match your filters</p>
                <Button variant="ghost" size="sm" onClick={() => setSearchParams({})}>
                  Clear filters
                </Button>
              </>
            ) : totalEver === 0 ? (
              /* First-run state - user has never scored a record */
              <div className="max-w-sm">
                <div className="w-12 h-12 bg-indigo-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <Inbox size={24} className="text-indigo-400" />
                </div>
                <h3 className="text-sm font-semibold text-indigo-950 mb-2">No records yet</h3>
                <p className="text-xs text-indigo-400 mb-5 leading-relaxed">
                  Score your first LLM response to start the feedback loop.
                  Records appear here automatically as you score.
                </p>
                <pre className="w-full text-left bg-indigo-25 border border-indigo-100 rounded-lg
                                p-4 text-[11px] font-mono-score text-indigo-700 mb-4 leading-relaxed">
{`import scroot

auditor = scroot.Auditor()
result  = auditor.score(
    query    = "your question",
    response = "llm response",
    context  = "grounding docs"
)`}
                </pre>
                <a
                  href="https://github.com/sunnyguntuka/scroot"
                  target="_blank" rel="noopener noreferrer"
                  className="text-xs text-indigo-500 hover:text-indigo-700 underline underline-offset-2 transition-colors"
                >
                  View documentation →
                </a>
              </div>
            ) : (
              <>
                <CheckCircle2 size={32} className="text-green-300" />
                <p className="text-sm text-indigo-400">Queue is empty - all records reviewed</p>
              </>
            )}
          </div>
        ) : (
          <>
            {records.map(r => (
              <RecordRow
                key={r.id}
                record={r}
                selected={selected.has(r.id)}
                onSelect={handleSelect}
                threshold={threshold}
              />
            ))}
          </>
        )}
      </div>

      {/* Bulk action bar */}
      <BulkBar
        count={selected.size}
        onExport={() => addToast('Export started', 'info')}
        onMarkReviewed={() => {
          addToast(`Marked ${selected.size} records reviewed`, 'success');
          setSelected(new Set());
        }}
        onClear={() => setSelected(new Set())}
      />
    </div>
  );
}
