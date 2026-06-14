import { useEffect, useState } from 'react';
import { BarChart2, TrendingUp, TrendingDown } from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';
import { TopBar } from '../components/layout/TopBar';
import { Button } from '../components/ui/Button';
import { IQSBadge } from '../components/ui/IQSBadge';
import { api } from '../api/client';
import { METRIC_COLORS, METRIC_LABELS, fmtIQS, getIQSStatus, STATUS_COLORS } from '../utils/iqs';

// ─── Summary metric card ──────────────────────────────────────────────
function SummaryCard({ label, value, delta, isIqs = false, className = '' }) {
  const isPositive = delta > 0;
  return (
    <div className={`card p-4 ${className}`}>
      <p className="section-label mb-2">{label}</p>
      <p className={`${isIqs ? 'font-mono-score' : 'font-sans'} font-semibold text-2xl text-indigo-950 leading-none mb-1.5`}>
        {isIqs ? fmtIQS(value) : (value ?? '—')}
      </p>
      {delta !== undefined && delta !== null && (
        <div className={`flex items-center gap-1 text-xs ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
          {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
          <span>{isPositive ? '+' : ''}{(delta * 100).toFixed(0)}% vs prior period</span>
        </div>
      )}
    </div>
  );
}

// ─── Chart tooltip ────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-3 py-2.5 shadow-card-hover">
      <p className="text-[11px] text-indigo-400 mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} className="text-sm font-mono-score" style={{ color: p.color || '#4F46E5' }}>
          {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </p>
      ))}
    </div>
  );
}

// ─── Skeleton placeholder ─────────────────────────────────────────────
function ChartSkeleton({ height = 200 }) {
  return <div className="skeleton rounded-lg" style={{ height }} />;
}

const PERIOD_OPTS = [
  { value: '7d',  label: '7D' },
  { value: '30d', label: '30D' },
  { value: '90d', label: '90D' },
  { value: 'all', label: 'All' },
];

export function Analytics({ avgIqs, threshold = 0.70 }) {
  const [period, setPeriod]   = useState('30d');
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getAnalytics(period)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [period]);

  // Distribution bar color
  const distColor = (bucket) => {
    const low = parseFloat(bucket);
    if (low >= threshold) return '#4F46E5';        // pass - indigo
    if (low >= threshold * 0.7) return '#D97706';  // warn - amber
    return '#DC2626';                              // fail - red
  };

  return (
    <div className="flex flex-col h-full page-enter">
      <TopBar
        title="Analytics"
        avgIqs={avgIqs}
        threshold={threshold}
        actions={
          <div className="flex gap-1 p-0.5 bg-indigo-50 rounded-lg">
            {PERIOD_OPTS.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setPeriod(value)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors
                  ${period === value
                    ? 'bg-white text-indigo-700 shadow-sm'
                    : 'text-indigo-400 hover:text-indigo-600'
                  }`}
              >
                {label}
              </button>
            ))}
          </div>
        }
      />

      <div className="flex-1 overflow-y-auto p-6 space-y-5">

        {/* Summary cards */}
        <div className="grid grid-cols-4 gap-4">
          <SummaryCard label="Total scored"    value={data?.total_scored} />
          <SummaryCard label="Avg IQS"         value={data?.avg_iqs} isIqs delta={data?.avg_iqs_delta} />
          <SummaryCard label="Pending review"  value={data?.pending_review ?? data?.pending_count} />
          <div className="card p-4">
            <p className="section-label mb-2">Pass / Warn / Fail</p>
            {loading ? (
              <div className="skeleton h-8 rounded" />
            ) : data ? (
              <div className="flex gap-2 items-end">
                <div className="text-center">
                  <div className="font-semibold text-green-700 text-lg leading-none">{data.pass_count ?? 0}</div>
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mt-0.5">Pass</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-amber-700 text-lg leading-none">{data.warn_count ?? 0}</div>
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mt-0.5">Warn</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-red-700 text-lg leading-none">{data.fail_count ?? 0}</div>
                  <div className="text-[10px] text-indigo-400 uppercase tracking-wide mt-0.5">Fail</div>
                </div>
              </div>
            ) : <div className="text-sm text-indigo-300">—</div>}
          </div>
        </div>

        {/* IQS trend chart */}
        <div className="card p-5">
          <p className="section-label mb-4">IQS trend</p>
          {loading ? <ChartSkeleton height={200} /> : !data?.iqs_trend?.length ? (
            <div className="flex flex-col items-center justify-center h-[200px] gap-2">
              <BarChart2 size={28} className="text-indigo-200" />
              <p className="text-sm text-indigo-400">No trend data yet</p>
              <p className="text-xs text-indigo-300">Score some records to see IQS over time</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={data.iqs_trend} margin={{ top: 4, right: 16, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="#E0E7FF" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#A5B4FC', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  tickLine={false} axisLine={false}
                />
                <YAxis
                  domain={[0, 1]} tickCount={6}
                  tick={{ fill: '#A5B4FC', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                  tickLine={false} axisLine={false}
                  tickFormatter={v => v.toFixed(1)}
                />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine
                  y={threshold} stroke="#C7D2FE" strokeDasharray="4 4" strokeWidth={1.5}
                  label={{ value: 'threshold', position: 'right', fontSize: 10, fill: '#A5B4FC' }}
                />
                <Line
                  type="monotone" dataKey="avg_iqs" name="Avg IQS"
                  stroke="#4F46E5" strokeWidth={2}
                  dot={false} activeDot={{ r: 4, fill: '#4F46E5' }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Two-column: Flag frequency + IQS distribution */}
        <div className="grid grid-cols-2 gap-5">
          {/* Flag frequency */}
          <div className="card p-5">
            <p className="section-label mb-4">Flag frequency</p>
            {loading ? <ChartSkeleton height={180} /> : (() => {
              const flagData = data?.flag_frequency
                ? Object.entries(data.flag_frequency)
                    .map(([metric, count]) => ({ metric, count, label: METRIC_LABELS[metric] || metric }))
                    .sort((a, b) => b.count - a.count)
                : [];
              const hasFlags = flagData.some(d => d.count > 0);
              if (!hasFlags) return (
                <div className="flex flex-col items-center justify-center h-[180px] gap-2">
                  <BarChart2 size={24} className="text-indigo-200" />
                  <p className="text-sm text-indigo-400">No flags recorded yet</p>
                  <p className="text-xs text-indigo-300">Flags appear when metrics fall below threshold</p>
                </div>
              );
              return (
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={flagData} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 8 }}>
                    <CartesianGrid stroke="#E0E7FF" horizontal={false} />
                    <XAxis type="number" tick={{ fill: '#A5B4FC', fontSize: 10 }} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="label"
                      tick={{ fill: '#818CF8', fontSize: 11 }} tickLine={false} axisLine={false} width={90} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="count" name="Flags" radius={[0, 4, 4, 0]}>
                      {flagData.map(({ metric }) => (
                        <Cell key={metric} fill={METRIC_COLORS[metric] || '#4F46E5'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              );
            })()}
          </div>

          {/* IQS distribution */}
          <div className="card p-5">
            <p className="section-label mb-4">Score distribution</p>
            {loading ? <ChartSkeleton height={180} /> : !data?.iqs_distribution?.length ? (
              <div className="flex flex-col items-center justify-center h-[180px] gap-2">
                <BarChart2 size={24} className="text-indigo-200" />
                <p className="text-sm text-indigo-400">No distribution data yet</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={data.iqs_distribution} margin={{ top: 0, right: 8, bottom: 0, left: -8 }}>
                  <CartesianGrid stroke="#E0E7FF" vertical={false} />
                  <XAxis dataKey="bucket"
                    tick={{ fill: '#A5B4FC', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                    tickLine={false} axisLine={false}
                  />
                  <YAxis tick={{ fill: '#A5B4FC', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="count" name="Records" radius={[4, 4, 0, 0]}>
                    {data.iqs_distribution.map(b => (
                      <Cell key={b.bucket} fill={distColor(b.bucket)} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Per-agent table - only when ≥2 agents */}
        {(data?.per_agent ?? data?.agents)?.length >= 2 && (
          <div className="card p-5">
            <p className="section-label mb-4">Per-agent breakdown</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-indigo-50">
                  {['Agent', 'Count', 'Avg IQS'].map(h => (
                    <th key={h} className="text-left text-[11px] uppercase tracking-wider text-indigo-400 pb-2 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-indigo-50">
                {[...(data.per_agent ?? data.agents ?? [])]
                  .sort((a, b) => a.avg_iqs - b.avg_iqs)
                  .map(agent => {
                    const id = agent.agent_id ?? agent.agent;
                    const st = getIQSStatus(agent.avg_iqs, threshold);
                    const col = STATUS_COLORS[st];
                    return (
                      <tr key={id} className="hover:bg-indigo-25 transition-colors">
                        <td className="py-2.5 text-indigo-700 font-medium">{id}</td>
                        <td className="py-2.5 text-indigo-500 font-mono-score tabular-nums">{agent.count}</td>
                        <td className="py-2.5">
                          <IQSBadge score={agent.avg_iqs} size="sm" threshold={threshold} />
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        )}

        {/* Empty state */}
        {!loading && !error && data?.total_scored === 0 && (
          <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
            <BarChart2 size={32} className="text-indigo-200" />
            <p className="text-sm text-indigo-400">
              No scored records yet. Run scroot.score() to populate the dashboard.
            </p>
            <code className="text-xs text-indigo-300 mt-1 block font-mono-score">
              auditor.score(query, response, context)
            </code>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center gap-3 py-12">
            <p className="text-sm text-red-600">{error}</p>
            <Button variant="secondary" size="sm" onClick={() => setPeriod(p => p)}>Retry</Button>
          </div>
        )}
      </div>
    </div>
  );
}
