import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronRight, Download, FolderOpen, Info, ShieldCheck,
  Trash2, X, AlertTriangle, CheckCircle, Loader2,
} from 'lucide-react';
import { TopBar } from '../components/layout/TopBar';
import { Button } from '../components/ui/Button';
import { useToast } from '../components/ui/Toast';
import { api } from '../api/client';
import { METRIC_LABELS, ALL_METRICS } from '../utils/iqs';

// ─── Helpers ──────────────────────────────────────────────────────────

function detectProviderName(apiKey, baseUrl) {
  if (baseUrl) {
    if (baseUrl.includes('groq')) return 'Groq';
    if (baseUrl.includes('openrouter')) return 'OpenRouter';
    if (baseUrl.includes('anthropic')) return 'Anthropic';
    return 'Custom';
  }
  if (!apiKey || apiKey.length < 4) return null;
  if (apiKey.startsWith('sk-ant-')) return 'Anthropic';
  if (apiKey.startsWith('AIza')) return 'Google Gemini';
  if (apiKey.startsWith('sk-')) return 'OpenAI';
  return 'OpenRouter';
}

// ─── Shared sub-components ────────────────────────────────────────────

function SectionCard({ title, children, id }) {
  return (
    <div className="card p-6" id={id}>
      <h2 className="text-[15px] font-semibold text-indigo-950 mb-5">{title}</h2>
      {children}
    </div>
  );
}

function FieldRow({ label, hint, children }) {
  return (
    <div className="flex flex-col gap-1.5 mb-5">
      <label className="text-sm font-medium text-indigo-700">{label}</label>
      {children}
      {hint && <p className="text-xs text-indigo-400">{hint}</p>}
    </div>
  );
}

function ConfirmModal({ title, description, confirmLabel, onConfirm, onCancel }) {
  return (
    <div className="fixed inset-0 bg-indigo-950/50 flex items-center justify-center z-[1000] animate-fade-in">
      <div className="card p-6 w-[400px] max-w-[calc(100vw-32px)] shadow-2xl animate-slide-up">
        <h3 className="text-[15px] font-semibold text-indigo-950 mb-2">{title}</h3>
        <p className="text-sm text-indigo-500 mb-5">{description}</p>
        <div className="flex justify-end gap-2">
          <Button variant="secondary" size="sm" onClick={onCancel}>Cancel</Button>
          <Button variant="danger" size="sm" onClick={onConfirm}>{confirmLabel}</Button>
        </div>
      </div>
    </div>
  );
}

// ─── Mode selector card ───────────────────────────────────────────────

function ModeCard({ value, selected, title, subtitle, tagline, onChange }) {
  return (
    <label
      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors
        ${selected
          ? 'border-indigo-400 bg-indigo-50'
          : 'border-indigo-100 hover:border-indigo-200'
        }`}
    >
      <input
        type="radio" name="corrector_mode" value={value}
        checked={selected}
        onChange={() => onChange(value)}
        className="accent-indigo-600 mt-0.5"
      />
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-semibold text-indigo-900">{title}</span>
        <span className="text-xs text-indigo-500">{subtitle}</span>
        {tagline && <span className="text-xs text-indigo-400 mt-0.5">{tagline}</span>}
      </div>
    </label>
  );
}

// ─── NLI guarantee strip ──────────────────────────────────────────────

function NliStrip() {
  return (
    <div className="flex items-start gap-2 mt-5 p-3 rounded-lg bg-indigo-50 border border-indigo-100">
      <ShieldCheck size={13} className="text-indigo-400 shrink-0 mt-0.5" />
      <p className="text-[11px] text-indigo-400 leading-relaxed">
        NLI re-scores every correction before any commit - regardless of which provider generated it.
        The LLM is the intern. NLI is the senior reviewer.
      </p>
    </div>
  );
}

// ─── First-run banner ─────────────────────────────────────────────────

function FirstRunBanner({ onSelectLocal, onSelectApi, onDismiss }) {
  return (
    <div className="mb-5 rounded-lg border border-indigo-200 bg-indigo-50 p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-indigo-900 mb-1">
            Enable the LLM corrector to unlock pipeline automation
          </p>
          <p className="text-xs text-indigo-500 mb-3">
            Local LLM: free, private, runs on your machine.{' '}
            API: faster, requires an API key.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button variant="primary" size="sm" onClick={onSelectLocal}>
              Set up Local LLM
            </Button>
            <Button variant="secondary" size="sm" onClick={onSelectApi}>
              Use API instead
            </Button>
          </div>
        </div>
        <button onClick={onDismiss} className="text-indigo-400 hover:text-indigo-600 shrink-0">
          <X size={14} />
        </button>
      </div>
    </div>
  );
}

// ─── Model download card ──────────────────────────────────────────────

function ModelCard({ model, selected, onSelect, onDownload, onDelete, downloadState }) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const ds = downloadState || {};
  const status = model.downloaded ? 'ready' : (ds.status || 'idle');

  return (
    <div
      className={`rounded-lg border p-3 transition-colors ${
        selected ? 'border-indigo-400 bg-indigo-50' : 'border-indigo-100'
      }`}
    >
      <div className="flex items-start gap-2 mb-2">
        <input
          type="radio" name="local_model"
          checked={selected}
          onChange={onSelect}
          className="accent-indigo-600 mt-0.5"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-indigo-900">{model.name}</span>
            {model.is_default && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-600 font-medium">default</span>
            )}
            <span className="text-xs text-indigo-400 ml-auto font-mono-score tabular-nums">{model.size_gb} GB</span>
          </div>
          <p className="text-xs text-indigo-400 mt-0.5">{model.description}</p>
          <p className="text-[10px] text-indigo-300 mt-0.5">{model.license} license</p>
        </div>
      </div>

      {/* Status row */}
      {status === 'ready' && (
        <div className="flex items-center gap-2 pl-5">
          <CheckCircle size={12} className="text-green-500" />
          <span className="text-xs text-green-600">Downloaded</span>
          <div className="flex-1" />
          {confirmDelete ? (
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-red-600">Delete {model.name} ({model.size_gb} GB)?</span>
              <button
                onClick={() => setConfirmDelete(false)}
                className="text-xs text-indigo-500 hover:text-indigo-700 px-1.5 py-0.5 border border-indigo-200 rounded"
              >Cancel</button>
              <button
                onClick={() => { setConfirmDelete(false); onDelete(); }}
                className="text-xs text-white bg-red-600 hover:bg-red-700 px-1.5 py-0.5 rounded"
              >Delete</button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmDelete(true)}
              className="text-xs text-red-500 hover:text-red-700 transition-colors"
            >Delete model</button>
          )}
        </div>
      )}

      {status === 'idle' && (
        <div className="pl-5">
          <button
            onClick={onDownload}
            className="flex items-center gap-1.5 text-xs font-medium text-white bg-indigo-600 hover:bg-indigo-700
                       px-2.5 py-1 rounded transition-colors"
          >
            <Download size={11} />
            Download {model.size_gb} GB
          </button>
        </div>
      )}

      {(status === 'pending' || status === 'downloading') && (
        <div className="pl-5 space-y-1">
          <div className="flex items-center justify-between text-xs text-indigo-500">
            <span>
              {ds.progress_pct > 0
                ? `${((ds.progress_bytes || 0) / 1e9).toFixed(1)} / ${model.size_gb} GB`
                : 'Starting download…'}
            </span>
            {ds.eta_seconds > 0 && (
              <span>~{Math.ceil(ds.eta_seconds / 60)}m remaining</span>
            )}
          </div>
          <div className="h-1.5 rounded-full bg-indigo-100 overflow-hidden">
            <div
              className="h-full bg-indigo-600 transition-all"
              style={{ width: `${ds.progress_pct || 0}%` }}
            />
          </div>
          <p className="text-[10px] text-indigo-400">{ds.progress_pct || 0}%</p>
        </div>
      )}

      {status === 'failed' && (
        <div className="pl-5 flex items-center gap-2">
          <AlertTriangle size={12} className="text-amber-500" />
          <span className="text-xs text-amber-600">Download failed: {ds.error}</span>
          <button onClick={onDownload} className="text-xs text-indigo-600 hover:underline">Retry</button>
        </div>
      )}
    </div>
  );
}

// ─── Local LLM panel ──────────────────────────────────────────────────

function LocalLLMPanel({ corrector, onModelChange, onSave }) {
  const addToast = useToast();
  const [selectedModel, setSelectedModel] = useState(corrector?.local?.model_id || 'phi4-mini');
  const [models, setModels] = useState([]);
  const [runtime, setRuntime] = useState(null);
  const [downloadStates, setDownloadStates] = useState({});
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const pollRef = useRef({});

  useEffect(() => {
    api.getCorrectorModels().then(r => setModels(r.models || [])).catch(() => {});
    api.getCorrectorRuntime().then(setRuntime).catch(() => {});
  }, []);

  const pollStatus = (modelId) => {
    if (pollRef.current[modelId]) return;
    const interval = setInterval(async () => {
      try {
        const s = await api.getModelDownloadStatus(modelId);
        setDownloadStates(prev => ({ ...prev, [modelId]: s }));
        if (s.status === 'ready' || s.status === 'failed') {
          clearInterval(interval);
          delete pollRef.current[modelId];
          if (s.status === 'ready') {
            setModels(prev => prev.map(m => m.id === modelId ? { ...m, downloaded: true } : m));
          }
        }
      } catch {
        clearInterval(interval);
        delete pollRef.current[modelId];
      }
    }, 2000);
    pollRef.current[modelId] = interval;
  };

  useEffect(() => () => {
    Object.values(pollRef.current).forEach(clearInterval);
  }, []);

  const handleDownload = async (modelId) => {
    try {
      await api.downloadModel(modelId);
      setDownloadStates(prev => ({ ...prev, [modelId]: { status: 'pending', progress_pct: 0 } }));
      pollStatus(modelId);
    } catch (e) {
      addToast('Download failed: ' + e.message, 'error');
    }
  };

  const handleDelete = async (modelId) => {
    try {
      await api.deleteModel(modelId);
      setModels(prev => prev.map(m => m.id === modelId ? { ...m, downloaded: false } : m));
      setDownloadStates(prev => { const n = { ...prev }; delete n[modelId]; return n; });
      addToast('Model deleted', 'info');
    } catch (e) {
      addToast('Delete failed: ' + e.message, 'error');
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await api.testCorrector();
      setTestResult(r);
    } catch (e) {
      setTestResult({ error: e.message });
    } finally {
      setTesting(false);
    }
  };

  const handleModelSelect = (modelId) => {
    setSelectedModel(modelId);
    onModelChange(modelId);
    onSave({ corrector: { mode: 'local', local: { model_id: modelId } } });
  };

  return (
    <div className="mt-4 space-y-4">
      <p className="section-label text-xs text-indigo-400 uppercase tracking-wider font-semibold">Model</p>

      {runtime && !runtime.llama_cpp_installed && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle size={14} className="text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-amber-900">Inference engine not installed</p>
              <p className="text-xs text-amber-700 mt-1">
                Download a model and run this in your terminal:
              </p>
              <code className="block text-xs font-mono bg-white border border-amber-100 rounded px-2 py-1 mt-1.5">
                pip install &apos;scroot[local]&apos;
              </code>
              <p className="text-xs text-amber-600 mt-1.5">Then restart the server:</p>
              <code className="block text-xs font-mono bg-white border border-amber-100 rounded px-2 py-1 mt-1.5">
                scroot serve
              </code>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-2">
        {models.map(model => (
          <ModelCard
            key={model.id}
            model={model}
            selected={selectedModel === model.id}
            onSelect={() => handleModelSelect(model.id)}
            onDownload={() => handleDownload(model.id)}
            onDelete={() => handleDelete(model.id)}
            downloadState={downloadStates[model.id]}
          />
        ))}
      </div>

      <Button
        variant="secondary"
        size="sm"
        loading={testing}
        icon={testing ? <Loader2 size={12} className="animate-spin" /> : null}
        onClick={handleTest}
      >
        Test local LLM
      </Button>

      {testResult && (
        <div className={`rounded-lg border p-3 text-xs ${
          testResult.error
            ? 'border-red-200 bg-red-50'
            : 'border-green-200 bg-green-50'
        }`}>
          {testResult.error ? (
            <div className="flex items-start gap-1.5">
              <AlertTriangle size={12} className="text-red-500 shrink-0 mt-0.5" />
              <div>
                <p className="text-red-700 font-medium">Test failed</p>
                <p className="text-red-600 mt-0.5">{testResult.error}</p>
                <p className="text-red-400 mt-0.5">Check that the model is downloaded.</p>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-1.5">
              <CheckCircle size={12} className="text-green-500 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <p className="text-green-700 font-medium">Local LLM is working</p>
                <p className="text-green-600">Model: {testResult.model}</p>
                <p className="text-green-600">Latency: {(testResult.latency_ms / 1000).toFixed(1)}s</p>
                {testResult.tok_per_sec && (
                  <p className="text-green-600">Speed: {testResult.tok_per_sec} tok/s</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── API panel ────────────────────────────────────────────────────────
//
// Why three fields when "you just need an API key"? The key only authenticates
// (the auth header). A request still needs a destination (base_url) and a model.
// - base_url is auto-detected from the key prefix, so it stays optional/advanced.
// - model is a MANDATORY request param and is NOT derivable from the key (one
//   Anthropic key calls Opus/Sonnet/Haiku - which one is a user decision), so it
//   cannot be removed, only defaulted.
// Full rationale: src/scroot/corrector/api.py module docstring.

function APIPanel({ corrector, onSave }) {
  const addToast = useToast();
  const [apiKey, setApiKey]   = useState('');
  const [baseUrl, setBaseUrl] = useState(corrector?.api?.base_url || '');
  const [model, setModel]     = useState(corrector?.api?.model || 'gpt-4o-mini');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const detectedProvider = detectProviderName(apiKey, baseUrl);

  const persistApi = (updates = {}) => {
    const patch = {
      corrector: {
        mode: 'api',
        api: {
          api_key: apiKey,
          base_url: baseUrl,
          model,
          ...updates,
        },
      },
    };
    onSave(patch);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await api.testCorrector();
      setTestResult(r);
    } catch (e) {
      setTestResult({ error: e.message });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="mt-4 space-y-4">
      <FieldRow
        label="API key"
        hint="Works with OpenAI, Anthropic, Google Gemini, and OpenRouter. Provider is detected automatically from your key."
      >
        <div className="space-y-1.5">
          <input
            type="password"
            className="input h-9 text-sm font-mono-score tracking-widest w-full"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            onBlur={() => persistApi({ api_key: apiKey })}
            placeholder="sk-••••••••••••••••"
            autoComplete="off"
          />
          {detectedProvider && apiKey.length > 10 && (
            <span className="inline-block text-xs px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100">
              Detected: {detectedProvider}
            </span>
          )}
        </div>
      </FieldRow>

      <FieldRow label="Model" hint="e.g. gpt-4o-mini, claude-haiku-4-5, gemini-2.0-flash">
        <input
          type="text"
          className="input h-9 text-sm w-full"
          value={model}
          onChange={e => setModel(e.target.value)}
          onBlur={() => persistApi({ model })}
          placeholder="gpt-4o-mini"
        />
      </FieldRow>

      <button
        onClick={() => setShowAdvanced(v => !v)}
        className="flex items-center gap-1 text-xs text-indigo-500 hover:text-indigo-700 transition-colors"
      >
        <ChevronRight size={12} className={`transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
        Advanced
      </button>

      {showAdvanced && (
        <FieldRow label="Base URL" hint="Leave blank for auto-detected provider. Set for Groq, OpenRouter, or custom endpoints.">
          <input
            type="text"
            className="input h-9 text-sm font-mono-score w-full"
            value={baseUrl}
            onChange={e => setBaseUrl(e.target.value)}
            onBlur={() => persistApi({ base_url: baseUrl })}
            placeholder="https://openrouter.ai/api/v1"
          />
        </FieldRow>
      )}

      <Button
        variant="secondary"
        size="sm"
        loading={testing}
        onClick={handleTest}
      >
        Test connection
      </Button>

      {testResult && (
        <div className={`rounded-lg border p-3 text-xs ${
          testResult.error ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'
        }`}>
          {testResult.error ? (
            <p className="text-red-700">{testResult.error}</p>
          ) : (
            <div className="flex items-start gap-1.5">
              <CheckCircle size={12} className="text-green-500 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <p className="text-green-700 font-medium">Connection OK</p>
                <p className="text-green-600">Latency: {testResult.latency_ms}ms</p>
                {testResult.model && <p className="text-green-600">Model: {testResult.model}</p>}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main Settings page ───────────────────────────────────────────────

export function Settings({ avgIqs, threshold: thresholdProp = 0.70 }) {
  const addToast = useToast();

  const [settings, setSettings]   = useState(null);
  const [loading, setLoading]     = useState(true);
  const [saving, setSaving]       = useState(false);
  const [showClearModal, setShowClearModal] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(
    () => localStorage.getItem('corrector_banner_dismissed') === '1'
  );

  const [threshold, setThreshold] = useState(0.70);
  const [weights, setWeights]     = useState({
    groundedness: 0.35, completeness: 0.25, relevance: 0.20, consistency: 0.15, confidence: 0.05,
  });
  const [correctorMode, setCorrectorMode] = useState('disabled');
  const [corrector, setCorrector] = useState(null);

  useEffect(() => {
    api.getSettings()
      .then(s => {
        setSettings(s);
        setThreshold(s.iqs_threshold ?? 0.70);
        setWeights(s.metric_weights ?? {
          groundedness: 0.35, completeness: 0.25, relevance: 0.20, consistency: 0.15, confidence: 0.05,
        });
        const c = s.corrector || {};
        setCorrectorMode(c.mode || 'disabled');
        setCorrector(c);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const save = useCallback(async (patch) => {
    setSaving(true);
    try {
      await api.updateSettings(patch);
      addToast('Settings saved', 'success');
    } catch {
      addToast('Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  }, [addToast]);

  const setWeight = (metric, raw) => {
    const val = parseFloat(raw);
    const others = ALL_METRICS.filter(m => m !== metric);
    const oldSum = others.reduce((s, m) => s + weights[m], 0);
    const remaining = Math.max(0, 1 - val);
    const newWeights = { ...weights, [metric]: val };
    if (oldSum > 0) {
      const ratio = remaining / oldSum;
      others.forEach(m => { newWeights[m] = +(weights[m] * ratio).toFixed(4); });
    } else {
      const each = +(remaining / others.length).toFixed(4);
      others.forEach(m => { newWeights[m] = each; });
    }
    setWeights(newWeights);
  };

  const weightSum = Object.values(weights).reduce((s, v) => s + v, 0);

  const handleModeChange = (mode) => {
    setCorrectorMode(mode);
    save({ corrector: { mode } });
  };

  const handleDismissBanner = () => {
    localStorage.setItem('corrector_banner_dismissed', '1');
    setBannerDismissed(true);
  };

  const handleClearRecords = async () => {
    try {
      await api.updateSettings({ clear_all_records: true });
      addToast('All records cleared', 'info');
    } catch {
      addToast('Failed to clear records', 'error');
    }
    setShowClearModal(false);
  };

  const showBanner = !bannerDismissed && correctorMode === 'disabled';

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <TopBar title="Settings" avgIqs={avgIqs} threshold={thresholdProp} />
        <div className="p-6 space-y-4">
          {[1, 2, 3].map(i => <div key={i} className="skeleton h-40 rounded-card" />)}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full page-enter">
      <TopBar title="Settings" avgIqs={avgIqs} threshold={thresholdProp} />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl space-y-5">

          {/* Section: Scoring */}
          <SectionCard title="Scoring">
            <FieldRow label="IQS threshold" hint="Below this score, records are flagged for review.">
              <div className="flex items-center gap-3">
                <input
                  type="range" min={0.50} max={0.95} step={0.01}
                  value={threshold}
                  onChange={e => setThreshold(parseFloat(e.target.value))}
                  onMouseUp={() => save({ iqs_threshold: threshold })}
                  className="flex-1 accent-indigo-600"
                />
                <span className="text-sm font-mono-score text-indigo-700 w-10 text-right tabular-nums">
                  {threshold.toFixed(2)}
                </span>
              </div>
            </FieldRow>

            <div className="mb-1">
              <div className="flex items-center justify-between mb-3">
                <p className="section-label mb-0">Metric weights</p>
                <span className={`text-xs font-mono-score ${Math.abs(weightSum - 1) < 0.001 ? 'text-green-600' : 'text-amber-600'}`}>
                  Total: {weightSum.toFixed(2)} {Math.abs(weightSum - 1) < 0.001 ? '✓' : '⚠'}
                </span>
              </div>
              <p className="text-xs text-indigo-400 mb-4">Must sum to 1.00 - sliders auto-normalize.</p>
              <div className="space-y-3">
                {ALL_METRICS.map(m => (
                  <div key={m} className="flex items-center gap-3">
                    <span className="w-28 shrink-0 text-sm text-indigo-700">{METRIC_LABELS[m]}</span>
                    <input
                      type="range" min={0} max={1} step={0.01}
                      value={weights[m]}
                      onChange={e => setWeight(m, e.target.value)}
                      onMouseUp={() => save({ metric_weights: weights })}
                      className="flex-1 accent-indigo-600"
                    />
                    <span className="text-sm font-mono-score text-indigo-600 w-10 text-right tabular-nums">
                      {weights[m].toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t border-indigo-50 mt-4 pt-4">
              <Button
                variant="ghost" size="sm"
                onClick={() => {
                  const def = { groundedness: 0.35, completeness: 0.25, relevance: 0.20, consistency: 0.15, confidence: 0.05 };
                  setWeights(def);
                  setThreshold(0.70);
                  save({ metric_weights: def, iqs_threshold: 0.70 });
                }}
              >
                Reset to defaults
              </Button>
            </div>
          </SectionCard>

          {/* Section: LLM Corrector */}
          <SectionCard title="LLM corrector" id="corrector">
            {showBanner && (
              <FirstRunBanner
                onSelectLocal={() => handleModeChange('local')}
                onSelectApi={() => handleModeChange('api')}
                onDismiss={handleDismissBanner}
              />
            )}

            <div className="space-y-2">
              <ModeCard
                value="disabled"
                selected={correctorMode === 'disabled'}
                title="Disabled"
                subtitle="No LLM correction. NLI scoring only."
                onChange={handleModeChange}
              />
              <ModeCard
                value="local"
                selected={correctorMode === 'local'}
                title="Local LLM"
                subtitle="Runs on your machine. Data never leaves your environment."
                tagline="No API key. No cost per correction."
                onChange={handleModeChange}
              />
              <ModeCard
                value="api"
                selected={correctorMode === 'api'}
                title="LLM via API"
                subtitle="Send corrections to an external LLM provider. Requires an API key."
                tagline="Correction text is sent to your chosen provider."
                onChange={handleModeChange}
              />
            </div>

            {correctorMode === 'local' && (
              <LocalLLMPanel
                corrector={corrector}
                onModelChange={(modelId) => setCorrector(c => ({ ...c, local: { ...c?.local, model_id: modelId } }))}
                onSave={save}
              />
            )}

            {correctorMode === 'api' && (
              <APIPanel
                corrector={corrector}
                onSave={save}
              />
            )}

            <NliStrip />
          </SectionCard>

          {/* Section: Storage */}
          <SectionCard title="Storage">
            <FieldRow label="Feedback store path">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  className="input h-9 text-sm font-mono-score text-indigo-500 bg-indigo-25"
                  value={settings?.store_path || '~/.scroot/feedback.jsonl'}
                  readOnly
                />
                <Button
                  variant="secondary" size="sm"
                  icon={<FolderOpen size={14} />}
                  onClick={() => addToast('Open in Explorer not supported in browser context', 'info')}
                >
                  Open
                </Button>
              </div>
            </FieldRow>

            <div className="flex items-center gap-6 text-sm text-indigo-500 mb-5">
              <span>Records: <span className="font-mono-score text-indigo-700">{settings?.record_count ?? '—'}</span></span>
              <span>Store size: <span className="font-mono-score text-indigo-700">{settings?.store_size ?? '—'}</span></span>
            </div>

            <Button
              variant="danger-outline" size="sm"
              icon={<Trash2 size={14} />}
              onClick={() => setShowClearModal(true)}
            >
              Clear all records…
            </Button>
          </SectionCard>
        </div>
      </div>

      {showClearModal && (
        <ConfirmModal
          title="Clear all records?"
          description={`This will permanently delete all ${settings?.record_count ?? ''} records from the feedback store. This cannot be undone.`}
          confirmLabel="Delete all records"
          onConfirm={handleClearRecords}
          onCancel={() => setShowClearModal(false)}
        />
      )}
    </div>
  );
}
