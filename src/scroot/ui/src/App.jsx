import { useEffect, useState } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { ToastProvider } from './components/ui/Toast';
import { Queue }        from './pages/Queue';
import { RecordDetail } from './pages/RecordDetail';
import { Analytics }    from './pages/Analytics';
import { Export }       from './pages/Export';
import { Settings }     from './pages/Settings';
import { Pipeline }     from './pages/Pipeline';
import { FlaskConical } from 'lucide-react';
import { api, DEMO_MODE } from './api/client';

// Minimum width at which sidebar collapses to icon-only
const COLLAPSE_AT = 1280;

export default function App() {
  const location = useLocation();

  // Global state that cross-cuts all pages
  const [pendingCount,  setPendingCount]  = useState(0);
  const [avgIqs,        setAvgIqs]        = useState(null);
  const [threshold,     setThreshold]     = useState(0.70);
  const [hasCorrector,  setHasCorrector]  = useState(false);
  const [corrector,     setCorrector]     = useState(null);
  const [activeGuardrails, setActiveGuardrails] = useState(0);
  const [collapsed,     setCollapsed]     = useState(window.innerWidth < COLLAPSE_AT);

  // Viewport resize → sidebar collapse
  useEffect(() => {
    const handle = () => setCollapsed(window.innerWidth < COLLAPSE_AT);
    window.addEventListener('resize', handle);
    return () => window.removeEventListener('resize', handle);
  }, []);

  // Load health + settings on mount and every 30s
  useEffect(() => {
    const load = async () => {
      try {
        const [health, settings, guardrails] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getSettings().catch(() => null),
          api.getGuardrailStats().catch(() => null),
        ]);
        if (health) {
          setPendingCount(health.pending_count ?? 0);
          setAvgIqs(health.avg_iqs_today ?? health.avg_iqs ?? null);
        }
        if (settings) {
          setThreshold(settings.iqs_threshold ?? 0.70);
          const c = settings.corrector;
          const hasC = c ? c.mode !== 'disabled' : (
            settings.llm_corrector?.provider !== 'none' && !!settings.llm_corrector?.provider
          );
          setHasCorrector(hasC);
          if (c) setCorrector(c);
        }
        if (guardrails) {
          setActiveGuardrails(guardrails.active_guardrails ?? 0);
        }
      } catch {}
    };
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, []);

  const sidebarW = collapsed ? 48 : 220;

  return (
    <ToastProvider>
      <div className="flex h-screen overflow-hidden bg-indigo-50">
        {DEMO_MODE && (
          <div className="fixed top-0 left-0 right-0 z-[2000] flex items-center justify-center gap-2
                          bg-indigo-50 border-b border-indigo-100 text-indigo-500 text-[11px] py-1">
            <FlaskConical size={12} strokeWidth={2} />
            <span>
              Demo mode · sample data · remove
              <span className="font-mono-score mx-1">?demo</span>
              from the URL to connect to your backend
            </span>
          </div>
        )}
        <Sidebar pendingCount={pendingCount} collapsed={collapsed} activeGuardrails={activeGuardrails} />

        <main
          className="flex-1 flex flex-col overflow-hidden"
          style={{ marginLeft: sidebarW }}
        >
          <Routes>
            <Route path="/"           element={<Navigate to="/queue" replace />} />
            <Route path="/queue"      element={<Queue avgIqs={avgIqs} threshold={threshold} />} />
            <Route path="/queue/:id"  element={<RecordDetail threshold={threshold} hasLlmCorrector={hasCorrector} corrector={corrector} />} />
            <Route path="/analytics"  element={<Analytics avgIqs={avgIqs} threshold={threshold} />} />
            <Route path="/export"     element={<Export avgIqs={avgIqs} threshold={threshold} />} />
            <Route path="/settings"   element={<Settings avgIqs={avgIqs} threshold={threshold} />} />
            <Route path="/pipeline"   element={<Pipeline pendingCount={pendingCount} avgIqs={avgIqs} threshold={threshold} hasCorrector={hasCorrector} corrector={corrector} />} />
            <Route path="*"           element={<Navigate to="/queue" replace />} />
          </Routes>
        </main>
      </div>
    </ToastProvider>
  );
}
