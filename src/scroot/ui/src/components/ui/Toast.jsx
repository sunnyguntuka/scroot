import { createContext, useCallback, useContext, useState } from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

const ToastCtx = createContext(null);

const ICON = {
  success: <CheckCircle2 size={16} className="text-green-600 shrink-0" />,
  error:   <AlertCircle  size={16} className="text-red-600 shrink-0" />,
  info:    <Info         size={16} className="text-indigo-500 shrink-0" />,
};

const STYLE = {
  success: 'border-green-200 bg-green-50',
  error:   'border-red-200   bg-red-50',
  info:    'border-indigo-100 bg-white',
};

function Toast({ id, message, type, onDismiss }) {
  return (
    <div
      className={`flex items-start gap-3 w-[320px] max-w-[calc(100vw-32px)]
        rounded-xl border px-4 py-3 shadow-lg animate-slide-up ${STYLE[type]}`}
      role="alert"
      aria-live="polite"
    >
      {ICON[type]}
      <span className="flex-1 text-sm text-indigo-950 leading-snug">{message}</span>
      <button
        onClick={() => onDismiss(id)}
        className="shrink-0 text-indigo-300 hover:text-indigo-600 transition-colors -mt-0.5"
        aria-label="Dismiss"
      >
        <X size={14} />
      </button>
    </div>
  );
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now() + Math.random();
    setToasts(t => [...t, { id, message, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3200);
  }, []);

  const dismiss = useCallback((id) => {
    setToasts(t => t.filter(x => x.id !== id));
  }, []);

  return (
    <ToastCtx.Provider value={{ addToast }}>
      {children}
      {/* Stack - bottom-right, newest on top */}
      <div className="fixed bottom-6 right-6 flex flex-col-reverse gap-2 z-[2000]">
        {toasts.map(t => (
          <Toast key={t.id} {...t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx.addToast;
}

// Legacy compat - do not use in new code
export function ToastContainer() { return null; }
