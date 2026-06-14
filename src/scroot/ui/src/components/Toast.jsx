import { useContext } from 'react';
import { DashboardContext } from '../context/DashboardContext';

const BORDER_COLORS = {
  success: 'var(--green)',
  error:   'var(--red)',
  info:    'var(--accent)',
  warning: 'var(--yellow)',
};

function ToastItem({ id, msg, type }) {
  const color = BORDER_COLORS[type] || BORDER_COLORS.info;
  return (
    <div style={{
      width: 320,
      background: 'var(--bg-elevated)',
      border: '1px solid var(--bg-border)',
      borderLeft: `4px solid ${color}`,
      borderRadius: 'var(--radius-md)',
      padding: '12px 16px',
      fontFamily: 'var(--font-sans)',
      fontSize: 13,
      color: 'var(--text-primary)',
      boxShadow: 'var(--shadow-card)',
      animation: 'toastIn 200ms ease-out',
    }}>
      {msg}
    </div>
  );
}

export function ToastContainer() {
  const ctx = useContext(DashboardContext);
  if (!ctx) return null;
  return (
    <div className="toast-container">
      <style>{`
        @keyframes toastIn {
          from { opacity: 0; transform: translateX(120px); }
          to   { opacity: 1; transform: translateX(0); }
        }
      `}</style>
      {ctx.toasts.map(t => (
        <ToastItem key={t.id} {...t} />
      ))}
    </div>
  );
}
