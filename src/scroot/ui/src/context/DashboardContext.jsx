import { createContext, useEffect, useState } from 'react';

export const DashboardContext = createContext(null);

export function DashboardProvider({ children }) {
  const [queueStats, setQueueStats] = useState(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [toasts, setToasts] = useState([]);

  const refresh = () =>
    fetch('/api/queue/stats')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data) {
          setQueueStats(data);
          setPendingCount(data.pending ?? 0);
        }
      })
      .catch(() => {});

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, []);

  const addToast = (msg, type = 'info') => {
    const id = Date.now();
    setToasts(t => [...t, { id, msg, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  };

  return (
    <DashboardContext.Provider value={{
      queueStats, pendingCount, toasts, addToast, refreshStats: refresh,
    }}>
      {children}
    </DashboardContext.Provider>
  );
}
