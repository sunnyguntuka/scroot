import { useEffect, useRef, useState } from 'react';

function useCountUp(target, duration = 600) {
  const [val, setVal] = useState(0);
  const start = useRef(null);

  useEffect(() => {
    if (target === null || target === undefined) return;
    const numTarget = typeof target === 'number' ? target : parseFloat(target) || 0;
    start.current = null;

    const step = (ts) => {
      if (!start.current) start.current = ts;
      const progress = Math.min((ts - start.current) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setVal(numTarget * eased);
      if (progress < 1) requestAnimationFrame(step);
      else setVal(numTarget);
    };
    requestAnimationFrame(step);
  }, [target, duration]);

  return val;
}

export function MetricCard({ value, label, format = 'int', suffix = '' }) {
  const animated = useCountUp(typeof value === 'number' ? value : 0);
  const display = format === 'float'
    ? animated.toFixed(2)
    : format === 'hours'
    ? animated.toFixed(1)
    : Math.round(animated).toString();

  return (
    <div className="stat-pill count-up">
      <div className="value">{display}{suffix}</div>
      <div className="label">{label}</div>
    </div>
  );
}
