export function IQSBadge({ value }) {
  const color = value >= 0.8 ? 'var(--green)'
    : value >= 0.6 ? 'var(--yellow)'
    : 'var(--red)';
  return (
    <span style={{
      fontFamily: 'var(--font-mono)',
      fontSize: 13,
      fontWeight: 700,
      color,
      background: `${color}18`,
      border: `1px solid ${color}40`,
      padding: '2px 8px',
      borderRadius: 'var(--radius-sm)',
      whiteSpace: 'nowrap',
    }}>
      {typeof value === 'number' ? value.toFixed(2) : '—'}
    </span>
  );
}
