export function EmptyState({ icon = '◇', title, subtitle, action }) {
  return (
    <div style={{ textAlign: 'center', padding: '64px 32px' }}>
      <div style={{ fontSize: 32, marginBottom: 16, color: 'var(--text-muted)' }}>
        {icon}
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 16,
        color: 'var(--text-primary)',
        marginBottom: 8,
      }}>
        {title}
      </div>
      <div style={{
        fontSize: 14,
        color: 'var(--text-secondary)',
        marginBottom: 24,
      }}>
        {subtitle}
      </div>
      {action}
    </div>
  );
}
