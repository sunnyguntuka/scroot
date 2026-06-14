export function TopBar({ title, actions }) {
  return (
    <header style={{
      height: 'var(--topbar-height)',
      background: 'var(--bg-base)',
      borderBottom: '1px solid var(--bg-border)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      position: 'sticky',
      top: 0,
      zIndex: 50,
    }}>
      <h1 style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 13,
        fontWeight: 700,
        color: 'var(--text-primary)',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        margin: 0,
      }}>
        {title}
      </h1>
      {actions && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontFamily: 'var(--font-sans)',
        }}>
          {actions}
        </div>
      )}
    </header>
  );
}
