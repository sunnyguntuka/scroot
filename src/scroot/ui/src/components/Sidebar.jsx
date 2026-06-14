import { useContext, useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { DashboardContext } from '../context/DashboardContext';

const NAV = [
  { to: '/queue',     label: 'Queue',     icon: '⬡' },
  { to: '/analytics', label: 'Analytics', icon: '◈' },
  { to: '/export',    label: 'Export',    icon: '↗' },
  { to: '/settings',  label: 'Settings',  icon: '◎' },
];

export function Sidebar() {
  const ctx = useContext(DashboardContext);
  const pending = ctx?.pendingCount ?? 0;
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.ok && setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  return (
    <aside style={{
      position: 'fixed', left: 0, top: 0,
      width: 'var(--sidebar-width)',
      height: '100vh',
      background: 'var(--bg-surface)',
      borderRight: '1px solid var(--bg-border)',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 100,
    }}>
      {/* Logo */}
      <div style={{
        padding: '24px 16px 20px',
        fontFamily: 'var(--font-mono)',
        fontWeight: 700,
        fontSize: 15,
        color: 'var(--accent)',
        letterSpacing: '0.05em',
        borderBottom: '1px solid var(--bg-border)',
      }}>
        ◆ SCROOT
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, paddingTop: 8 }}>
        {NAV.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              height: 48,
              padding: '0 16px',
              color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
              background: isActive ? 'var(--bg-elevated)' : 'transparent',
              borderLeft: isActive ? '3px solid var(--accent)' : '3px solid transparent',
              fontFamily: 'var(--font-sans)',
              fontSize: 14,
              textDecoration: 'none',
              transition: 'all var(--transition-fast)',
              cursor: 'pointer',
              userSelect: 'none',
              position: 'relative',
            })}
            onMouseEnter={e => {
              if (!e.currentTarget.classList.contains('active')) {
                e.currentTarget.style.background = 'var(--bg-elevated)';
              }
            }}
            onMouseLeave={e => {
              if (!e.currentTarget.classList.contains('active')) {
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            <span style={{ fontSize: 14, width: 16, textAlign: 'center' }}>{icon}</span>
            <span style={{ flex: 1 }}>{label}</span>
            {label === 'Queue' && pending > 0 && (
              <span style={{
                background: 'var(--accent)',
                color: '#0A0C10',
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                fontWeight: 700,
                padding: '1px 7px',
                borderRadius: 10,
                minWidth: 22,
                textAlign: 'center',
              }}>
                {pending}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Status */}
      <div style={{
        padding: '12px 16px',
        borderTop: '1px solid var(--bg-border)',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 4,
        }}>
          <span style={{
            width: 8, height: 8,
            borderRadius: '50%',
            background: connected ? 'var(--green)' : 'var(--red)',
            boxShadow: connected ? '0 0 6px var(--green)' : 'none',
            animation: connected ? 'pulse 2s infinite' : 'none',
          }} />
          <style>{`
            @keyframes pulse {
              0%,100% { opacity: 1; }
              50% { opacity: 0.5; }
            }
          `}</style>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: connected ? 'var(--green)' : 'var(--red)',
            fontWeight: 700,
            letterSpacing: '0.05em',
          }}>
            {connected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
        }}>
          v0.1.0
        </div>
      </div>
    </aside>
  );
}
