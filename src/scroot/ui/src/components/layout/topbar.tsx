import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Sun, Moon, Download } from 'lucide-react'
import { useTheme } from '../../stores/theme'

const TABS = [
  { to: '/', label: 'Overview', end: true },
  { to: '/inbox', label: 'Inbox' },
  { to: '/evidence', label: 'Evidence' },
  { to: '/scores', label: 'Scores' },
  { to: '/calibration', label: 'Calibration' },
  { to: '/flags', label: 'Flags' },
  { to: '/pipeline', label: 'Pipeline' },
  { to: '/export', label: 'Export' },
  { to: '/settings', label: 'Settings' },
]

function Wordmark() {
  return (
    <svg width="130" height="30" viewBox="0 0 208 49" fill="none" aria-label="scroot" style={{ color: 'var(--text-1)', flexShrink: 0 }}>
      <g fill="currentColor">
        <path d="M94.6,12.6 L91.6,12.1 L87.5,12.1 L85.7,12.4 L81.4,14.6 L78.5,17.9 L77.3,20.5 L76.7,23.6 L76.7,43.1 L77.3,44.3 L78.7,45.1 L80.0,44.9 L81.2,43.9 L81.4,43.3 L81.4,23.6 L82.4,20.7 L83.2,19.5 L84.6,18.3 L87.5,17.0 L90.2,16.8 L92.8,17.4 L95.0,17.2 L95.9,16.2 L96.1,14.4 L95.9,13.8 Z" />
        <path d="M51.0,12.1 L48.0,12.8 L44.1,15.0 L40.0,19.5 L37.6,26.2 L37.8,32.5 L40.1,38.4 L44.9,43.1 L50.0,45.3 L57.2,45.3 L61.4,43.7 L65.5,40.3 L67.3,37.8 L67.5,36.6 L66.9,35.2 L65.9,34.5 L63.7,34.6 L60.4,38.6 L57.0,40.3 L52.1,40.7 L49.2,40.0 L44.9,36.6 L43.1,33.5 L42.3,28.0 L43.3,23.6 L44.7,21.3 L47.2,18.7 L49.4,17.6 L52.7,16.8 L55.9,17.0 L58.4,17.8 L60.6,19.1 L63.1,22.5 L64.1,23.1 L66.3,22.9 L67.3,21.7 L67.5,20.3 L65.7,17.4 L61.8,14.0 L56.5,12.1 Z" />
        <path d="M13.2,12.1 L9.3,13.4 L6.0,16.2 L4.2,20.9 L4.8,24.6 L7.5,28.2 L10.1,29.5 L20.5,31.7 L22.9,32.9 L24.2,35.2 L23.8,38.0 L22.1,39.6 L18.5,40.7 L14.8,40.7 L11.9,40.0 L9.9,38.8 L7.3,36.0 L5.2,36.2 L4.0,38.6 L6.2,41.9 L10.1,44.5 L15.6,45.7 L20.7,45.3 L25.4,43.1 L28.2,40.0 L29.1,36.8 L28.4,31.9 L26.2,29.1 L23.1,27.4 L13.2,25.4 L10.9,24.4 L9.3,22.7 L9.1,20.7 L10.3,18.7 L11.7,17.8 L15.2,16.8 L17.9,16.8 L21.1,17.6 L25.8,21.3 L28.2,20.1 L28.6,17.9 L26.0,14.8 L22.7,12.8 L18.9,11.9 Z" />
        <path d="M188.5,4.0 L186.9,5.8 L186.9,35.0 L187.7,38.4 L189.7,41.7 L191.4,43.3 L194.0,44.7 L196.5,45.3 L199.3,45.3 L202.4,44.7 L203.8,43.5 L204.0,42.7 L203.8,41.5 L202.8,40.3 L200.7,40.1 L199.5,40.5 L196.5,40.5 L195.4,40.1 L192.8,38.0 L191.6,35.2 L191.6,17.6 L202.0,17.4 L203.6,16.2 L204.0,15.2 L203.8,13.8 L203.0,12.8 L201.8,12.4 L191.8,12.4 L191.4,5.2 L190.1,4.0 Z" />
      </g>
      <g fill="var(--accent)" fillRule="evenodd">
        <path d="M100.9,13.4 L100.5,14.2 L100.9,16.2 L103.6,18.3 L104.6,20.7 L104.8,31.7 L105.2,33.7 L106.9,37.8 L110.1,41.5 L112.1,42.9 L115.4,44.5 L118.3,45.1 L122.5,45.1 L125.2,44.5 L128.4,43.1 L131.5,40.7 L133.1,39.0 L134.6,36.4 L135.8,33.3 L136.2,31.3 L136.2,26.8 L135.8,24.4 L134.1,20.3 L132.9,18.5 L130.3,16.0 L127.8,14.2 L122.9,12.6 L101.8,12.6 Z M108.1,17.2 L108.3,17.0 L121.5,17.0 L124.8,17.8 L127.4,19.3 L130.1,22.3 L131.3,24.6 L131.9,27.2 L131.9,30.9 L131.5,32.7 L130.9,34.3 L129.3,36.6 L128.0,38.0 L126.0,39.4 L124.2,40.1 L121.3,40.7 L119.1,40.7 L116.0,40.0 L113.2,38.4 L111.5,36.6 L110.5,35.2 L109.3,32.3 L109.1,31.3 L109.1,21.3 L108.7,19.5 L108.1,18.3 Z" />
        <path d="M176.7,12.8 L155.3,12.6 L150.8,14.0 L147.6,16.0 L145.5,18.1 L143.5,21.1 L142.3,24.0 L141.7,26.8 L141.7,30.9 L142.3,33.9 L143.9,37.4 L145.3,39.4 L147.6,41.7 L150.0,43.3 L152.7,44.5 L155.5,45.1 L159.8,45.1 L162.0,44.7 L165.3,43.3 L167.7,41.7 L171.0,38.0 L172.8,33.9 L173.2,32.3 L173.4,21.1 L174.5,18.3 L177.5,16.0 L177.7,14.2 Z M170.0,17.0 L170.2,17.9 L169.4,19.3 L169.0,21.1 L169.0,30.9 L168.8,31.9 L167.7,35.0 L165.7,37.6 L163.9,39.0 L162.0,40.0 L159.0,40.7 L156.7,40.7 L154.5,40.3 L151.7,39.2 L150.6,38.4 L148.4,36.2 L146.4,32.5 L146.0,30.5 L146.0,27.2 L146.8,24.2 L147.6,22.7 L148.6,21.3 L150.6,19.3 L153.3,17.8 L156.3,17.0 Z" />
      </g>
      <g fill="var(--accent)">
        <circle cx="157.7" cy="28.7" r="4.2" />
        <circle cx="120.4" cy="28.7" r="4.2" />
      </g>
    </svg>
  )
}

function LiveClock() {
  const [time, setTime] = useState('')

  useEffect(() => {
    const fmt = () => {
      const now = new Date()
      const mon = now.toLocaleString('en-US', { month: 'short' })
      const day = now.getDate()
      const hh = now.getHours().toString().padStart(2, '0')
      const mm = now.getMinutes().toString().padStart(2, '0')
      const ss = now.getSeconds().toString().padStart(2, '0')
      const ampm = now.getHours() >= 12 ? 'PM' : 'AM'
      const h12 = (now.getHours() % 12 || 12).toString().padStart(2, '0')
      return `${mon} ${day} · ${h12}:${mm}:${ss} ${ampm}`
    }
    setTime(fmt())
    const id = setInterval(() => setTime(fmt()), 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <span
      style={{
        fontFamily: "'DM Mono', monospace",
        fontSize: 11,
        color: 'var(--text-3)',
      }}
    >
      {time}
    </span>
  )
}

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const isDark = theme === 'dark'

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      style={{
        width: 36,
        height: 36,
        borderRadius: '50%',
        border: '0.5px solid var(--border)',
        background: 'var(--bg-2)',
        color: 'var(--text-2)',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s cubic-bezier(0.22,1,0.36,1)',
      }}
    >
      <span
        style={{
          position: 'absolute',
          transition: 'all 0.4s cubic-bezier(0.34,1.56,0.64,1)',
          opacity: isDark ? 1 : 0,
          transform: isDark ? 'translateY(0) rotate(0)' : 'translateY(-10px) rotate(30deg)',
        }}
      >
        <Sun size={14} />
      </span>
      <span
        style={{
          position: 'absolute',
          transition: 'all 0.4s cubic-bezier(0.34,1.56,0.64,1)',
          opacity: isDark ? 0 : 1,
          transform: isDark ? 'translateY(10px) rotate(-30deg)' : 'translateY(0) rotate(0)',
        }}
      >
        <Moon size={14} />
      </span>
    </button>
  )
}

export function Topbar() {
  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 20,
        background: 'var(--bg-glass)',
        backdropFilter: 'blur(24px) saturate(1.4)',
        WebkitBackdropFilter: 'blur(24px) saturate(1.4)',
        borderBottom: '0.5px solid var(--border)',
        padding: '0 32px',
        height: 52,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}
    >
      {/* Left: brand + tabs */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 0, marginRight: 4 }}>
          <Wordmark />
          <span
            style={{
              fontSize: 8,
              fontWeight: 500,
              letterSpacing: '1.2px',
              textTransform: 'uppercase',
              color: 'var(--text-3)',
              position: 'relative',
              top: 2,
              marginLeft: 3,
            }}
          >
            STUDIO
          </span>
        </div>

        {/* Tabs */}
        <nav style={{ display: 'flex' }}>
          {TABS.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              end={tab.end}
              style={({ isActive }) => ({
                fontSize: 12,
                fontWeight: 500,
                color: isActive ? 'var(--text-1)' : 'var(--text-3)',
                padding: '16px 16px',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                textDecoration: 'none',
                position: 'relative',
                transition: 'color 0.2s cubic-bezier(0.22,1,0.36,1)',
              })}
            >
              {({ isActive }) => (
                <>
                  {tab.label}
                  {isActive && (
                    <span
                      style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 16,
                        right: 16,
                        height: 1.5,
                        background: 'var(--accent)',
                        borderRadius: 1,
                      }}
                    />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Right: clock + theme toggle + export */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <LiveClock />
        <ThemeToggle />
        <button
          aria-label="Export"
          style={{
            fontFamily: "'Inter', system-ui, sans-serif",
            fontSize: 11,
            fontWeight: 500,
            padding: '6px 14px',
            borderRadius: 6,
            border: '0.5px solid var(--border)',
            background: 'transparent',
            color: 'var(--text-2)',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 5,
            transition: 'all 0.2s cubic-bezier(0.22,1,0.36,1)',
          }}
        >
          <Download size={12} />
          Export
        </button>
      </div>
    </header>
  )
}
