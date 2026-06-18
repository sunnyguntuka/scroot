import { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { Sidebar } from './components/layout/sidebar'
import { Topbar } from './components/layout/topbar'
import { useTheme } from './stores/theme'

export function App() {
  const { theme } = useTheme()

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const location = useLocation()

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '60px 1fr',
        height: '100vh',
      }}
    >
      <Sidebar />
      <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Topbar />
        <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>
          <AnimatePresence mode="wait">
            <Outlet key={location.pathname} />
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
