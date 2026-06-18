import { StrictMode, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './globals.css'
import { App } from './app'

const Overview = lazy(() => import('./pages/overview').then((m) => ({ default: m.Overview })))
const Inbox = lazy(() => import('./pages/inbox').then((m) => ({ default: m.Inbox })))
const Evidence = lazy(() => import('./pages/evidence').then((m) => ({ default: m.Evidence })))
const Scores = lazy(() => import('./pages/scores').then((m) => ({ default: m.Scores })))
const Calibration = lazy(() => import('./pages/calibration').then((m) => ({ default: m.Calibration })))
const Flags = lazy(() => import('./pages/flags').then((m) => ({ default: m.Flags })))
const Pipeline = lazy(() => import('./pages/pipeline').then((m) => ({ default: m.Pipeline })))
const Export = lazy(() => import('./pages/export').then((m) => ({ default: m.Export })))
const Settings = lazy(() => import('./pages/settings').then((m) => ({ default: m.Settings })))

function PageFallback() {
  return (
    <div style={{ padding: '28px 32px', fontSize: 12, color: 'var(--text-3)' }} aria-busy="true">
      Loading…
    </div>
  )
}

function withSuspense(node: React.ReactNode) {
  return <Suspense fallback={<PageFallback />}>{node}</Suspense>
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: withSuspense(<Overview />) },
      { path: 'inbox', element: withSuspense(<Inbox />) },
      { path: 'evidence', element: withSuspense(<Evidence />) },
      { path: 'evidence/:id', element: withSuspense(<Evidence />) },
      { path: 'scores', element: withSuspense(<Scores />) },
      { path: 'calibration', element: withSuspense(<Calibration />) },
      { path: 'flags', element: withSuspense(<Flags />) },
      { path: 'pipeline', element: withSuspense(<Pipeline />) },
      { path: 'export', element: withSuspense(<Export />) },
      { path: 'settings', element: withSuspense(<Settings />) },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
)
