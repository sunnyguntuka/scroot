import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './globals.css'
import { App } from './app'
import { Overview } from './pages/overview'
import { Inbox } from './pages/inbox'
import { Evidence } from './pages/evidence'
import { Scores } from './pages/scores'
import { Calibration } from './pages/calibration'
import { Flags } from './pages/flags'
import { Pipeline } from './pages/pipeline'
import { Export } from './pages/export'
import { Settings } from './pages/settings'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Overview /> },
      { path: 'inbox', element: <Inbox /> },
      { path: 'evidence/:id', element: <Evidence /> },
      { path: 'scores', element: <Scores /> },
      { path: 'calibration', element: <Calibration /> },
      { path: 'flags', element: <Flags /> },
      { path: 'pipeline', element: <Pipeline /> },
      { path: 'export', element: <Export /> },
      { path: 'settings', element: <Settings /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
)
