import { create } from 'zustand'

type Theme = 'dark' | 'light'

interface ThemeStore {
  theme: Theme
  toggle: () => void
}

export const useTheme = create<ThemeStore>((set) => {
  const stored = (typeof localStorage !== 'undefined'
    ? localStorage.getItem('scroot-theme')
    : null) as Theme | null
  const initial: Theme = stored ?? 'dark'

  if (typeof document !== 'undefined') {
    document.documentElement.setAttribute('data-theme', initial)
  }

  return {
    theme: initial,
    toggle: () =>
      set((s) => {
        const next: Theme = s.theme === 'dark' ? 'light' : 'dark'
        localStorage.setItem('scroot-theme', next)
        document.documentElement.setAttribute('data-theme', next)
        return { theme: next }
      }),
  }
})
