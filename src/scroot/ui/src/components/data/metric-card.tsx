import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

interface MetricCardProps {
  label: string
  value: string | number
  delta: string
  deltaDirection: 'up' | 'down' | 'neutral'
  decimals?: number
  suffix?: string
}

const deltaColor = {
  up: 'var(--green)',
  down: 'var(--red)',
  neutral: 'var(--text-3)',
} as const

function useCountUp(target: number, decimals: number, enabled: boolean) {
  const [value, setValue] = useState(enabled ? 0 : target)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    if (!enabled) {
      setValue(target)
      return
    }
    const duration = 1200
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setValue(target * eased)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [target, enabled])

  return decimals ? value.toFixed(decimals) : Math.round(value).toLocaleString()
}

export function MetricCard({ label, value, delta, deltaDirection, decimals = 0, suffix = '' }: MetricCardProps) {
  const isNumeric = typeof value === 'number'
  const reduced =
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const counted = useCountUp(isNumeric ? (value as number) : 0, decimals, isNumeric && !reduced)
  const display = isNumeric ? counted + suffix : String(value)

  return (
    <motion.div
      whileHover={{ y: -2, boxShadow: '0 8px 32px rgba(0,0,0,0.3)' }}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      style={{
        background: 'var(--bg-2)',
        border: '0.5px solid var(--border)',
        borderRadius: 12,
        padding: '16px 18px',
        cursor: 'default',
      }}
    >
      <div style={{ fontSize: 11, color: 'var(--text-3)', marginBottom: 8, fontWeight: 500 }}>{label}</div>
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 28,
          fontWeight: 500,
          letterSpacing: '-1px',
          lineHeight: 1.1,
          marginBottom: 6,
        }}
      >
        {display}
      </div>
      <div style={{ fontSize: 11, fontWeight: 500, color: deltaColor[deltaDirection] }}>{delta}</div>
    </motion.div>
  )
}
