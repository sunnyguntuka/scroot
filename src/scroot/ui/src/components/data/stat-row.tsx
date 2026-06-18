import type { ReactNode } from 'react'

interface StatRowProps {
  label: string
  value: ReactNode
  border?: boolean
}

export function StatRow({ label, value, border = true }: StatRowProps) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '7px 0',
        fontSize: 12,
        borderBottom: border ? '0.5px solid var(--border)' : 'none',
      }}
    >
      <span style={{ color: 'var(--text-2)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--mono)', fontWeight: 500 }}>{value}</span>
    </div>
  )
}
