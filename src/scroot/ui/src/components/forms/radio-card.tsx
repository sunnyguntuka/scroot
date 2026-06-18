import type { ReactNode } from 'react'

interface RadioCardProps {
  label: string
  description?: string
  badge?: ReactNode
  selected: boolean
  disabled?: boolean
  onChange: () => void
}

export function RadioCard({ label, description, badge, selected, disabled, onChange }: RadioCardProps) {
  return (
    <label
      className="radio-card"
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: 10,
        padding: '12px 14px',
        borderRadius: 8,
        marginBottom: 6,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        border: selected ? '0.5px solid rgba(91,140,255,0.25)' : '0.5px solid var(--border)',
        background: selected ? 'var(--accent-dim)' : 'transparent',
      }}
    >
      <input
        type="radio"
        checked={selected}
        disabled={disabled}
        onChange={onChange}
        style={{ accentColor: 'var(--accent)', marginTop: 2 }}
      />
      <div>
        <div
          style={{
            fontSize: 13,
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: disabled ? 'var(--text-3)' : 'var(--text-1)',
          }}
        >
          {label}
          {badge}
        </div>
        {description && (
          <div style={{ fontSize: 11, color: disabled ? 'var(--text-4)' : 'var(--text-3)', marginTop: 4 }}>
            {description}
          </div>
        )}
      </div>
    </label>
  )
}
