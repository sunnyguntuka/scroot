interface Option {
  value: string
  label: string
}

interface FilterSelectProps {
  options: Option[]
  value: string
  onChange: (v: string) => void
  label?: string
}

export function FilterSelect({ options, value, onChange, label }: FilterSelectProps) {
  return (
    <select
      className="sel"
      value={value}
      aria-label={label ?? 'filter'}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: 'var(--bg-2)',
        border: '0.5px solid var(--border)',
        borderRadius: 6,
        padding: '7px 12px',
        color: 'var(--text-2)',
        fontFamily: 'var(--sans)',
        fontSize: 11,
        cursor: 'pointer',
        outline: 'none',
        appearance: 'none',
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}
