interface ThresholdSliderProps {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  hint?: string
  decimals?: number
  showSign?: boolean
}

export function ThresholdSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  hint,
  decimals = 2,
  showSign = false,
}: ThresholdSliderProps) {
  const display = (showSign && value >= 0 ? '+' : '') + value.toFixed(decimals)
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 8 }}>
        <span style={{ color: 'var(--text-2)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--mono)', fontWeight: 500, color: 'var(--accent)' }}>{display}</span>
      </div>
      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {hint && <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 4 }}>{hint}</div>}
    </div>
  )
}
