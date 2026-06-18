interface SearchBarProps {
  value: string
  onChange: (v: string) => void
  placeholder?: string
}

export function SearchBar({ value, onChange, placeholder = 'Search...' }: SearchBarProps) {
  return (
    <div
      className="search-box"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        background: 'var(--bg-2)',
        border: '0.5px solid var(--border)',
        borderRadius: 6,
        padding: '7px 12px',
        flex: 1,
        maxWidth: 320,
      }}
    >
      <span style={{ color: 'var(--text-3)', fontSize: 14 }} aria-hidden="true">
        ⌕
      </span>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        aria-label={placeholder}
        onChange={(e) => onChange(e.target.value)}
        style={{
          background: 'none',
          border: 'none',
          outline: 'none',
          color: 'var(--text-1)',
          fontFamily: 'var(--sans)',
          fontSize: 12,
          width: '100%',
        }}
      />
    </div>
  )
}
