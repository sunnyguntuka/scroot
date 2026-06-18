import { MOCK_HEATMAP } from '../../lib/mock-data'

interface HeatmapProps {
  data?: number[][]
}

function cellColor(v: number): string {
  if (v < 0.3) return '#F87171'
  if (v < 0.5) return '#FBBF24'
  if (v < 0.8) return '#34D399'
  return '#0D9668'
}

function cellOpacity(v: number): number {
  // map 0.55 - 1.0 based on intensity within band
  return 0.55 + Math.min(v, 1) * 0.45
}

export function Heatmap({ data = MOCK_HEATMAP }: HeatmapProps) {
  const flat = data.flat()
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(24, 1fr)', gap: 3 }}>
        {flat.map((v, i) => (
          <div
            key={i}
            className="hm-cell"
            title={v.toFixed(2)}
            style={{
              aspectRatio: '1',
              borderRadius: 3,
              background: cellColor(v),
              opacity: cellOpacity(v),
              cursor: 'pointer',
            }}
          />
        ))}
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginTop: 10,
          fontSize: 10,
          color: 'var(--text-3)',
        }}
      >
        <span>quality:</span>
        <Legend color="#F87171" label="low" />
        <Legend color="#FBBF24" label="mid" />
        <Legend color="#34D399" label="good" />
        <Legend color="#0D9668" label="excellent" />
      </div>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <>
      <div style={{ width: 10, height: 10, borderRadius: 3, background: color }} />
      <span>{label}</span>
    </>
  )
}
