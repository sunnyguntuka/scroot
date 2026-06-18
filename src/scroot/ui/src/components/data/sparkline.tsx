interface SparklineProps {
  data: number[]
  declining?: boolean
}

// Bars sized by percentage height (0-100). Last bar turns red if declining.
export function Sparkline({ data, declining }: SparklineProps) {
  return (
    <span
      className="spark"
      style={{ display: 'inline-flex', alignItems: 'flex-end', gap: 2, height: 18, verticalAlign: 'middle' }}
    >
      {data.map((h, i) => {
        const isLast = i === data.length - 1
        return (
          <b
            key={i}
            style={{
              width: 3.5,
              borderRadius: 1.5,
              height: `${h}%`,
              background: declining && isLast ? 'var(--red)' : 'var(--accent)',
              opacity: 0.6,
              transition: 'opacity 0.15s',
            }}
          />
        )
      })}
    </span>
  )
}
