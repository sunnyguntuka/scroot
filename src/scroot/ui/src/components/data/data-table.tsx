import type { ReactNode } from 'react'

export interface Column {
  key: string
  label: string
  width?: string
}

export type Row = Record<string, ReactNode>

interface DataTableProps {
  columns: Column[]
  data: Row[]
  onRowClick?: (row: Row) => void
}

export function DataTable({ columns, data, onRowClick }: DataTableProps) {
  return (
    <table className="tbl" style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          {columns.map((c) => (
            <th
              key={c.key}
              style={{
                textAlign: 'left',
                fontWeight: 500,
                fontSize: 10,
                color: 'var(--text-3)',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                padding: '10px 12px',
                borderBottom: '0.5px solid var(--border)',
                width: c.width,
              }}
            >
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, ri) => (
          <tr
            key={ri}
            onClick={onRowClick ? () => onRowClick(row) : undefined}
            style={{ cursor: onRowClick ? 'pointer' : 'default' }}
          >
            {columns.map((c) => (
              <td
                key={c.key}
                style={{
                  padding: 12,
                  borderBottom: '0.5px solid var(--border)',
                  fontSize: 12.5,
                  transition: 'background 0.15s',
                }}
              >
                {row[c.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
