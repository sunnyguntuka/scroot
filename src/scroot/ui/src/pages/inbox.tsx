import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { PageWrapper } from '../components/layout/page-wrapper'
import { SearchBar } from '../components/forms/search-bar'
import { FilterSelect } from '../components/forms/filter-select'
import { FlagPill } from '../components/data/flag-pill'
import { StatusPill } from '../components/data/status-pill'
import { MOCK_INBOX_RECORDS } from '../lib/mock-data'

const BADGE_TONE = {
  hi: { bg: 'var(--green-dim)', color: 'var(--green)' },
  mid: { bg: 'var(--amber-dim)', color: 'var(--amber)' },
  lo: { bg: 'var(--red-dim)', color: 'var(--red)' },
} as const

export function Inbox() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('all')
  const [time, setTime] = useState('all')
  const [sort, setSort] = useState('newest')

  const records = useMemo(
    () =>
      MOCK_INBOX_RECORDS.filter((r) => {
        if (query && !r.query.toLowerCase().includes(query.toLowerCase())) return false
        if (status !== 'all' && r.status.kind !== status) return false
        return true
      }),
    [query, status]
  )

  return (
    <PageWrapper>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 20,
        }}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 1 }}>
          <SearchBar value={query} onChange={setQuery} placeholder="Search queries..." />
          <FilterSelect
            label="status"
            value={status}
            onChange={setStatus}
            options={[
              { value: 'all', label: 'All status' },
              { value: 'pending', label: 'Pending' },
              { value: 'corrected', label: 'Corrected' },
              { value: 'passed', label: 'Passed' },
            ]}
          />
          <FilterSelect
            label="time"
            value={time}
            onChange={setTime}
            options={[
              { value: 'all', label: 'All time' },
              { value: 'today', label: 'Today' },
              { value: '7d', label: '7 days' },
              { value: '30d', label: '30 days' },
            ]}
          />
          <FilterSelect
            label="sort"
            value={sort}
            onChange={setSort}
            options={[
              { value: 'newest', label: 'Newest first' },
              { value: 'lowest', label: 'Lowest IQS' },
              { value: 'flags', label: 'Most flags' },
            ]}
          />
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-3)', fontFamily: 'var(--mono)' }}>847 records</span>
      </div>

      {records.map((r) => {
        const tone = BADGE_TONE[r.tone]
        return (
          <motion.div
            key={r.id}
            whileHover={{ y: -2, borderColor: 'var(--border-hover)' }}
            transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
            onClick={() => navigate(`/evidence/${r.id}`)}
            style={{
              background: 'var(--bg-2)',
              border: '0.5px solid var(--border)',
              borderRadius: 12,
              marginBottom: 8,
              padding: '14px 18px',
              cursor: 'pointer',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div
                style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 13,
                  fontWeight: 500,
                  padding: '5px 12px',
                  borderRadius: 5,
                  background: tone.bg,
                  color: tone.color,
                }}
              >
                {r.iqs.toFixed(2)}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 13, marginBottom: 3 }}>{r.query}</div>
                <div style={{ fontSize: 11, color: 'var(--text-3)' }}>{r.meta}</div>
              </div>
              {r.flag ? (
                <FlagPill flag={r.flag.label} />
              ) : (
                <span style={{ fontSize: 11, color: 'var(--text-3)' }}>no flags</span>
              )}
              <StatusPill status={r.status.kind} />
            </div>
          </motion.div>
        )
      })}
    </PageWrapper>
  )
}
