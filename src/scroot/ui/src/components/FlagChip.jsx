const FLAG_COLORS = {
  hallucination_risk: { bg: '#F5544218', border: '#F5544240', text: '#F55442' },
  incomplete:         { bg: '#F5C84218', border: '#F5C84240', text: '#F5C842' },
  off_topic:          { bg: '#4A9EF518', border: '#4A9EF540', text: '#4A9EF5' },
  self_contradictory: { bg: '#B45EF518', border: '#B45EF540', text: '#B45EF5' },
  ungrounded:         { bg: '#F5544218', border: '#F5544240', text: '#F55442' },
};

export function FlagChip({ flag }) {
  const c = FLAG_COLORS[flag] || FLAG_COLORS.off_topic;
  return (
    <span style={{
      fontSize: 11,
      fontFamily: 'var(--font-mono)',
      padding: '2px 7px',
      borderRadius: 3,
      background: c.bg,
      border: `1px solid ${c.border}`,
      color: c.text,
      whiteSpace: 'nowrap',
    }}>
      ⚑ {flag.replace(/_/g, ' ')}
    </span>
  );
}
