/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        indigo: {
          25: '#F5F7FF',
          950: '#1E1B4B',
        },
      },
      fontFamily: {
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        card: '12px',
      },
      fontSize: {
        '2xs': ['10px', '14px'],
      },
      boxShadow: {
        card: '0 1px 3px 0 rgba(79,70,229,0.06), 0 0 0 1px rgba(224,231,255,0.8)',
        'card-hover': '0 4px 12px 0 rgba(79,70,229,0.10), 0 0 0 1px rgba(199,210,254,0.9)',
      },
      animation: {
        'slide-up': 'slideUp 200ms ease-out',
        'fade-in':  'fadeIn 150ms ease-out',
        shimmer:    'shimmer 1.6s infinite linear',
      },
      keyframes: {
        slideUp: {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-600px 0' },
          '100%': { backgroundPosition: '600px 0' },
        },
      },
    },
  },
  plugins: [],
};
