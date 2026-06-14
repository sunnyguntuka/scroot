import { Loader2 } from 'lucide-react';
import { forwardRef } from 'react';

const VARIANT = {
  primary:   'bg-indigo-600 text-white hover:bg-indigo-700 border-transparent shadow-sm',
  secondary: 'bg-white border-indigo-200 text-indigo-700 hover:bg-indigo-50',
  ghost:     'bg-transparent border-transparent text-indigo-600 hover:bg-indigo-50',
  danger:    'bg-red-600 text-white hover:bg-red-700 border-transparent shadow-sm',
  'danger-outline': 'bg-white border-red-200 text-red-700 hover:bg-red-50',
};

const SIZE = {
  sm: 'px-3 py-1.5 text-xs h-8',
  md: 'px-4 py-2 text-sm h-9',
  lg: 'px-5 py-2.5 text-sm h-10',
};

export const Button = forwardRef(function Button(
  { variant = 'primary', size = 'md', loading = false, disabled = false,
    icon, iconRight, children, className = '', fullWidth = false, ...props },
  ref
) {
  const isDisabled = disabled || loading;

  return (
    <button
      ref={ref}
      disabled={isDisabled}
      className={`
        inline-flex items-center justify-center gap-2 rounded-lg border font-medium
        transition-colors duration-150 cursor-pointer select-none
        disabled:opacity-50 disabled:cursor-not-allowed
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1
        ${VARIANT[variant]} ${SIZE[size]}
        ${fullWidth ? 'w-full' : ''}
        ${className}
      `}
      {...props}
    >
      {loading ? (
        <Loader2 size={14} className="animate-spin shrink-0" />
      ) : icon ? (
        <span className="shrink-0">{icon}</span>
      ) : null}
      {children}
      {iconRight && !loading && <span className="shrink-0">{iconRight}</span>}
    </button>
  );
});
