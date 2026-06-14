import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Inbox, BarChart2, Download, Settings, Wand2, Zap, ShieldCheck } from 'lucide-react';
import { api } from '../../api/client';

// Stacked-lamella logo mark - pure SVG, matches assets/scroot-logo.png
function ScrootLogo() {
  return (
    <svg width="22" height="22" viewBox="162 84 236 200" fill="none" aria-hidden>
      <defs>
        <linearGradient id="sidebar-g1" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#c4b5fd" /><stop offset="1" stopColor="#9061f9" />
        </linearGradient>
        <linearGradient id="sidebar-g2" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#a78bfa" /><stop offset="1" stopColor="#7c6df9" />
        </linearGradient>
        <linearGradient id="sidebar-g3" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#93b6fb" /><stop offset="1" stopColor="#6ea8fb" />
        </linearGradient>
        <linearGradient id="sidebar-g4" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#8dd3fd" /><stop offset="1" stopColor="#5ec5fb" />
        </linearGradient>
        <linearGradient id="sidebar-g5" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#a5f3fc" /><stop offset="1" stopColor="#67e8f9" />
        </linearGradient>
      </defs>
      <g strokeLinejoin="round" strokeWidth="10">
        <path d="M280 222 L388 248 L280 274 L172 248 Z" fill="url(#sidebar-g5)" stroke="#67e8f9" />
        <path d="M280 190 L388 216 L280 242 L172 216 Z" fill="url(#sidebar-g4)" stroke="#5ec5fb" />
        <path d="M280 158 L388 184 L280 210 L172 184 Z" fill="url(#sidebar-g3)" stroke="#6ea8fb" />
        <path d="M280 126 L388 152 L280 178 L172 152 Z" fill="url(#sidebar-g2)" stroke="#7c6df9" />
        <path d="M280  94 L388 120 L280 146 L172 120 Z" fill="url(#sidebar-g1)" stroke="#9061f9" />
      </g>
    </svg>
  );
}

const NAV = [
  { to: '/queue',     label: 'Inbox',    Icon: Inbox,    badge: 'pending' },
  { to: '/analytics', label: 'Analytics',Icon: BarChart2 },
  { to: '/export',    label: 'Export',   Icon: Download },
  { to: '/pipeline',  label: 'Pipeline', Icon: Wand2,    newBadge: true },
  { to: '/settings',  label: 'Settings', Icon: Settings },
];

export function Sidebar({ pendingCount = 0, collapsed = false, activeGuardrails = 0 }) {
  const [newPipeline, setNewPipeline] = useState(false);

  useEffect(() => {
    const seen = localStorage.getItem('pipeline-seen');
    if (!seen) setNewPipeline(true);
  }, []);

  const clearNewBadge = () => {
    localStorage.setItem('pipeline-seen', '1');
    setNewPipeline(false);
  };

  return (
    <aside
      className={`
        fixed left-0 top-0 h-full bg-white border-r border-indigo-100 flex flex-col z-50
        transition-all duration-200
        ${collapsed ? 'w-12' : 'w-[220px]'}
      `}
    >
      {/* Logo */}
      <div className={`flex items-center gap-2.5 px-4 py-5 border-b border-indigo-100 shrink-0 ${collapsed ? 'justify-center px-0' : ''}`}>
        <ScrootLogo />
        {!collapsed && (
          <div>
            <div className="text-[15px] font-semibold text-indigo-950 leading-none font-sans">
              scroot
            </div>
            <div className="text-[10px] font-mono-score text-indigo-400 mt-0.5">v0.1.2</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV.map(({ to, label, Icon, badge, newBadge }) => (
          <NavLink
            key={to}
            to={to}
            onClick={newBadge ? clearNewBadge : undefined}
            className={({ isActive }) => `
              flex items-center gap-3 mx-2 px-3 py-2.5 rounded-lg text-sm
              transition-colors duration-100 select-none
              ${isActive
                ? 'bg-indigo-50 text-indigo-700 font-medium'
                : 'text-gray-500 hover:text-indigo-600 hover:bg-indigo-25'
              }
              ${collapsed ? 'justify-center' : ''}
            `}
            title={collapsed ? label : undefined}
          >
            <Icon size={17} strokeWidth={1.8} className="shrink-0" />
            {!collapsed && (
              <>
                <span className="flex-1">{label}</span>
                {/* Pending count badge on Inbox */}
                {badge === 'pending' && pendingCount > 0 && (
                  <span className="bg-indigo-600 text-white font-mono-score text-[10px] rounded-full px-1.5 min-w-[18px] text-center leading-[18px]">
                    {pendingCount > 99 ? '99+' : pendingCount}
                  </span>
                )}
                {/* "New" badge on Pipeline */}
                {newBadge && newPipeline && (
                  <span className="bg-green-100 text-green-700 text-[9px] font-medium uppercase tracking-wide rounded-full px-1.5 py-0.5">
                    new
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Guardrail status - "loop closed" signal */}
      {!collapsed && activeGuardrails > 0 && (
        <div className="px-4 py-2.5 border-t border-indigo-50">
          <div className="flex items-center gap-1.5 text-[11px] text-indigo-400">
            <ShieldCheck size={12} strokeWidth={2} className="text-green-500 shrink-0" />
            <span>{activeGuardrails} correction{activeGuardrails === 1 ? '' : 's'} active as guardrails</span>
          </div>
        </div>
      )}

      {/* Cloud upgrade footnote - feels like a footnote, never a banner */}
      {!collapsed && (
        <div className="px-4 py-3 border-t border-indigo-50">
          <a
            href="https://scroot.dev/cloud"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-[11px] text-indigo-400 hover:text-indigo-500 transition-colors"
          >
            <Zap size={11} strokeWidth={2} />
            Get team features →
          </a>
        </div>
      )}
    </aside>
  );
}
