import { NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'

const NAV_ITEMS = [
  { path: '/', label: 'Home' },
  { path: '/concepts', label: 'Concepts' },
  { path: '/simulator', label: 'Simulator' },
  { path: '/detector', label: 'Detector' },
  { path: '/results', label: 'Results' },
  { path: '/rl-honesty', label: 'RL Proof' },
  { path: '/playground', label: 'Playground' },
  { path: '/about', label: 'About' },
]

function BrandMark() {
  return (
    <div className="w-8 h-8 grid place-items-center border border-gold-line relative"
         style={{ background: 'linear-gradient(135deg, var(--color-gold-glow), transparent)' }}>
      <div className="absolute inset-[4px] border border-gold/60" />
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
           className="w-4 h-4 text-gold relative z-10">
        <path d="M3 14l5-9 4 7 4-4 5 8" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="12" cy="12" r="2" fill="currentColor" stroke="none" opacity="0.4"/>
      </svg>
    </div>
  )
}

export default function Layout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen flex flex-col relative z-[2]">
      {/* Ticker Bar */}
      <TickerBar />

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border-soft"
              style={{
                backdropFilter: 'blur(18px) saturate(140%)',
                WebkitBackdropFilter: 'blur(18px) saturate(140%)',
                background: 'color-mix(in srgb, var(--color-bg) 75%, transparent)',
              }}>
        <div className="max-w-[1320px] mx-auto px-8 flex items-center justify-between h-[68px]">
          <NavLink to="/" className="flex items-center gap-3 no-underline text-text">
            <BrandMark />
            <span className="font-[Fraunces] font-medium text-[17px] tracking-[0.01em]">
              Algo<span className="text-gold">·</span>Collusion <em>Detector</em>
            </span>
          </NavLink>

          {/* Desktop Nav */}
          <nav className="hidden lg:flex gap-7" style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
            {NAV_ITEMS.map(item => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `relative py-1 transition-colors duration-180 ${isActive ? 'text-gold' : 'text-text-muted hover:text-text'}`
                }
              >
                {item.label}
                {location.pathname === item.path && (
                  <span className="absolute left-0 right-0 -bottom-0.5 h-px bg-gold" />
                )}
              </NavLink>
            ))}
          </nav>

          {/* Mobile hamburger */}
          <button className="lg:hidden w-10 h-10 grid place-items-center border border-border text-text-soft"
                  onClick={() => setMobileOpen(!mobileOpen)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-5 h-5">
              {mobileOpen
                ? <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round"/>
                : <><path d="M3 8h18M3 16h18" strokeLinecap="round"/></>
              }
            </svg>
          </button>
        </div>

        {/* Mobile Nav */}
        {mobileOpen && (
          <nav className="lg:hidden border-t border-border-soft bg-bg-elev px-8 py-4 flex flex-col gap-3"
               style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
            {NAV_ITEMS.map(item => (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) =>
                  `py-2 transition-colors ${isActive ? 'text-gold' : 'text-text-muted'}`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-border-soft py-14 px-8">
        <div className="max-w-[1320px] mx-auto flex justify-between items-end gap-8 flex-wrap">
          <div className="flex flex-col gap-2" style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-text-muted)' }}>
            <span style={{ fontSize: '10px', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-text-dim)' }}>
              Algo·Collusion Detector / v1.0 / 2026
            </span>
            <span>An undergraduate research project on algorithmic market integrity.</span>
          </div>
          <div className="flex gap-4 flex-wrap"
               style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.16em', textTransform: 'uppercase', color: 'var(--color-text-dim)' }}>
            {['C++14', 'Python', 'PyTorch', 'Gymnasium', 'React', 'Three.js'].map(t => (
              <span key={t} className="px-2.5 py-1 border border-border-soft">{t}</span>
            ))}
          </div>
        </div>
      </footer>
    </div>
  )
}

function TickerBar() {
  const symbols = [
    ['NIFTY 50', '24,318.42', '+0.34%', true],
    ['BTC/USD',  '69,824.10', '-1.21%', false],
    ['ES1!',     '5,732.75',  '+0.18%', true],
    ['GC1!',     '2,419.30',  '+0.42%', true],
    ['VIX',      '14.83',     '-2.10%', false],
    ['SPX',      '5,729.84',  '+0.11%', true],
    ['DXY',      '104.22',    '-0.08%', false],
    ['CRUDE',    '78.41',     '+1.04%', true],
  ]

  const block = symbols.map(([sym, px, chg, up]) =>
    `<span><strong style="color:var(--color-text)">${sym}</strong> <span style="color:var(--color-text-muted)">${px}</span> <span style="color:var(--color-${up ? 'green' : 'red'})">${chg}</span> <span style="color:var(--color-gold);margin:0 8px">·</span></span>`
  ).join('')

  return (
    <div className="relative z-[5] h-8 bg-bg-deep border-b border-border-soft overflow-hidden flex items-center"
         style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', letterSpacing: '0.06em', color: 'var(--color-text-muted)' }}>
      <div className="shrink-0 bg-gold text-bg-deep font-semibold px-3.5 py-1"
           style={{ letterSpacing: '0.18em', textTransform: 'uppercase', fontSize: '10px' }}>
        LIVE FEED
      </div>
      <div className="flex gap-12 whitespace-nowrap pl-12"
           style={{ animation: 'ticker 90s linear infinite' }}
           dangerouslySetInnerHTML={{ __html: block + block }} />
    </div>
  )
}
