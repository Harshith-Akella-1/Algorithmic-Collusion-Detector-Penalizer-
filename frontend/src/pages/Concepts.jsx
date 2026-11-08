import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import PageWrapper from '../components/PageWrapper'

const SCHEMES = [
  {
    key: 'wash', num: '01', name: 'Wash Trading', color: '#F44336',
    tagline: 'Round-trip · same name',
    oneLiner: 'Buying and selling the same instrument between coordinated parties to inflate apparent volume.',
    harmful: 'Misleads other traders about liquidity and demand, causing them to make decisions based on false signals.',
    signal: 'High AB trade count, regular cadence, near-zero net position change.',
    config: 'ColluderPair("CA","CB", mode="wash") — A posts limit at mid, B crosses with market order every 8–20s.',
    vizData: Array.from({length:14}, (_, i) => ({ x: i, y: i % 2 === 0 ? 20 : 45 })),
  },
  {
    key: 'paint', num: '02', name: 'Painting the Tape', color: '#FF9800',
    tagline: 'Synthetic prints · low impact',
    oneLiner: 'Rhythmic small trades between coordinated parties at or near the mid price — manufacturing the illusion of activity.',
    harmful: 'Creates fake price marks that mislead technical analysis and trigger algorithmic reactions in other bots.',
    signal: 'Low inter-trade coefficient of variation, tight price deviation from mid, alternating buy/sell symmetry.',
    config: 'ColluderPair("CA","CB", mode="paint") — A posts limit near mid, B crosses opposite side. Alternates buy/sell every 15–35s.',
    vizData: Array.from({length:24}, (_, i) => ({ x: i, y: 30 + Math.sin(i * 1.8) * 4 + ((i*7)%5 - 2) })),
  },
  {
    key: 'spoof', num: '03', name: 'Spoofing', color: '#9C27B0',
    tagline: 'Layered intent · pulled fast',
    oneLiner: 'Stacking large phantom limit orders to move the mid, then cancelling before execution while an accomplice trades the induced move.',
    harmful: 'Distorts the apparent supply/demand balance, causing the mid price to shift artificially.',
    signal: 'CA cancel burst, one-sided depth evaporation, phantom volume ratio, short order lifetimes.',
    config: 'ColluderPair("CA","CB", mode="spoof") — A places 2–6 layered limits, B market-trades opposite, A cancels 60–100% within 0.3s.',
    vizData: Array.from({length:8}, (_, i) => ({ x: i, h: i < 5 ? 36 : 6 })),
  },
  {
    key: 'mirror', num: '04', name: 'Mirror Quoting', color: '#2196F3',
    tagline: 'Symmetric depth · sync cancel',
    oneLiner: 'Two parties build matching ladders on the same side, hold to mark the price, then cancel in sub-second sync.',
    harmful: 'Creates the appearance of deep liquidity on one side, marking the price, then evaporates simultaneously.',
    signal: 'High sync-cancel ratio, same-side depth evaporation, symmetric ladder pattern.',
    config: 'ColluderPair("CA","CB", mode="mirror") — A builds 3–6 layers, B mirrors with ~TICK/2 offset, both cancel within 0.1–1.0s lag.',
    vizData: Array.from({length:5}, (_, i) => ({ y: 14 + i * 8, w: 50 + i * 10 })),
  },
]

function SchemeViz({ scheme }) {
  const W = 280, H = 60
  if (scheme.key === 'wash') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {scheme.vizData.map((p, i) => (
          <g key={i}>
            <circle cx={(p.x / 13) * W} cy={p.y} r="3" fill={scheme.color} opacity={0.5 + (p.x % 2) * 0.4} />
            {i > 0 && (
              <line
                x1={((scheme.vizData[i-1].x) / 13) * W}
                y1={scheme.vizData[i-1].y}
                x2={(p.x / 13) * W}
                y2={p.y}
                stroke={scheme.color} strokeWidth="0.6" opacity="0.5"
              />
            )}
          </g>
        ))}
      </svg>
    )
  }
  if (scheme.key === 'paint') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        <line x1="0" y1="30" x2={W} y2="30" stroke="var(--color-text-muted)" strokeDasharray="2,3" opacity="0.5" />
        {scheme.vizData.map((p, i) => (
          <circle key={i} cx={(p.x / 23) * W} cy={p.y} r="2.4" fill={scheme.color} opacity="0.85" />
        ))}
      </svg>
    )
  }
  if (scheme.key === 'spoof') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {scheme.vizData.map((p, i) => (
          <rect key={i} x={(p.x / 7) * (W * 0.65)} y={50 - p.h} width="6" height={p.h}
                fill={scheme.color} opacity={p.h > 10 ? 0.7 : 0.3} />
        ))}
        <text x={W * 0.72} y="34" fontFamily="var(--font-mono)" fontSize="10" fill={scheme.color} letterSpacing="1.5">CANCEL ✕</text>
      </svg>
    )
  }
  if (scheme.key === 'mirror') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {scheme.vizData.map((p, i) => (
          <g key={i}>
            <rect x="20" y={p.y} width={p.w} height="3" fill={scheme.color} opacity="0.85" />
            <rect x={20 + p.w + 14} y={p.y} width={p.w} height="3" fill={scheme.color} opacity="0.5" />
          </g>
        ))}
        <text x={W/2} y="58" textAnchor="middle" fontFamily="var(--font-mono)" fontSize="9" fill="var(--color-text-muted)" letterSpacing="1.2">MIRRORED LADDERS</text>
      </svg>
    )
  }
  return null
}

export default function Concepts() {
  const [expanded, setExpanded] = useState(null)

  return (
    <PageWrapper>
      <section className="py-24 px-8">
        <div className="max-w-[1320px] mx-auto">
          <div className="section-label">
            <span className="num">01</span>
            <span>Collusion Types</span>
          </div>

          <div className="mb-14">
            <h2 className="mb-5 max-w-[800px]"
                style={{
                  fontSize: 'clamp(36px, 4.4vw, 56px)',
                  fontVariationSettings: '"SOFT" 40, "opsz" 144',
                  letterSpacing: '-0.03em',
                }}>
              Four ways algorithms <span className="italic text-gold">cheat</span>.
            </h2>
            <p className="text-[17px] text-text-soft max-w-[640px] font-light">
              Each scheme leaves a different fingerprint on the limit order book. The detector is trained against all four — plus a <em>none</em> baseline of legitimate noise-trader activity.
            </p>
          </div>

          {/* Cards Grid */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {SCHEMES.map((s, i) => (
              <motion.article
                key={s.key}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="bg-bg-elev border border-border p-7 flex flex-col gap-4 relative cursor-pointer group
                           hover:-translate-y-0.5 hover:border-gold-line transition-all duration-280 overflow-hidden"
                onClick={() => setExpanded(expanded === s.key ? null : s.key)}
              >
                {/* Top accent */}
                <div className="absolute top-0 left-0 h-px transition-all duration-320 group-hover:w-full"
                     style={{ width: '24px', background: s.color }} />

                <div className="eyebrow">{s.num} / {s.tagline.split(' · ')[0].toUpperCase()}</div>

                <h3 style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '28px',
                  fontVariationSettings: '"SOFT" 50, "opsz" 144',
                  letterSpacing: '-0.02em',
                  lineHeight: 1.05,
                }}>
                  {s.name.split(' ').slice(0, -1).join(' ')}{' '}
                  <span className="italic" style={{ color: s.color }}>{s.name.split(' ').slice(-1)}</span>
                </h3>

                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                  color: 'var(--color-text-muted)',
                }}>{s.tagline}</div>

                <p className="text-[13.5px] leading-[1.65] text-text-soft font-light">{s.oneLiner}</p>

                {/* Mini LOB viz */}
                <div className="h-[60px] mt-1 pt-4 border-t border-border-soft">
                  <SchemeViz scheme={s} />
                </div>

                {/* Tags */}
                <div className="flex gap-2 flex-wrap mt-1">
                  <span className="px-2 py-1 border text-[10px] font-mono tracking-wider uppercase"
                        style={{ borderColor: s.color, color: s.color, opacity: 0.8 }}>
                    Harmful
                  </span>
                  <span className="px-2 py-1 border border-gold-line text-gold text-[10px] font-mono tracking-wider uppercase opacity-80">
                    Detectable
                  </span>
                </div>

                {/* Expanded details */}
                <AnimatePresence>
                  {expanded === s.key && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="pt-4 border-t border-border-soft space-y-3">
                        <div>
                          <div className="eyebrow mb-1">Why it's harmful</div>
                          <p className="text-[12px] text-text-muted leading-[1.6]">{s.harmful}</p>
                        </div>
                        <div>
                          <div className="eyebrow mb-1">Detector signal</div>
                          <p className="text-[12px] text-text-muted leading-[1.6]">{s.signal}</p>
                        </div>
                        <div>
                          <div className="eyebrow mb-1">Simulator config</div>
                          <p className="text-[11px] text-text-dim leading-[1.6] font-mono">{s.config}</p>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.article>
            ))}
          </div>

          {/* Takeaway */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="mt-16 p-8 bg-bg-elev border border-border flex items-center justify-between flex-wrap gap-6"
          >
            <div>
              <h3 className="text-[22px] text-text italic mb-2" style={{ fontFamily: 'var(--font-display)', fontVariationSettings: '"SOFT" 60, "opsz" 144' }}>
                So what?
              </h3>
              <p className="text-[14px] text-text-muted font-light max-w-[500px]">
                Each scheme requires a different detection strategy. A single CNN learns all four from raw event sequences — no hand-crafted features, no rule-based thresholds.
              </p>
            </div>
            <Link to="/simulator" className="btn-primary">
              <span>Watch it in action</span>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </Link>
          </motion.div>
        </div>
      </section>
    </PageWrapper>
  )
}
