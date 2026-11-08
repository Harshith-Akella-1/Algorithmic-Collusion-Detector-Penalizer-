import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import PageWrapper from '../components/PageWrapper'

const RL_EPISODES = [
  { label: 'mappo_ep_0', prediction: 'none', confidence: 0.012, windows: 19, flagged: 0 },
  { label: 'mappo_ep_1', prediction: 'none', confidence: 0.008, windows: 19, flagged: 0 },
  { label: 'mappo_ep_2', prediction: 'none', confidence: 0.015, windows: 19, flagged: 0 },
  { label: 'mappo_ep_3', prediction: 'none', confidence: 0.021, windows: 19, flagged: 0 },
  { label: 'mappo_ep_4', prediction: 'none', confidence: 0.006, windows: 19, flagged: 0 },
]

const COLLUSION_EPISODES = [
  { label: 'wash_ep_1773', prediction: 'wash', confidence: 0.938, windows: 19, flagged: 11 },
  { label: 'spoof_ep_3438', prediction: 'spoof', confidence: 0.992, windows: 19, flagged: 13 },
  { label: 'mirror_ep_4433', prediction: 'mirror', confidence: 0.998, windows: 19, flagged: 10 },
  { label: 'paint_ep_2654', prediction: 'paint', confidence: 0.827, windows: 19, flagged: 9 },
]

function EpisodeRow({ ep, isCollusion }) {
  const color = isCollusion ? 'var(--color-red)' : 'var(--color-green)'
  return (
    <div className="grid gap-3 items-center p-4 border border-border-soft bg-bg-elev-2 hover:border-gold-line transition-colors"
         style={{ gridTemplateColumns: '20px 1fr 90px 80px 80px' }}>
      <span className="text-[14px]">{isCollusion ? '✕' : '✓'}</span>
      <span className="font-mono text-[12px] text-text">{ep.label}</span>
      <span className="font-mono text-[12px] text-right" style={{ color }}>{ep.prediction}</span>
      <span className="font-mono text-[11px] text-right text-text-muted">{(ep.confidence * 100).toFixed(1)}%</span>
      <span className="font-mono text-[11px] text-right text-text-dim">{ep.flagged}/{ep.windows}</span>
    </div>
  )
}

export default function RLHonesty() {
  return (
    <PageWrapper>
      <section className="py-24 px-8">
        <div className="max-w-[1320px] mx-auto">
          <div className="section-label">
            <span className="num">05</span>
            <span>RL Honesty Proof</span>
          </div>
          <div className="mb-14">
            <h2 className="mb-5 max-w-[800px]" style={{ fontSize: 'clamp(36px, 4.4vw, 56px)', fontVariationSettings: '"SOFT" 40, "opsz" 144', letterSpacing: '-0.03em' }}>
              Profitable ≠ <span className="italic text-gold">colluding</span>.
            </h2>
            <p className="text-[17px] text-text-soft max-w-[640px] font-light">
              Two independent PPO agents trade the same limit order book profitably — without any coordination. The CNN correctly classifies every window as "none."
            </p>
          </div>

          {/* Two-column comparison */}
          <div className="grid lg:grid-cols-2 gap-6 mb-12">
            {/* RL Agents — All Clear */}
            <motion.div initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
              className="panel relative"
            >
              <div className="absolute top-0 left-0 w-full h-px bg-green" />
              <div className="flex items-center justify-between mb-5">
                <span className="eyebrow" style={{ color: 'var(--color-green)' }}>RL Agents — IPPO</span>
                <span className="font-mono text-[12px] font-semibold text-green">PASS</span>
              </div>

              {/* Column headers */}
              <div className="grid gap-3 items-center px-4 py-2 mb-1 font-mono text-[9px] tracking-widest uppercase text-text-dim"
                   style={{ gridTemplateColumns: '20px 1fr 90px 80px 80px' }}>
                <span></span><span>Episode</span><span className="text-right">Pred</span>
                <span className="text-right">Score</span><span className="text-right">Flagged</span>
              </div>

              <div className="space-y-1">
                {RL_EPISODES.map((ep, i) => (
                  <motion.div key={i} initial={{ opacity: 0, x: -10 }} whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
                    <EpisodeRow ep={ep} isCollusion={false} />
                  </motion.div>
                ))}
              </div>

              <div className="mt-5 p-4 border border-green/30 bg-green-soft">
                <p className="font-mono text-[11px] text-green leading-[1.6]">
                  ✓ ALL {RL_EPISODES.length} EPISODES CLASSIFIED AS "NONE"<br/>
                  ✓ ZERO WINDOWS FLAGGED ACROSS {RL_EPISODES.length * 19} TOTAL WINDOWS<br/>
                  ✓ MEAN COLLUSION SCORE: {(RL_EPISODES.reduce((s,e)=>s+e.confidence,0)/RL_EPISODES.length).toFixed(3)}
                </p>
              </div>
            </motion.div>

            {/* Collusion Episodes — Caught */}
            <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
              className="panel relative"
            >
              <div className="absolute top-0 left-0 w-full h-px bg-red" />
              <div className="flex items-center justify-between mb-5">
                <span className="eyebrow" style={{ color: 'var(--color-red)' }}>Collusion Episodes</span>
                <span className="font-mono text-[12px] font-semibold text-red">CAUGHT</span>
              </div>

              <div className="grid gap-3 items-center px-4 py-2 mb-1 font-mono text-[9px] tracking-widest uppercase text-text-dim"
                   style={{ gridTemplateColumns: '20px 1fr 90px 80px 80px' }}>
                <span></span><span>Episode</span><span className="text-right">Pred</span>
                <span className="text-right">Score</span><span className="text-right">Flagged</span>
              </div>

              <div className="space-y-1">
                {COLLUSION_EPISODES.map((ep, i) => (
                  <motion.div key={i} initial={{ opacity: 0, x: 10 }} whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }} transition={{ delay: i * 0.08 }}>
                    <EpisodeRow ep={ep} isCollusion={true} />
                  </motion.div>
                ))}
              </div>

              <div className="mt-5 p-4 border border-red/30 bg-red-soft">
                <p className="font-mono text-[11px] text-red leading-[1.6]">
                  ✕ ALL {COLLUSION_EPISODES.length} EPISODES CORRECTLY FLAGGED<br/>
                  ✕ MEAN CONFIDENCE: {(COLLUSION_EPISODES.reduce((s,e)=>s+e.confidence,0)/COLLUSION_EPISODES.length*100).toFixed(1)}%<br/>
                  ✕ 43/{COLLUSION_EPISODES.length * 19} WINDOWS ABOVE THRESHOLD
                </p>
              </div>
            </motion.div>
          </div>

          {/* IPPO Architecture */}
          <div className="grid lg:grid-cols-[1fr_1.2fr] gap-6 mb-12">
            <div className="panel">
              <span className="eyebrow mb-5 block">IPPO Architecture</span>
              <div className="space-y-3 text-[13px] text-text-soft leading-[1.65]">
                <p><strong className="text-text">Two independent PPO agents</strong> (RL_A and RL_B) trade the same limit order book alongside 13 noise traders and 1 market maker.</p>
                <ul className="space-y-2 pl-4">
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text-soft">No shared critic</strong> — this is IPPO (Independent PPO), not centralized MAPPO</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text-soft">No shared observations</strong> — each agent sees only its own position, cash, PnL</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text-soft">Independent rewards</strong> — pure PnL maximization, zero collusion incentive</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text-soft">Centralized training, decentralized execution</strong></span>
                  </li>
                </ul>
              </div>
              <div className="stat-grid mt-6">
                <div><div className="l">Iterations</div><div className="v">300</div></div>
                <div><div className="l">Rollout</div><div className="v">4,800</div></div>
                <div><div className="l">Best Eval A</div><div className="v">6.57</div></div>
                <div><div className="l">Best Eval B</div><div className="v">4.63</div></div>
              </div>
            </div>

            {/* Training curves */}
            <div className="panel">
              <span className="eyebrow mb-4 block">IPPO Training Curves</span>
              <img src="/images/mappo_training_curves.png" alt="IPPO Training Curves — Reward, Entropy, Inventory"
                   className="w-full rounded-sm" style={{ filter: 'brightness(1.05)' }} />
              <p className="text-[11.5px] text-text-muted mt-4 leading-[1.5]">
                Top-left: reward convergence (both agents become profitable). Top-right: policy entropy decreases as agents learn. Bottom: inventory control stays well below the 500-unit hard cap.
              </p>
            </div>
          </div>

          {/* Takeaway */}
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
            className="mt-4 p-8 bg-bg-elev border border-border flex items-center justify-between flex-wrap gap-6"
          >
            <div>
              <h3 className="text-[22px] text-text italic mb-2" style={{ fontFamily: 'var(--font-display)' }}>So what?</h3>
              <p className="text-[14px] text-text-muted font-light max-w-[500px]">
                The detector doesn't false-positive on legitimate, profitable multi-agent trading. This is the <em className="text-gold italic">intellectual punchline</em> of the entire project.
              </p>
            </div>
            <Link to="/playground" className="btn-primary">
              <span>Try it yourself</span>
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
