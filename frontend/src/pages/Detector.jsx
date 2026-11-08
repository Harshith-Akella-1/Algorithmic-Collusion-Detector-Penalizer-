import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import PageWrapper from '../components/PageWrapper'

const CNN_LAYERS = [
  { idx: 'L01', name: 'Input', meta: 'events × channels', shape: '[200, 6]', desc: '6 channels: normalized_ts, side (+1/−1), log qty, is_CA, is_limit, is_market' },
  { idx: 'L02', name: 'Conv1d', meta: 'kernel 5, pad 2', shape: '[200, 32]', desc: '32 filters learn local patterns: cancel bursts, trade cadence, agent clustering' },
  { idx: 'L03', name: 'BatchNorm + ReLU', meta: '', shape: '[200, 32]', desc: 'Normalize activations, introduce non-linearity' },
  { idx: 'L04', name: 'MaxPool ÷2', meta: '', shape: '[100, 32]', desc: 'Downsample temporal dimension by half' },
  { idx: 'L05', name: 'Dropout', meta: 'p=0.2', shape: '[100, 32]', desc: 'Regularization — prevents co-adaptation of filter responses' },
  { idx: 'L06', name: 'Conv1d', meta: 'kernel 5, pad 2', shape: '[100, 64]', desc: '64 filters capture higher-order temporal relationships across wider windows' },
  { idx: 'L07', name: 'BatchNorm + ReLU', meta: '', shape: '[100, 64]', desc: 'Second normalization + activation' },
  { idx: 'L08', name: 'AdaptiveMaxPool', meta: '→ 1', shape: '[1, 64]', desc: 'Global max pool — collapses temporal axis, keeping the strongest activations' },
  { idx: 'L09', name: 'Linear → ReLU', meta: '64→64', shape: '[64]', desc: 'Dense layer for feature mixing' },
  { idx: 'L10', name: 'Dropout → Linear', meta: 'p=0.3, 64→5', shape: '[5]', desc: 'Final classifier: softmax over 5 classes (none, wash, paint, spoof, mirror)' },
]

const FEATURES = [
  { name: 'normalized_ts', desc: 'Event time normalized to [0, 1] within the window', example: 'Rhythmic trading shows periodic spikes' },
  { name: 'side', desc: '+1 for buy, -1 for sell', example: 'Wash trading alternates buy/sell symmetrically' },
  { name: 'qty_normalized', desc: 'log1p(quantity) / 6', example: 'Spoofing shows large phantom orders' },
  { name: 'is_CA', desc: '1 if trader is colluder A, else 0', example: 'Bursts of CA activity during scheme windows' },
  { name: 'is_limit', desc: '1 if order type is limit', example: 'Spoofers flood limit orders then cancel' },
  { name: 'is_market', desc: '1 if order type is market', example: 'Wash trading uses market orders to cross' },
]

export default function Detector() {
  return (
    <PageWrapper>
      <section className="py-24 px-8">
        <div className="max-w-[1320px] mx-auto">
          <div className="section-label">
            <span className="num">03</span>
            <span>The Detector</span>
          </div>
          <div className="mb-14">
            <h2 className="mb-5 max-w-[800px]" style={{ fontSize: 'clamp(36px, 4.4vw, 56px)', fontVariationSettings: '"SOFT" 40, "opsz" 144', letterSpacing: '-0.03em' }}>
              A 25K-parameter <span className="italic text-gold">watchdog</span>.
            </h2>
            <p className="text-[17px] text-text-soft max-w-[640px] font-light">
              Two convolutional blocks over 200-event sliding windows. Each event carries six channels — the CNN learns to spot collusion patterns without hand-crafted features.
            </p>
          </div>

          <div className="grid lg:grid-cols-[1.2fr_1fr] gap-8">
            {/* Architecture image + layer breakdown */}
            <div className="space-y-6">
              {/* CNN Architecture Image */}
              <div className="panel">
                <div className="flex items-center justify-between mb-6">
                  <span className="eyebrow">CNN1D Architecture</span>
                  <span className="font-mono text-[10px] text-text-dim">~25,000 parameters</span>
                </div>
                <img src="/images/cnn_architecture.png" alt="CNN Architecture Diagram"
                     className="w-full rounded-sm border border-border-soft"
                     style={{ filter: 'brightness(1.05) contrast(1.02)' }} />
              </div>

              {/* Layer-by-layer */}
              <div className="panel">
                <span className="eyebrow mb-4 block">Layer-by-layer</span>
                <div className="space-y-1.5">
                  {CNN_LAYERS.map((l, i) => (
                    <motion.div
                      key={l.idx}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.3, delay: i * 0.04 }}
                      className="grid gap-3 items-center p-3 border border-border-soft bg-bg-elev-2 group hover:border-gold-line transition-colors cursor-default"
                      style={{ gridTemplateColumns: '56px 1fr 90px' }}
                    >
                      <span className="font-mono text-gold text-[11px] tracking-wider">{l.idx}</span>
                      <div>
                        <span className="font-mono text-[12px] text-text font-medium">{l.name}</span>
                        {l.meta && <span className="font-mono text-[10.5px] text-text-muted ml-2">{l.meta}</span>}
                        <div className="text-[10px] text-text-dim mt-0.5 hidden group-hover:block leading-[1.5]">{l.desc}</div>
                      </div>
                      <span className="font-mono text-[10.5px] text-text-muted text-right tracking-wide">{l.shape}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Input features + model card */}
            <div className="space-y-6">
              {/* 6 Input Channels */}
              <div className="panel">
                <span className="eyebrow mb-5 block">Input Channels (6 per event)</span>
                <div className="space-y-4">
                  {FEATURES.map((f, i) => (
                    <motion.div
                      key={f.name}
                      initial={{ opacity: 0, y: 10 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.3, delay: i * 0.05 }}
                      className="pb-4 border-b border-border-soft last:border-0"
                    >
                      <div className="flex items-baseline justify-between mb-1">
                        <span className="font-mono text-[13px] text-gold font-medium">{f.name}</span>
                        <span className="font-mono text-[9px] text-text-dim tracking-wider">CH {i + 1}</span>
                      </div>
                      <p className="text-[12.5px] text-text-soft leading-[1.5]">{f.desc}</p>
                      <p className="text-[11px] text-text-muted mt-1 italic">↳ {f.example}</p>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Model Card */}
              <div className="panel">
                <span className="eyebrow mb-5 block">Model Card</span>
                <div className="stat-grid">
                  <div><div className="l">Parameters</div><div className="v">~25,000</div></div>
                  <div><div className="l">Inference</div><div className="v">&lt; 50ms</div></div>
                  <div><div className="l">Window</div><div className="v">60s / 30s stride</div></div>
                  <div><div className="l">Classes</div><div className="v">5</div></div>
                  <div><div className="l">Loss</div><div className="v">Weighted CE</div></div>
                  <div><div className="l">Split</div><div className="v">70/15/15 ep.</div></div>
                  <div><div className="l">Epochs</div><div className="v">80 (early stop)</div></div>
                  <div><div className="l">Optimizer</div><div className="v">Adam 1e-3</div></div>
                </div>
                <p className="text-[12px] text-text-muted mt-4 leading-[1.6]">
                  <strong className="text-text-soft">Episode-level grouping:</strong> No window from a given episode appears in more than one of train / val / test — preventing information leakage.
                </p>
              </div>

              {/* Training highlights */}
              <div className="panel">
                <span className="eyebrow mb-3 block">Training Details</span>
                <ul className="space-y-2 text-[12.5px] text-text-soft leading-[1.6]">
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text">Class-weighted CE loss</strong> — balances the majority "none" class against 4 minority collusion types</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text">Cosine LR schedule</strong> — warm restart every 20 epochs, min LR 1e-5</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text">Early stopping</strong> on macro-F1 with patience 10, best checkpoint saved as <code className="bg-bg-deep px-1 py-0.5 text-gold-bright text-[11px]">cnn_best.pt</code></span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">▹</span>
                    <span><strong className="text-text">No data augmentation</strong> — temporal order matters for these patterns</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Takeaway */}
          <motion.div initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}
            className="mt-12 p-8 bg-bg-elev border border-border flex items-center justify-between flex-wrap gap-6"
          >
            <div>
              <h3 className="text-[22px] text-text italic mb-2" style={{ fontFamily: 'var(--font-display)' }}>So what?</h3>
              <p className="text-[14px] text-text-muted font-light max-w-[500px]">
                25K parameters is remarkably small — smaller than most image classifiers' first layer alone. This proves that collusion patterns have strong, learnable temporal signatures.
              </p>
            </div>
            <Link to="/results" className="btn-primary">
              <span>See the numbers</span>
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
