import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import PageWrapper from '../components/PageWrapper'

/* Real data from the confusion matrix PNG (cnn_confusion_matrix.png, Macro F1 = 0.829) */
const CM = {
  classes: ['none', 'wash', 'paint', 'spoof', 'mirror'],
  matrix: [
    [7860, 199, 1316, 79, 247],   // true: none
    [11, 640, 24, 0, 2],          // true: wash
    [147, 34, 1139, 19, 10],      // true: paint
    [11, 3, 20, 1229, 1],         // true: spoof
    [28, 0, 5, 0, 1226],          // true: mirror
  ],
}

const COLORS = { none: '#4cc38a', wash: '#F44336', paint: '#FF9800', spoof: '#9C27B0', mirror: '#2196F3' }

// Compute per-class precision, recall, F1
function computeMetrics() {
  const n = CM.classes.length
  const metrics = []
  for (let c = 0; c < n; c++) {
    const tp = CM.matrix[c][c]
    const fp = CM.matrix.reduce((sum, row, i) => sum + (i !== c ? row[c] : 0), 0)
    const fn = CM.matrix[c].reduce((sum, v, j) => sum + (j !== c ? v : 0), 0)
    const precision = tp / (tp + fp)
    const recall = tp / (tp + fn)
    const f1 = 2 * precision * recall / (precision + recall)
    const support = CM.matrix[c].reduce((s, v) => s + v, 0)
    metrics.push({ class: CM.classes[c], precision, recall, f1, support, color: COLORS[CM.classes[c]] })
  }
  return metrics
}

const metrics = computeMetrics()
const macroF1 = metrics.reduce((s, m) => s + m.f1, 0) / metrics.length

export default function Results() {
  return (
    <PageWrapper>
      <section className="py-24 px-8">
        <div className="max-w-[1320px] mx-auto">
          <div className="section-label">
            <span className="num">04</span>
            <span>Results</span>
          </div>
          <div className="mb-14">
            <h2 className="mb-5 max-w-[800px]" style={{ fontSize: 'clamp(36px, 4.4vw, 56px)', fontVariationSettings: '"SOFT" 40, "opsz" 144', letterSpacing: '-0.03em' }}>
              The <span className="italic text-gold">numbers</span>, honestly.
            </h2>
            <p className="text-[17px] text-text-soft max-w-[640px] font-light">
              We show failures alongside victories. The confusion matrix tells the full story — including where the detector struggles.
            </p>
          </div>

          {/* Top stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border-soft border border-border-soft mb-12">
            {[
              ['Macro F1', macroF1.toFixed(3)],
              ['Test Windows', '14,769'],
              ['Best Per-class', 'spoof · 0.97'],
              ['Weakest', 'none · 0.81'],
            ].map(([label, val], i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                className="bg-bg-elev p-5">
                <div className="font-mono text-[10px] tracking-widest uppercase text-text-muted mb-2">{label}</div>
                <div className="font-mono text-[22px] font-medium text-gold">{val}</div>
              </motion.div>
            ))}
          </div>

          <div className="grid lg:grid-cols-[1.3fr_1fr] gap-8">
            {/* Confusion Matrix */}
            <div className="panel">
              <div className="flex items-center justify-between mb-6">
                <span className="eyebrow">Confusion Matrix</span>
                <span className="font-mono text-[10px] text-text-dim">rows = true, cols = predicted</span>
              </div>
              <img src="/images/cnn_confusion_matrix.png" alt="CNN Confusion Matrix — Macro F1 = 0.829"
                   className="w-full rounded-sm" style={{ filter: 'brightness(1.1) contrast(1.05)' }} />

              {/* Failure analysis */}
              <div className="mt-6 pt-6 border-t border-border-soft">
                <div className="eyebrow mb-3">Failure Analysis</div>
                <div className="space-y-3 text-[12.5px] text-text-soft leading-[1.6]">
                  <div className="flex items-start gap-2">
                    <span className="text-red mt-0.5">⚠</span>
                    <span><strong className="text-text">14% of "none" windows misclassified as "paint"</strong> — the biggest confusion cell. Paint trading's subtle signature (small trades near mid) overlaps with natural noise-trader activity.</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-green mt-0.5">✓</span>
                    <span><strong className="text-text">Spoof and mirror achieve 97% recall</strong> — their aggressive cancel patterns and synchronized depth are distinctive enough for the CNN to learn confidently.</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-gold mt-0.5">→</span>
                    <span><strong className="text-text">Wash trading: 95% accuracy</strong> — the round-trip cadence between CA and CB is the strongest single signal.</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Per-class metrics */}
            <div className="space-y-6">
              <div className="panel">
                <span className="eyebrow mb-5 block">Per-Class F1 Score</span>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={metrics} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 50 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" opacity={0.3} horizontal={false} />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--color-text-dim)' }} />
                    <YAxis type="category" dataKey="class" tick={{ fontSize: 11, fill: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }} />
                    <Tooltip contentStyle={{ background: 'var(--color-bg-elev)', border: '1px solid var(--color-border)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                             formatter={(v) => [v.toFixed(3), 'F1']} />
                    <Bar dataKey="f1" radius={[0, 3, 3, 0]}>
                      {metrics.map((m, i) => <Cell key={i} fill={m.color} opacity={0.85} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Precision & Recall Table */}
              <div className="panel">
                <span className="eyebrow mb-4 block">Precision / Recall / Support</span>
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-[11.5px]">
                    <thead>
                      <tr className="text-text-muted text-left border-b border-border-soft">
                        <th className="py-2 pr-4">Class</th>
                        <th className="py-2 px-3 text-right">Prec.</th>
                        <th className="py-2 px-3 text-right">Recall</th>
                        <th className="py-2 px-3 text-right">F1</th>
                        <th className="py-2 pl-3 text-right">Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map(m => (
                        <tr key={m.class} className="border-b border-border-soft/50 hover:bg-bg-elev-2 transition-colors">
                          <td className="py-2.5 pr-4 font-medium" style={{ color: m.color }}>{m.class}</td>
                          <td className="py-2.5 px-3 text-right text-text-soft">{m.precision.toFixed(3)}</td>
                          <td className="py-2.5 px-3 text-right text-text-soft">{m.recall.toFixed(3)}</td>
                          <td className="py-2.5 px-3 text-right text-gold font-medium">{m.f1.toFixed(3)}</td>
                          <td className="py-2.5 pl-3 text-right text-text-dim">{m.support.toLocaleString()}</td>
                        </tr>
                      ))}
                      <tr className="font-semibold">
                        <td className="py-2.5 pr-4 text-text">Macro</td>
                        <td className="py-2.5 px-3 text-right text-text-soft">{(metrics.reduce((s,m)=>s+m.precision,0)/5).toFixed(3)}</td>
                        <td className="py-2.5 px-3 text-right text-text-soft">{(metrics.reduce((s,m)=>s+m.recall,0)/5).toFixed(3)}</td>
                        <td className="py-2.5 px-3 text-right text-gold">{macroF1.toFixed(3)}</td>
                        <td className="py-2.5 pl-3 text-right text-text-dim">{metrics.reduce((s,m)=>s+m.support,0).toLocaleString()}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Collusion Evidence */}
              <div className="panel">
                <span className="eyebrow mb-4 block">Collusion Evidence — Order Flow + Detector Confidence</span>
                <img src="/images/collusion_evidence.png" alt="Collusion Evidence — 5-panel order flow visualization"
                     className="w-full rounded-sm" style={{ filter: 'brightness(1.05)' }} />
                <p className="text-[11.5px] text-text-muted mt-4 leading-[1.5]">
                  Five episodes — one per class — showing CA/CB order positions (scatter) against mid price (line), with the detector's per-window collusion score below. Shaded regions mark ground-truth scheme windows.
                </p>
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
                Showing failures honestly is what differentiates a student project from research. The none → paint confusion is real and documented — not hidden.
              </p>
            </div>
            <Link to="/rl-honesty" className="btn-primary">
              <span>The honesty proof</span>
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
