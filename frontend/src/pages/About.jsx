import { motion } from 'framer-motion'
import PageWrapper from '../components/PageWrapper'

export default function About() {
  return (
    <PageWrapper>
      <section className="py-24 px-8">
        <div className="max-w-[960px] mx-auto">
          <div className="section-label">
            <span className="num">07</span>
            <span>About</span>
          </div>

          <h2 className="mb-10 max-w-[700px]" style={{ fontSize: 'clamp(36px, 4.4vw, 56px)', fontVariationSettings: '"SOFT" 40, "opsz" 144', letterSpacing: '-0.03em' }}>
            The <span className="italic text-gold">story</span> behind the project.
          </h2>

          <div className="grid md:grid-cols-[1.3fr_1fr] gap-8 mb-12">
            {/* Methods */}
            <div className="space-y-8">
              <div className="panel">
                <span className="eyebrow mb-4 block">Methodology</span>
                <div className="space-y-4 text-[14px] text-text-soft leading-[1.7]">
                  <p>
                    <strong className="text-text">The question:</strong> Can a lightweight neural network reliably distinguish between algorithmic collusion and legitimate multi-agent trading in a simulated limit order book?
                  </p>
                  <p>
                    <strong className="text-text">The approach:</strong> Generate 5,000 episodes of realistic microstructure data across five classes (none + four collusion schemes), extract 200-event sliding windows with 6 per-event channels, and train a 1D-CNN classifier with episode-level group splitting to prevent data leakage.
                  </p>
                  <p>
                    <strong className="text-text">The validation:</strong> Train two independent PPO agents (IPPO) in the same LOB environment and confirm the detector classifies them as "none" — proving no false positives on legitimate multi-agent trading.
                  </p>
                </div>
              </div>

              <div className="panel">
                <span className="eyebrow mb-4 block">Tech Stack</span>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { cat: 'Simulation', items: ['C++14 LOB Engine', 'Python simulator.py', 'std::mt19937_64 RNG'] },
                    { cat: 'Detection', items: ['PyTorch 2.x', '1D-CNN (25K params)', 'scikit-learn eval'] },
                    { cat: 'RL Agents', items: ['Gymnasium', 'PPO + GAE-λ', 'IPPO (2 agents)'] },
                    { cat: 'Frontend', items: ['React + Vite', 'Three.js', 'Recharts + Framer Motion'] },
                  ].map(group => (
                    <div key={group.cat} className="space-y-2">
                      <div className="font-mono text-[10px] tracking-widest uppercase text-gold">{group.cat}</div>
                      {group.items.map(item => (
                        <div key={item} className="font-mono text-[11.5px] text-text-muted">{item}</div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* References + Project Info */}
            <div className="space-y-8">
              <div className="panel">
                <span className="eyebrow mb-4 block">Key References</span>
                <div className="space-y-4">
                  {[
                    { authors: 'Calvano et al. (2020)', title: 'Artificial Intelligence, Algorithmic Pricing, and Collusion', venue: 'American Economic Review', note: 'Shows Q-learning agents learn tacit collusion in pricing games.' },
                    { authors: 'Cont, Stoikov, Talreja (2010)', title: 'A stochastic model for order book dynamics', venue: 'Operations Research', note: 'Foundational LOB model that informs our simulator design.' },
                    { authors: 'Assad et al. (2024)', title: 'Algorithmic Pricing and Competition', venue: 'Journal of Political Economy', note: 'Empirical evidence of algorithmic collusion in retail gasoline markets.' },
                  ].map((ref, i) => (
                    <motion.div key={i} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                      className="pb-4 border-b border-border-soft last:border-0"
                    >
                      <div className="font-mono text-[11px] text-gold mb-1">{ref.authors}</div>
                      <div className="text-[13px] text-text font-medium mb-0.5">{ref.title}</div>
                      <div className="text-[11px] text-text-muted italic">{ref.venue}</div>
                      <div className="text-[11px] text-text-dim mt-1">{ref.note}</div>
                    </motion.div>
                  ))}
                </div>
              </div>

              <div className="panel">
                <span className="eyebrow mb-4 block">Project</span>
                <div className="space-y-3 text-[13px] text-text-soft">
                  <p><strong className="text-text">Type:</strong> Undergraduate Research Project</p>
                  <p><strong className="text-text">Scope:</strong> Algorithmic Market Integrity</p>
                  <p><strong className="text-text">Status:</strong> <span className="text-green">Complete</span> — all pipeline stages functional</p>
                </div>
                <div className="mt-6 stat-grid">
                  <div><div className="l">Source Files</div><div className="v">20+</div></div>
                  <div><div className="l">Languages</div><div className="v">C++ · Python</div></div>
                  <div><div className="l">LOC</div><div className="v">~8,000</div></div>
                  <div><div className="l">Year</div><div className="v">2026</div></div>
                </div>
              </div>

              <div className="panel">
                <span className="eyebrow mb-4 block">Source Code Structure</span>
                <div className="font-mono text-[11px] text-text-muted leading-[1.8]">
                  <div><span className="text-gold">├──</span> run.py <span className="text-text-dim">· pipeline orchestrator</span></div>
                  <div><span className="text-gold">├──</span> predict.py <span className="text-text-dim">· inference CLI</span></div>
                  <div><span className="text-gold">├──</span> viz.py <span className="text-text-dim">· evidence visualization</span></div>
                  <div><span className="text-gold">├──</span> Simulator/ <span className="text-text-dim">· LOB + traders</span></div>
                  <div><span className="text-gold">├──</span> detectors/ <span className="text-text-dim">· CNN training + eval</span></div>
                  <div><span className="text-gold">├──</span> rl_bots/ <span className="text-text-dim">· IPPO agents</span></div>
                  <div><span className="text-gold">├──</span> cpp_engine/ <span className="text-text-dim">· C++ LOB port</span></div>
                  <div><span className="text-gold">└──</span> data_prep/ <span className="text-text-dim">· feature extraction</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </PageWrapper>
  )
}
