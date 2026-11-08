import { useState, useRef, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'
import PageWrapper from '../components/PageWrapper'

const MODES = ['none', 'wash', 'paint', 'spoof', 'mirror']
const MODE_COLORS = { none: '#4cc38a', wash: '#F44336', paint: '#FF9800', spoof: '#9C27B0', mirror: '#2196F3' }
const MODE_LABELS = { none: 'Honest Market', wash: 'Wash Trading', paint: 'Painting Tape', spoof: 'Spoofing', mirror: 'Mirror Quoting' }

/* Generate synthetic episode data for fallback */
function generateEpisode(mode) {
  const duration = 600
  const ticks = []
  const predictions = []
  let mid = 100.0
  const rng = () => Math.random()

  const wStart = mode === 'none' ? 0 : 120 + rng() * 180
  const wEnd = mode === 'none' ? 0 : wStart + 100 + rng() * 100

  for (let t = 0; t < duration; t += 1) {
    const inWindow = mode !== 'none' && t >= wStart && t <= wEnd
    let drift = (rng() - 0.5) * 0.06
    if (inWindow && mode === 'spoof') drift -= 0.03
    if (inWindow && mode === 'wash') drift += (rng() - 0.5) * 0.01
    mid += drift
    mid = Math.max(98, Math.min(102, mid))

    const orders = []
    const nOrders = Math.floor(4 + rng() * 8)
    for (let j = 0; j < nOrders; j++) {
      const isCA = inWindow && rng() < 0.3
      const isCB = inWindow && !isCA && rng() < 0.3
      orders.push({
        trader: isCA ? 'CA' : isCB ? 'CB' : `N${Math.floor(rng() * 10)}`,
        side: rng() < 0.5 ? 'buy' : 'sell',
        price: mid + (rng() - 0.5) * 0.2,
        qty: Math.floor(10 + rng() * 90),
      })
    }
    ticks.push({ t, mid: +mid.toFixed(3), orders })
  }

  for (let ws = 0; ws + 60 <= duration; ws += 30) {
    const we = ws + 60
    const overlapStart = Math.max(ws, wStart)
    const overlapEnd = Math.min(we, wEnd)
    const overlap = mode !== 'none' ? Math.max(0, overlapEnd - overlapStart) / 60 : 0

    const scores = { none: 0, wash: 0, paint: 0, spoof: 0, mirror: 0 }
    if (overlap > 0.25) {
      scores[mode] = 0.5 + rng() * 0.45
      scores.none = 1 - scores[mode] - 0.05
      Object.keys(scores).forEach(k => { if (k !== mode && k !== 'none') scores[k] = rng() * 0.03 })
    } else {
      scores.none = 0.7 + rng() * 0.25
      Object.keys(scores).forEach(k => { if (k !== 'none') scores[k] = (1 - scores.none) / 4 })
    }

    const pred = Object.entries(scores).reduce((a, b) => a[1] > b[1] ? a : b)
    predictions.push({
      window_start: ws,
      window_end: we,
      predicted: pred[0],
      confidence: +pred[1].toFixed(3),
      collusion_score: +(1 - scores.none).toFixed(3),
    })
  }

  return { id: 'synthetic', mode, duration, label_window: mode === 'none' ? null : [+wStart.toFixed(1), +wEnd.toFixed(1)], ticks, predictions }
}

export default function Simulator() {
  const [mode, setMode] = useState('spoof')
  const [episode, setEpisode] = useState(() => generateEpisode('spoof'))
  const [tick, setTick] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loading, setLoading] = useState(false)
  const [dataSource, setDataSource] = useState('synthetic') // 'real' | 'synthetic'
  const [manifest, setManifest] = useState(null)
  const rafRef = useRef(null)
  const lastTimeRef = useRef(0)

  const maxTick = episode.ticks.length - 1

  // Load manifest on mount
  useEffect(() => {
    fetch('/episodes/manifest.json')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data) {
          setManifest(data)
          // Auto-load first real episode
          loadRealEpisode(data, 'spoof')
        }
      })
      .catch(() => setManifest(null))
  }, [])

  async function loadRealEpisode(mf, m) {
    const episodes = mf?.[m]
    if (!episodes?.length) {
      setEpisode(generateEpisode(m))
      setDataSource('synthetic')
      return
    }

    setLoading(true)
    try {
      const ep = episodes[0]
      const res = await fetch(`/episodes/${ep.file}`)
      if (!res.ok) throw new Error()
      const data = await res.json()
      setEpisode(data)
      setDataSource('real')
    } catch {
      setEpisode(generateEpisode(m))
      setDataSource('synthetic')
    } finally {
      setLoading(false)
      setTick(0)
    }
  }

  const play = useCallback(() => {
    setPlaying(true)
    lastTimeRef.current = performance.now()
  }, [])

  const pause = useCallback(() => {
    setPlaying(false)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
  }, [])

  const reset = useCallback(() => {
    pause()
    setTick(0)
  }, [pause])

  const changeMode = useCallback((m) => {
    pause()
    setMode(m)
    setTick(0)
    if (manifest) {
      loadRealEpisode(manifest, m)
    } else {
      setEpisode(generateEpisode(m))
      setDataSource('synthetic')
    }
  }, [pause, manifest])

  useEffect(() => {
    if (!playing) return
    function step(now) {
      const dt = now - lastTimeRef.current
      if (dt > 1000 / (30 * speed)) {
        lastTimeRef.current = now
        setTick(prev => {
          if (prev >= maxTick) { setPlaying(false); return prev }
          return prev + 1
        })
      }
      rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [playing, maxTick, speed])

  const visibleTicks = episode.ticks.slice(0, tick + 1)
  const chartData = visibleTicks.map(t => ({ t: t.t, mid: t.mid }))

  const currentTime = episode.ticks[tick]?.t || 0
  const visiblePreds = episode.predictions.filter(p => p.window_end <= currentTime + 30)

  return (
    <PageWrapper>
      <section className="py-16 px-8">
        <div className="max-w-[1320px] mx-auto">
          <div className="section-label">
            <span className="num">02</span>
            <span>Live Simulator</span>
          </div>
          <div className="mb-10">
            <h2 className="mb-5 max-w-[800px]" style={{ fontSize: 'clamp(36px, 4.4vw, 56px)', fontVariationSettings: '"SOFT" 40, "opsz" 144', letterSpacing: '-0.03em' }}>
              Watch the market <span className="italic text-gold">unfold</span>.
            </h2>
            <p className="text-[17px] text-text-soft max-w-[640px] font-light">
              Pick a collusion mode, press play, and watch a 600-second episode replay tick-by-tick. The bottom band shows the CNN's confidence evolving in real time.
            </p>
          </div>

          <div className="grid lg:grid-cols-[1fr_320px] gap-6">
            {/* Main visualization */}
            <div className="space-y-4">
              {/* Price Chart */}
              <div className="panel">
                <div className="flex items-center justify-between mb-4">
                  <span className="eyebrow">Mid-Price + Order Flow</span>
                  <div className="flex items-center gap-3">
                    {dataSource === 'real' && (
                      <span className="font-mono text-[9px] tracking-widest uppercase px-2 py-0.5 border border-green/40 text-green bg-green-soft">
                        Real Data
                      </span>
                    )}
                    <span className="font-mono text-[11px] text-text-dim">
                      t = {currentTime.toFixed(0)}s / 600s
                    </span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" opacity={0.5} />
                    <XAxis dataKey="t" tick={{ fontSize: 10, fill: 'var(--color-text-dim)' }} />
                    <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: 'var(--color-text-dim)' }} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-bg-elev)', border: '1px solid var(--color-border)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                      labelStyle={{ color: 'var(--color-text-muted)' }}
                    />
                    <Line type="monotone" dataKey="mid" stroke="var(--color-text-muted)" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
                {episode.label_window && (
                  <div className="text-[10px] font-mono text-text-dim mt-2">
                    Collusion window: {episode.label_window[0].toFixed(0)}s — {episode.label_window[1].toFixed(0)}s
                  </div>
                )}
              </div>

              {/* Detector Score Band */}
              <div className="panel">
                <div className="flex items-center justify-between mb-4">
                  <span className="eyebrow">Detector Confidence</span>
                  <span className="font-mono text-[10px] text-text-dim">CNN sliding-window predictions</span>
                </div>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={visiblePreds} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-soft)" opacity={0.3} />
                    <XAxis dataKey="window_start" tick={{ fontSize: 9, fill: 'var(--color-text-dim)' }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: 'var(--color-text-dim)' }} />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-bg-elev)', border: '1px solid var(--color-border)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}
                      formatter={(v, n) => [v.toFixed(3), n]}
                    />
                    <Bar dataKey="collusion_score" radius={[2, 2, 0, 0]}>
                      {visiblePreds.map((p, i) => (
                        <Cell key={i} fill={MODE_COLORS[p.predicted] || '#4cc38a'} opacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                {/* Legend */}
                <div className="flex items-center gap-4 mt-2 flex-wrap">
                  {Object.entries(MODE_COLORS).map(([k, c]) => (
                    <span key={k} className="flex items-center gap-1.5 font-mono text-[10px] text-text-dim uppercase tracking-wider">
                      <span className="w-3 h-1.5 rounded-sm" style={{ background: c }} />
                      {k}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Controls Panel */}
            <div className="bg-bg-elev border border-border p-7 relative flex flex-col gap-6 h-fit">
              <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-gold" />
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-gold" />

              {/* Mode selector */}
              <div>
                <div className="font-mono text-[10px] tracking-widest uppercase text-text-muted mb-3">Collusion Mode</div>
                <div className="grid grid-cols-3 gap-1.5">
                  {MODES.map(m => (
                    <button key={m}
                      onClick={() => changeMode(m)}
                      disabled={loading}
                      className={`px-2 py-2.5 border font-mono text-[10.5px] tracking-wider uppercase cursor-pointer transition-all duration-160
                        ${mode === m
                          ? 'border-gold bg-gold-glow text-gold'
                          : 'border-border text-text-soft hover:border-gold-line hover:text-gold'}`}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              {/* Speed */}
              <div>
                <div className="font-mono text-[10px] tracking-widest uppercase text-text-muted mb-3">Speed</div>
                <div className="flex gap-1.5">
                  {[1, 2, 5, 10].map(s => (
                    <button key={s}
                      onClick={() => setSpeed(s)}
                      className={`flex-1 px-2 py-2 border font-mono text-[10.5px] cursor-pointer transition-all
                        ${speed === s ? 'border-gold bg-gold-glow text-gold' : 'border-border text-text-soft hover:border-gold-line'}`}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Playback controls */}
              <div className="flex gap-2">
                <button onClick={playing ? pause : play}
                  disabled={loading}
                  className="flex-1 bg-gold text-bg-deep border-none py-4 font-mono text-[12px] font-semibold tracking-widest uppercase cursor-pointer flex items-center justify-center gap-3"
                  style={{ boxShadow: '0 8px 30px var(--color-gold-glow)' }}>
                  {loading ? 'LOADING...' : playing ? '⏸ Pause' : '▶ Play'}
                </button>
                <button onClick={reset}
                  className="px-4 py-4 border border-border text-text-soft font-mono text-[12px] tracking-wider uppercase cursor-pointer hover:border-gold-line hover:text-gold transition-all">
                  ↻
                </button>
              </div>

              {/* Live telemetry */}
              <div className="stat-grid">
                <div><div className="l">Sim Time</div><div className="v">{currentTime.toFixed(0)}s</div></div>
                <div><div className="l">Events</div><div className="v">{(tick * 6).toLocaleString()}</div></div>
                <div><div className="l">Mode</div><div className="v" style={{ color: MODE_COLORS[mode] }}>{mode.toUpperCase()}</div></div>
                <div><div className="l">Status</div><div className="v" style={{ color: playing ? 'var(--color-green)' : 'var(--color-text-muted)' }}>{playing ? 'LIVE' : 'IDLE'}</div></div>
              </div>

              {/* Episode info */}
              {episode.id && episode.id !== 'synthetic' && (
                <div className="pt-4 border-t border-border-soft">
                  <div className="font-mono text-[10px] tracking-widest uppercase text-text-muted mb-2">Episode</div>
                  <div className="font-mono text-[12px] text-gold">{episode.id}</div>
                  {episode.episode_prediction && (
                    <div className="font-mono text-[10px] text-text-dim mt-1">
                      CNN verdict: <span style={{ color: MODE_COLORS[episode.episode_prediction] }}>{episode.episode_prediction}</span> ({(episode.episode_confidence * 100).toFixed(1)}%)
                    </div>
                  )}
                </div>
              )}

              {/* Progress */}
              <div>
                <div className="h-1 bg-bg-deep border border-border-soft relative">
                  <div className="absolute top-0 left-0 bottom-0 bg-gold transition-all duration-100"
                       style={{ width: `${(tick / maxTick) * 100}%` }} />
                </div>
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
                The detector catches manipulation in real time — confidence bars spike red during the scheme window and stay green outside it. No human had to label the order flow.
              </p>
            </div>
            <Link to="/detector" className="btn-ghost">
              <span>How the CNN works →</span>
            </Link>
          </motion.div>
        </div>
      </section>
    </PageWrapper>
  )
}
