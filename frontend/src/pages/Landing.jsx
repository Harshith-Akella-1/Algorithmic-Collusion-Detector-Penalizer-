import { useRef, useMemo, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { motion } from 'framer-motion'
import PageWrapper from '../components/PageWrapper'

/* ── 3D Particle Field ───────────────────────────────────── */
function ParticleField() {
  const count = 2000
  const meshRef = useRef()
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const particles = useMemo(() => {
    const arr = []
    for (let i = 0; i < count; i++) {
      arr.push({
        x: (Math.random() - 0.5) * 20,
        y: (Math.random() - 0.5) * 12,
        z: (Math.random() - 0.5) * 10,
        speed: 0.002 + Math.random() * 0.008,
        phase: Math.random() * Math.PI * 2,
        scale: 0.02 + Math.random() * 0.04,
      })
    }
    return arr
  }, [])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    particles.forEach((p, i) => {
      const x = p.x + Math.sin(t * p.speed * 20 + p.phase) * 0.3
      const y = p.y + Math.cos(t * p.speed * 15 + p.phase) * 0.2
      const z = p.z + Math.sin(t * p.speed * 10) * 0.1
      dummy.position.set(x, y, z)
      dummy.scale.setScalar(p.scale * (1 + Math.sin(t * 2 + p.phase) * 0.3))
      dummy.updateMatrix()
      meshRef.current.setMatrixAt(i, dummy.matrix)
    })
    meshRef.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial color="#e9c46a" transparent opacity={0.6} />
    </instancedMesh>
  )
}

function PriceWave() {
  const lineRef = useRef()
  const pointCount = 200

  const positions = useMemo(() => new Float32Array(pointCount * 3), [])
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return geo
  }, [positions])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    for (let i = 0; i < pointCount; i++) {
      const x = (i / pointCount) * 16 - 8
      const phase = t * 0.5
      // Create realistic-looking price action
      const y = Math.sin(x * 0.5 + phase) * 0.8
                + Math.sin(x * 1.2 + phase * 1.7) * 0.4
                + Math.sin(x * 3 + phase * 0.3) * 0.15
                + (x > 2 && x < 5 ? Math.sin(t * 3) * 0.6 : 0) // spoofing burst
      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = 0
    }
    geometry.attributes.position.needsUpdate = true
  })

  return (
    <line ref={lineRef} geometry={geometry}>
      <lineBasicMaterial color="#e9c46a" transparent opacity={0.8} linewidth={2} />
    </line>
  )
}

function Scene() {
  return (
    <>
      <ambientLight intensity={0.5} />
      <ParticleField />
      <PriceWave />
    </>
  )
}

/* ── Mock Terminal Feed ──────────────────────────────────── */
function TerminalFeed() {
  const [lines, setLines] = useState([])
  const feedRef = useRef(null)

  const TRADERS = ['N0','N1','N3','N5','N7','N9','MM','CA','CB']
  const TYPES = ['ORDER','ORDER','ORDER','TRADE','CANCEL']
  const SIDES = ['BUY','SELL']

  useEffect(() => {
    const initial = []
    for (let i = 0; i < 12; i++) {
      initial.push(generateLine(i * 0.1))
    }
    setLines(initial)

    let t = 1.2
    const interval = setInterval(() => {
      t += 0.1
      setLines(prev => {
        const next = [...prev, generateLine(t)]
        if (next.length > 16) next.shift()
        return next
      })
    }, 250)
    return () => clearInterval(interval)
  }, [])

  function generateLine(t) {
    const trader = TRADERS[Math.floor(Math.random() * TRADERS.length)]
    const type = TYPES[Math.floor(Math.random() * TYPES.length)]
    const side = SIDES[Math.floor(Math.random() * 2)]
    const px = (99.95 + Math.random() * 0.12).toFixed(2)
    const qty = Math.floor(10 + Math.random() * 90)
    const isBot = trader === 'CA' || trader === 'CB'
    return { t: t.toFixed(3), trader, type, side, px, qty, isBot, id: Math.random() }
  }

  return (
    <div className="relative bg-bg-elev border border-border h-[360px] overflow-hidden"
         style={{ boxShadow: '0 4px 24px rgba(0,0,0,0.6), 0 1px 2px rgba(0,0,0,0.4)' }}>
      {/* Terminal header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-soft bg-bg-deep"
           style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--color-text-muted)' }}>
        <span>BLACKROOK · LOB v2.6</span>
        <div className="flex gap-1.5">
          <span className="w-[7px] h-[7px] rounded-full bg-red/60" />
          <span className="w-[7px] h-[7px] rounded-full bg-gold/60" />
          <span className="w-[7px] h-[7px] rounded-full bg-green/60" />
        </div>
      </div>
      {/* Feed */}
      <div ref={feedRef} className="px-4 py-3 text-text-soft leading-[1.7]"
           style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px' }}>
        {lines.map(l => (
          <motion.div key={l.id}
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3 }}
                      className="flex gap-2.5">
            <span className="text-text-dim">{l.t}s</span>
            <span className={l.type === 'TRADE' ? 'text-green' : l.type === 'CANCEL' ? 'text-text-muted' : 'text-gold'}
                  style={{ width: '56px' }}>{l.type}</span>
            <span className={l.isBot ? 'text-gold-bright' : 'text-text'} style={{ width: '28px' }}>{l.trader}</span>
            <span className={l.side === 'BUY' ? 'text-green' : 'text-red'} style={{ width: '36px' }}>{l.side}</span>
            <span className="text-text" style={{ width: '56px', textAlign: 'right' }}>{l.px}</span>
            <span className="text-text-dim">{l.qty}</span>
          </motion.div>
        ))}
      </div>
      {/* Fade overlay */}
      <div className="absolute inset-0 pointer-events-none"
           style={{ background: 'linear-gradient(to bottom, var(--color-bg-elev) 0%, transparent 8%, transparent 80%, var(--color-bg-elev) 100%)' }} />
    </div>
  )
}

/* ── Landing Page ────────────────────────────────────────── */
export default function Landing() {
  return (
    <PageWrapper>
      {/* Hero Section */}
      <section className="relative overflow-hidden border-b border-border-soft"
               style={{ padding: '120px 0 72px' }}>
        {/* 3D Background */}
        <div className="absolute inset-0 opacity-40">
          <Canvas camera={{ position: [0, 0, 8], fov: 60 }} style={{ background: 'transparent' }}>
            <Scene />
          </Canvas>
        </div>

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 pointer-events-none opacity-40"
             style={{
               backgroundImage: 'linear-gradient(to right, var(--color-border-soft) 1px, transparent 1px), linear-gradient(to bottom, var(--color-border-soft) 1px, transparent 1px)',
               backgroundSize: '64px 64px',
               maskImage: 'radial-gradient(ellipse 80% 60% at 50% 50%, black, transparent 75%)',
               WebkitMaskImage: 'radial-gradient(ellipse 80% 60% at 50% 50%, black, transparent 75%)',
             }} />

        <div className="max-w-[1320px] mx-auto px-8 relative grid lg:grid-cols-[1.4fr_1fr] gap-16 items-center">
          {/* Left: Text */}
          <div>
            {/* Tag */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="eyebrow inline-flex items-center gap-2.5 mb-7"
            >
              <span className="w-6 h-px bg-gold" />
              <span>Market Surveillance</span>
              <span>·</span>
              <span>HFT Intelligence</span>
              <span className="w-1.5 h-1.5 rounded-full bg-gold ml-1"
                    style={{ animation: 'pulse 2.4s infinite' }} />
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              className="mb-8"
              style={{
                fontSize: 'clamp(48px, 7.4vw, 104px)',
                fontVariationSettings: '"SOFT" 50, "opsz" 144',
                letterSpacing: '-0.035em',
              }}
            >
              Can an AI catch<br/>
              trading bots{' '}
              <span className="italic text-gold relative"
                    style={{ fontVariationSettings: '"SOFT" 100, "opsz" 144' }}>
                colluding?
                <span className="absolute left-[-2%] right-[-2%] bottom-[0.05em] h-px bg-gold-line" />
              </span>
            </motion.h1>

            {/* Lede */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="text-[19px] leading-[1.55] text-text-soft max-w-[580px] mb-11 font-light"
            >
              We built a <strong className="text-text font-medium">market simulator</strong> with four kinds of trading fraud,
              trained a <strong className="text-text font-medium">1D-CNN</strong> to spot them, then proved it doesn't
              false-positive on legitimate <strong className="text-text font-medium">RL bots</strong>.
            </motion.p>

            {/* CTAs */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.7 }}
              className="flex gap-4 items-center flex-wrap"
            >
              <Link to="/concepts" className="btn-primary">
                <span>Start the Tour</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-3.5 h-3.5">
                  <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </Link>
              <Link to="/simulator" className="btn-ghost">
                <span>Jump to Simulator</span>
              </Link>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.9 }}
              className="grid grid-cols-2 sm:grid-cols-4 mt-[72px] pt-7 border-t border-border-soft"
            >
              {[
                ['5,000', 'Episodes\nGenerated'],
                ['95K+', 'Windows\nAnalyzed'],
                ['5', 'Collusion\nClasses'],
                ['1D-CNN', 'Detection\nBackbone'],
              ].map(([val, label], i) => (
                <div key={i} className={`px-5 ${i < 3 ? 'border-r border-border-soft' : ''} ${i === 0 ? 'pl-0' : ''}`}>
                  <div className="text-gold mb-2"
                       style={{
                         fontFamily: 'var(--font-display)',
                         fontSize: '38px',
                         fontVariationSettings: '"SOFT" 50, "opsz" 144',
                         lineHeight: 1,
                         letterSpacing: '-0.02em',
                       }}>{val}</div>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '10px',
                    letterSpacing: '0.18em',
                    textTransform: 'uppercase',
                    color: 'var(--color-text-muted)',
                    lineHeight: 1.4,
                    whiteSpace: 'pre-line',
                  }}>{label}</div>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right: Terminal */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="hidden lg:block"
          >
            <TerminalFeed />
          </motion.div>
        </div>
      </section>

      {/* Below fold: Story arc preview */}
      <section className="py-24 px-8">
        <div className="max-w-[1320px] mx-auto">
          <div className="section-label">
            <span className="num">THE STORY</span>
            <span>Three acts, one narrative</span>
          </div>

          <div className="grid md:grid-cols-3 gap-6 mt-12">
            {[
              {
                num: '01', title: 'Simulate', subtitle: 'Build the market',
                desc: 'A C++ limit order book hosts 15 noise traders, a market maker, and a colluder pair running one of four manipulation schemes.',
                link: '/concepts', cta: 'Learn the schemes →',
                color: 'var(--color-gold)',
              },
              {
                num: '02', title: 'Detect', subtitle: 'Train the watchdog',
                desc: 'A 25K-parameter 1D-CNN classifies 200-event windows into 5 classes — none, wash, paint, spoof, mirror — with 82.9% macro F1.',
                link: '/detector', cta: 'See the architecture →',
                color: 'var(--color-gold-bright)',
              },
              {
                num: '03', title: 'Validate', subtitle: 'The honesty proof',
                desc: 'Two independent RL agents trade the same book profitably — and the detector correctly labels every window as "none."',
                link: '/rl-honesty', cta: 'Read the proof →',
                color: 'var(--color-green)',
              },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.15 }}
                className="bg-bg-elev border border-border p-7 group hover:-translate-y-0.5 transition-all duration-280 relative overflow-hidden cursor-pointer"
              >
                <div className="absolute top-0 left-0 w-6 h-px transition-all duration-320 group-hover:w-full"
                     style={{ background: item.color }} />
                <div className="eyebrow mb-4">{item.num} / {item.subtitle.toUpperCase()}</div>
                <h3 className="text-[28px] text-text mb-4"
                    style={{ fontFamily: 'var(--font-display)', fontVariationSettings: '"SOFT" 50, "opsz" 144', letterSpacing: '-0.02em' }}>
                  {item.title}
                </h3>
                <p className="text-[13.5px] leading-[1.65] text-text-soft font-light mb-6">{item.desc}</p>
                <Link to={item.link} className="text-gold hover:text-gold-bright transition-colors"
                      style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', letterSpacing: '0.14em', textTransform: 'uppercase' }}>
                  {item.cta}
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </PageWrapper>
  )
}
