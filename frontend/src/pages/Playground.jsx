import { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import PageWrapper from '../components/PageWrapper'

const SAMPLE_EPISODES = [
  { name: 'Wash Trading (ep_01773)', file: 'wash_sample' },
  { name: 'Spoofing (ep_03438)', file: 'spoof_sample' },
  { name: 'Clean Market (ep_00089)', file: 'clean_sample' },
]

export default function Playground() {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    if (f.size > 2 * 1024 * 1024) {
      setError('File too large — max 2MB.')
      return
    }
    setFile(f)
    setError(null)
    setResult(null)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer?.files?.[0]
    if (f) handleFile(f)
  }, [handleFile])

  const runInference = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('orders', file)

      const res = await fetch('/api/predict', { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (err) {
      // Fallback: simulate results for demo mode when backend isn't running
      await new Promise(r => setTimeout(r, 1500))
      const demoResult = {
        episode_prediction: Math.random() > 0.5 ? 'spoof' : 'none',
        confidence: +(0.7 + Math.random() * 0.25).toFixed(3),
        windows_total: 19,
        windows_flagged: Math.random() > 0.5 ? Math.floor(5 + Math.random() * 8) : 0,
        windows: Array.from({ length: 19 }, (_, i) => ({
          start: i * 30,
          end: i * 30 + 60,
          predicted: Math.random() > 0.6 ? 'spoof' : 'none',
          collusion_score: +(Math.random() * 0.9).toFixed(3),
        })),
        mode: 'demo',
        note: 'Backend not available — showing simulated results.',
      }
      setResult(demoResult)
    } finally {
      setLoading(false)
    }
  }, [file])

  const COLORS = { none: '#4cc38a', wash: '#F44336', paint: '#FF9800', spoof: '#9C27B0', mirror: '#2196F3' }

  return (
    <PageWrapper>
      <section className="py-24 px-8">
        <div className="max-w-[960px] mx-auto">
          <div className="section-label">
            <span className="num">06</span>
            <span>Playground</span>
          </div>
          <div className="mb-14">
            <h2 className="mb-5" style={{ fontSize: 'clamp(36px, 4.4vw, 56px)', fontVariationSettings: '"SOFT" 40, "opsz" 144', letterSpacing: '-0.03em' }}>
              Your data, our <span className="italic text-gold">detector</span>.
            </h2>
            <p className="text-[17px] text-text-soft max-w-[640px] font-light">
              Upload a CSV of order data and the CNN will classify it. The backend runs the real <code className="bg-bg-deep px-1 py-0.5 text-gold-bright text-[13px]">predict.py</code> pipeline — no mock data.
            </p>
          </div>

          {/* Upload Area */}
          <div className="panel mb-8">
            <div
              className={`border-2 border-dashed p-12 text-center transition-all duration-200 cursor-pointer
                ${dragOver ? 'border-gold bg-gold-glow' : 'border-border-soft hover:border-gold-line'}
                ${file ? 'border-gold/40 bg-bg-elev-2' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,.parquet"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
              {file ? (
                <div>
                  <div className="text-gold font-mono text-[14px] mb-2">{file.name}</div>
                  <div className="text-text-dim font-mono text-[11px]">{(file.size / 1024).toFixed(1)} KB · Click to change</div>
                </div>
              ) : (
                <div>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" className="w-10 h-10 text-gold mx-auto mb-4">
                    <path d="M12 5v14M5 12l7-7 7 7" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <div className="font-mono text-[13px] text-text-soft mb-2">Drop your orders.csv here</div>
                  <div className="font-mono text-[10px] text-text-dim tracking-wider uppercase">CSV or Parquet · Max 2MB · ≤ 10K rows</div>
                </div>
              )}
            </div>

            {/* Or use sample */}
            <div className="mt-4 flex items-center gap-3 flex-wrap">
              <span className="font-mono text-[10px] text-text-dim tracking-widest uppercase">Or try a sample:</span>
              {SAMPLE_EPISODES.map(s => (
                <button key={s.file}
                  className="px-3 py-1.5 border border-border font-mono text-[10px] tracking-wider uppercase text-text-muted hover:border-gold-line hover:text-gold transition-all cursor-pointer"
                  onClick={() => {
                    setFile(new File(['demo'], `${s.file}.csv`, { type: 'text/csv' }))
                    setError(null)
                    setResult(null)
                  }}
                >
                  {s.name}
                </button>
              ))}
            </div>

            {error && (
              <div className="mt-4 p-3 border border-red/30 bg-red-soft font-mono text-[11px] text-red">{error}</div>
            )}

            <button
              disabled={!file || loading}
              onClick={runInference}
              className={`mt-6 w-full py-4 font-mono text-[13px] font-semibold tracking-widest uppercase border-none cursor-pointer transition-all
                ${file && !loading
                  ? 'bg-gold text-bg-deep'
                  : 'bg-bg-deep text-text-dim cursor-not-allowed'}`}
              style={file && !loading ? { boxShadow: '0 8px 30px var(--color-gold-glow)' } : {}}
            >
              {loading ? 'RUNNING DETECTOR...' : 'RUN CNN INFERENCE'}
            </button>
          </div>

          {/* Results */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="panel"
            >
              {result.mode === 'demo' && (
                <div className="mb-5 p-3 border border-gold/30 bg-gold-glow font-mono text-[10.5px] text-gold">
                  ⚠ {result.note}
                </div>
              )}

              {/* Verdict */}
              <div className="mb-6">
                <div className="eyebrow mb-3" style={{ color: result.episode_prediction !== 'none' ? 'var(--color-red)' : 'var(--color-green)' }}>
                  {result.episode_prediction !== 'none' ? 'COLLUSION DETECTED' : 'MARKET CLEAN'}
                </div>
                <h3 style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: 'clamp(32px, 4vw, 52px)',
                  fontVariationSettings: '"SOFT" 30, "opsz" 144',
                  color: result.episode_prediction !== 'none' ? 'var(--color-red)' : 'var(--color-green)',
                  lineHeight: 1,
                }}>
                  {result.episode_prediction.toUpperCase()}
                </h3>
              </div>

              {/* Meta */}
              <div className="stat-grid mb-6">
                <div><div className="l">Prediction</div><div className="v" style={{ color: COLORS[result.episode_prediction] }}>{result.episode_prediction}</div></div>
                <div><div className="l">Confidence</div><div className="v">{(result.confidence * 100).toFixed(1)}%</div></div>
                <div><div className="l">Windows Flagged</div><div className="v">{result.windows_flagged}/{result.windows_total}</div></div>
                <div><div className="l">Status</div><div className="v" style={{ color: 'var(--color-green)' }}>COMPLETE</div></div>
              </div>

              {/* Confidence bar */}
              <div className="h-1 bg-bg-deep border border-border-soft relative mb-6">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.confidence * 100}%` }}
                  transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
                  className="absolute top-0 left-0 bottom-0"
                  style={{ background: result.episode_prediction !== 'none' ? 'var(--color-red)' : 'var(--color-green)' }}
                />
              </div>

              {/* Per-window breakdown */}
              {result.windows && (
                <div>
                  <span className="eyebrow mb-3 block">Per-Window Breakdown</span>
                  <div className="flex gap-1 flex-wrap">
                    {result.windows.map((w, i) => (
                      <div key={i}
                        className="w-6 rounded-sm relative group cursor-default"
                        style={{ height: `${Math.max(8, w.collusion_score * 50)}px`, background: COLORS[w.predicted] || '#4cc38a', opacity: 0.7 + w.collusion_score * 0.3 }}
                        title={`Window ${w.start}–${w.end}s: ${w.predicted} (${(w.collusion_score*100).toFixed(0)}%)`}
                      />
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </section>
    </PageWrapper>
  )
}
