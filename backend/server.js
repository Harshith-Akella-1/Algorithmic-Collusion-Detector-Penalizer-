import express from 'express'
import cors from 'cors'
import path from 'path'
import predictRouter from './routes/predict.js'

const app = express()
const PORT = process.env.PORT || 3001

// Middleware
app.use(cors())
app.use(express.json())

// Serve pre-rendered episodes (static JSON)
const episodesDir = path.resolve(import.meta.dirname, '..', 'frontend', 'public', 'episodes')
app.use('/api/episodes', express.static(episodesDir))

// Prediction endpoint (CNN)
app.use('/api/predict', predictRouter)

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    python: 'available',
    model: 'cnn_best.pt',
  })
})

// Error handler
app.use((err, req, res, next) => {
  console.error('Error:', err.message)
  res.status(err.status || 500).json({ error: err.message })
})

app.listen(PORT, () => {
  console.log(`\n  ⚡ Algo·Collusion Backend`)
  console.log(`  → http://localhost:${PORT}`)
  console.log(`  → /api/predict   POST (upload CSV, CNN)`)
  console.log(`  → /api/health    GET`)
  console.log(`  → /api/episodes  GET (static JSON)\n`)
})
