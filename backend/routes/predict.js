import { Router } from 'express'
import multer from 'multer'
import path from 'path'
import fs from 'fs'
import { runPredict } from '../lib/runPython.js'

const router = Router()
const upload = multer({
  dest: path.resolve(import.meta.dirname, '..', 'uploads'),
  limits: { fileSize: 2 * 1024 * 1024 }, // 2MB
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase()
    if (['.csv', '.parquet'].includes(ext)) {
      cb(null, true)
    } else {
      cb(new Error('Only CSV and Parquet files are accepted'))
    }
  },
})

router.post('/', upload.single('orders'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' })
  }

  const csvPath = req.file.path
  try {
    const result = await runPredict(csvPath)
    res.json(result)
  } catch (err) {
    console.error('Prediction error:', err.message)
    res.status(500).json({ error: err.message })
  } finally {
    // Clean up uploaded file
    try { fs.unlinkSync(csvPath) } catch {}
  }
})

export default router
