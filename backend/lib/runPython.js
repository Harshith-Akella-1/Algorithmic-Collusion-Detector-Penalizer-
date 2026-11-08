import { spawn } from 'child_process'
import path from 'path'

/**
 * Run predict.py on the given CSV file and return parsed JSON results.
 * @param {string} csvPath - absolute path to the uploaded CSV
 * @returns {Promise<object>} - parsed prediction result
 */
export function runPredict(csvPath) {
  return new Promise((resolve, reject) => {
    const projectRoot = path.resolve(import.meta.dirname, '..', '..')
    const predictScript = path.join(projectRoot, 'predict.py')
    const modelPath = path.join(projectRoot, 'detectors', 'cnn_best.pt')

    const args = [
      predictScript,
      '--orders', csvPath,
      '--model', modelPath,
      '--json',
    ]

    const proc = spawn('python', args, {
      cwd: projectRoot,
      timeout: 30000,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (chunk) => { stdout += chunk.toString() })
    proc.stderr.on('data', (chunk) => { stderr += chunk.toString() })

    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`predict.py exited with code ${code}: ${stderr}`))
      }
      try {
        // Try to parse stdout as JSON
        const result = JSON.parse(stdout)
        resolve(result)
      } catch {
        // If predict.py doesn't output JSON natively, parse the text output
        resolve(parsePredictOutput(stdout))
      }
    })

    proc.on('error', (err) => {
      reject(new Error(`Failed to spawn predict.py: ${err.message}`))
    })
  })
}

/**
 * Parse the text output from predict.py into a structured result.
 * predict.py outputs lines like:
 *   Episode-level prediction: spoof (confidence: 0.992)
 *   Window 0-60: none (0.134)
 *   ...
 */
function parsePredictOutput(text) {
  const lines = text.trim().split('\n')
  const result = {
    episode_prediction: 'none',
    confidence: 0,
    windows_total: 0,
    windows_flagged: 0,
    windows: [],
  }

  for (const line of lines) {
    // Episode-level line
    const epMatch = line.match(/Episode.*?prediction:\s*(\w+)\s*\(.*?([\d.]+)/)
    if (epMatch) {
      result.episode_prediction = epMatch[1]
      result.confidence = parseFloat(epMatch[2])
      continue
    }

    // Window lines
    const winMatch = line.match(/[Ww]indow\s+(\d+)-(\d+):\s*(\w+)\s*\(([\d.]+)\)/)
    if (winMatch) {
      const predicted = winMatch[3]
      const score = parseFloat(winMatch[4])
      result.windows.push({
        start: parseInt(winMatch[1]),
        end: parseInt(winMatch[2]),
        predicted,
        collusion_score: predicted !== 'none' ? score : 1 - score,
      })
    }
  }

  result.windows_total = result.windows.length
  result.windows_flagged = result.windows.filter(w => w.predicted !== 'none').length
  return result
}
