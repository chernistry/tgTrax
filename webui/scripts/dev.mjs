import { spawn } from 'node:child_process'
import path, { dirname } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import net from 'node:net'

// Paths
const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const projectRoot = path.resolve(__dirname, '../../') // tgTrax/
const repoRoot = path.resolve(projectRoot, '..') // telegram/
const webuiRoot = path.resolve(__dirname, '..')

// Config
const PYTHON = process.env.PYTHON || 'python3'
const API_HOST = process.env.FLASK_HOST || '127.0.0.1'
let API_PORT = process.env.FLASK_PORT || '8050'
let NEXT_PORT = process.env.NEXT_PORT || '3000'

async function findFreePort(port) {
  const p = parseInt(port, 10)
  for (let candidate = p; candidate < p + 10; candidate++) {
    const used = await new Promise((resolve) => {
      const s = net.createServer()
      s.once('error', () => resolve(true))
      s.once('listening', () => s.close(() => resolve(false)))
      s.listen(candidate, '127.0.0.1')
    })
    if (!used) return String(candidate)
  }
  return String(p)
}

API_PORT = await findFreePort(API_PORT)
NEXT_PORT = await findFreePort(NEXT_PORT)

// Spawn Flask dashboard (backend API)
console.log(`[dev] Starting Flask API at http://${API_HOST}:${API_PORT} ...`)
const flaskEnv = { 
  ...process.env, 
  PYTHONPATH: `${repoRoot}:${process.env.PYTHONPATH || ''}`,
  FLASK_HOST: API_HOST,
  FLASK_PORT: API_PORT,
}
const flask = spawn(PYTHON, ['-m', 'tgTrax.main', 'dashboard'], {
  cwd: repoRoot,
  env: flaskEnv,
  stdio: 'inherit',
})

flask.on('exit', (code) => {
  console.log(`[dev] Flask process exited with code ${code}`)
})

// Small delay to let Flask bind port before starting Next.js
setTimeout(async () => {
  console.log(`[dev] Starting Next.js UI at http://localhost:${NEXT_PORT} ...`)
  const nextEnv = { ...process.env, NEXT_PUBLIC_TGTRAX_API: `http://${API_HOST}:${API_PORT}` }
  // Resolve Next.js bin reliably in ESM
  const { createRequire } = await import('node:module')
  const require = createRequire(import.meta.url)
  const nextBin = require.resolve('next/dist/bin/next')
  const next = spawn(process.execPath, [nextBin, 'dev', '-p', NEXT_PORT], {
    cwd: webuiRoot,
    env: nextEnv,
    stdio: 'inherit',
  })
  next.on('exit', (code) => {
    console.log(`[dev] Next.js exited with code ${code}`)
    // If UI exits, keep Flask until terminated manually
  })
}, 800)
