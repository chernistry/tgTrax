import { spawn } from 'node:child_process'
import path, { dirname } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'
import net from 'node:net'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const root = path.resolve(__dirname, '..') // tgTrax/
const parent = path.resolve(root, '..')    // telegram/
const webui = path.resolve(root, 'webui')

let API_PORT = process.env.FLASK_PORT || '8050'
let NEXT_PORT = process.env.NEXT_PORT || '3000'

async function findFreePort(port) {
  const p = parseInt(port, 10)
  for (let c = p; c < p + 20; c++) {
    const used = await new Promise((resolve) => {
      const s = net.createServer()
      s.once('error', () => resolve(true))
      s.once('listening', () => s.close(() => resolve(false)))
      s.listen(c, '127.0.0.1')
    })
    if (!used) return String(c)
  }
  return String(p)
}

API_PORT = await findFreePort(API_PORT)
NEXT_PORT = await findFreePort(NEXT_PORT)

console.log(`[dev] Flask API http://127.0.0.1:${API_PORT}`)
const flaskEnv = {
  ...process.env,
  PYTHONPATH: `${parent}:${process.env.PYTHONPATH || ''}`,
  FLASK_HOST: '127.0.0.1',
  FLASK_PORT: API_PORT,
}
const flask = spawn(process.env.PYTHON || 'python3', ['-m', 'tgTrax.main', 'dashboard'], {
  cwd: parent,
  env: flaskEnv,
  stdio: 'inherit',
})
flask.on('exit', (code) => console.log(`[dev] Flask exit code ${code}`))

setTimeout(async () => {
  console.log(`[dev] Next.js http://localhost:${NEXT_PORT}`)
  const { createRequire } = await import('node:module')
  const require = createRequire(path.join(webui, 'package.json'))
  const nextBin = require.resolve('next/dist/bin/next')
  const nextEnv = { ...process.env, NEXT_PUBLIC_TGTRAX_API: `http://127.0.0.1:${API_PORT}` }
  const next = spawn(process.execPath, [nextBin, 'dev', '-p', NEXT_PORT], {
    cwd: webui,
    env: nextEnv,
    stdio: 'inherit',
  })
  next.on('exit', (code) => console.log(`[dev] Next exit code ${code}`))
}, 600)

