import { spawn } from 'node:child_process'
import path, { dirname } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const root = path.resolve(__dirname, '..') // tgTrax/
const parent = path.resolve(root, '..')    // telegram/

const PY = process.env.PYTHON || 'python3'
const args = process.argv.slice(2)
const env = { ...process.env, PYTHONPATH: `${parent}:${process.env.PYTHONPATH || ''}` }

const proc = spawn(PY, ['-m', 'tgTrax.scripts.auth_console', ...args], {
  cwd: parent,
  env,
  stdio: 'inherit',
})
proc.on('exit', (code) => process.exit(code ?? 0))

