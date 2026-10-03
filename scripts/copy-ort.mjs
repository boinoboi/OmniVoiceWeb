import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist')
const dest = join(root, 'public', 'ort')

if (!existsSync(src)) {
  console.warn('[copy-ort] onnxruntime-web dist not found, skipping')
  process.exit(0)
}

mkdirSync(dest, { recursive: true })

const wanted = /^ort-wasm-simd-threaded.*\.(wasm|mjs)$/
let count = 0
for (const file of readdirSync(src)) {
  if (!wanted.test(file)) continue
  cpSync(join(src, file), join(dest, file))
  count++
}

console.log(`[copy-ort] copied ${count} runtime files -> public/ort`)
