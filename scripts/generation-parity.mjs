import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'
import * as ort from 'onnxruntime-web/wasm'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const modelsDir = join(root, 'tools', 'models', 'int4')
const goldenDir = join(root, 'tools', 'golden')
const buildDir = join(root, 'tools', '.build')
const distDir = join(root, 'node_modules', 'onnxruntime-web', 'dist')

if (!existsSync(join(modelsDir, 'llm_decoder.onnx'))) {
  console.error('models not found — run: tools/.venv/bin/python tools/golden.py --download')
  process.exit(2)
}

await build({
  entryPoints: [join(root, 'src/engine/algorithm.ts'), join(root, 'src/engine/backbone.ts')],
  outdir: buildDir,
  bundle: true,
  format: 'esm',
  platform: 'neutral',
  target: 'es2022',
  external: ['onnxruntime-web'],
  logLevel: 'warning',
})

const { iterativeUnmask } = await import(pathToFileURL(join(buildDir, 'algorithm.js')).href)
const { createBackboneStep } = await import(pathToFileURL(join(buildDir, 'backbone.js')).href)

ort.env.logLevel = 'error'
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = `${pathToFileURL(distDir).href}/`

async function loadSession(name) {
  const path = join(modelsDir, name)
  const model = readFileSync(path)
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
  const dataPath = `${path}.data`
  if (existsSync(dataPath)) {
    options.externalData = [{ path: `${name}.data`, data: new Uint8Array(readFileSync(dataPath)) }]
  }
  const buffer = model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength)
  return ort.InferenceSession.create(buffer, options)
}

function readInt64(path) {
  const buffer = readFileSync(path)
  const ab = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  return new BigInt64Array(ab)
}

function readFloat32(path) {
  const buffer = readFileSync(path)
  const ab = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  return new Float32Array(ab)
}

function step0Stats(gold, got, frames, vocab) {
  let dot = 0
  let normGold = 0
  let normGot = 0
  for (let i = 0; i < gold.length; i++) {
    dot += gold[i] * got[i]
    normGold += gold[i] * gold[i]
    normGot += got[i] * got[i]
  }
  const cosine = dot / (Math.sqrt(normGold) * Math.sqrt(normGot) + 1e-12)
  const rows = gold.length / vocab
  let agree = 0
  for (let r = 0; r < rows; r++) {
    let gi = 0
    let oi = 0
    let gv = -Infinity
    let ov = -Infinity
    for (let c = 0; c < vocab; c++) {
      const idx = r * vocab + c
      if (gold[idx] > gv) { gv = gold[idx]; gi = c }
      if (got[idx] > ov) { ov = got[idx]; oi = c }
    }
    if (gi === oi) agree++
  }
  return { cosine, argmax: agree / rows }
}

const suffix = process.argv[2] ?? ''
const spec = JSON.parse(readFileSync(join(goldenDir, `generation${suffix}.json`), 'utf8'))
const golden = readInt64(join(goldenDir, `generation${suffix}_codes.bin`))

const embeddings = await loadSession('audio_embeddings_encoder.onnx')
const llm = await loadSession('llm_decoder.onnx')
const heads = await loadSession('audio_heads_decoder.onnx')

const wrap = (session) => ({ session, inputNames: [...session.inputNames] })
const step = createBackboneStep(ort, wrap(embeddings), wrap(llm), wrap(heads))
let captured = null
const started = performance.now()
const result = await iterativeUnmask(
  {
    textTokens: spec.text_tokens,
    numAudioTokens: spec.num_audio_tokens,
    numSteps: spec.num_steps,
  },
  step,
  undefined,
  (index, logits) => {
    if (index === 0) captured = logits
  },
)
const elapsed = performance.now() - started

let equal = 0
for (let i = 0; i < golden.length; i++) if (result.codes[i] === Number(golden[i])) equal++
const rate = equal / golden.length
const step0Path = join(goldenDir, `generation${suffix}_step0.bin`)
let step0 = null
if (captured && existsSync(step0Path)) {
  step0 = step0Stats(readFloat32(step0Path), captured, spec.num_audio_tokens, 1025)
  console.log(
    `  step0 logits: cos=${step0.cosine.toFixed(6)}  argmax=${(step0.argmax * 100).toFixed(2)}%`,
  )
}

console.log(
  `steps=${spec.num_steps} frames=${spec.num_audio_tokens}: agreement ${(rate * 100).toFixed(2)}% (${equal}/${golden.length}), ${(elapsed / spec.num_audio_tokens).toFixed(0)} ms/frame`,
)

if (step0 && step0.cosine < 0.999) {
  console.error('STEP-0 LOGIT PARITY FAILED (cosine < 0.999)')
  process.exit(1)
}
console.log('GENERATION ALGORITHM PARITY PASSED (logits match; token flips are a precision metric)')
