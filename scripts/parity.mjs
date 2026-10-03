import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import * as ort from 'onnxruntime-web/wasm'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const goldenDir = join(root, 'tools', 'golden')
const modelsDir = join(root, 'tools', 'models')
const distDir = join(root, 'node_modules', 'onnxruntime-web', 'dist')

ort.env.wasm.wasmPaths = `${pathToFileURL(distDir).href}/`
ort.env.wasm.numThreads = 1
ort.env.logLevel = 'error'

const specs = JSON.parse(readFileSync(join(goldenDir, 'parity.json'), 'utf8'))

function readArray(file, dtype) {
  const buffer = readFileSync(join(goldenDir, file))
  const ab = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  if (dtype === 'float32') return new Float32Array(ab)
  if (dtype === 'int64') return new BigInt64Array(ab)
  return new Uint8Array(ab)
}

function product(shape) {
  return shape.reduce((a, b) => a * b, 1)
}

function makeTensor(desc) {
  if (desc.file) return new ort.Tensor(desc.dtype, readArray(desc.file, desc.dtype), desc.shape)
  const size = product(desc.shape)
  if (desc.zeros) {
    if (desc.dtype === 'int64') return new ort.Tensor('int64', new BigInt64Array(size), desc.shape)
    return new ort.Tensor('float32', new Float32Array(size), desc.shape)
  }
  if (desc.ones) {
    if (desc.dtype === 'int64') return new ort.Tensor('int64', new BigInt64Array(size).fill(1n), desc.shape)
    return new ort.Tensor('float32', new Float32Array(size).fill(1), desc.shape)
  }
  throw new Error(`unsupported tensor descriptor ${desc.name}`)
}

async function loadSession(rel) {
  const modelPath = join(modelsDir, rel)
  const model = readFileSync(modelPath)
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
  const dataPath = `${modelPath}.data`
  if (existsSync(dataPath)) {
    options.externalData = [
      { path: `${rel.split('/').pop()}.data`, data: new Uint8Array(readFileSync(dataPath)) },
    ]
  }
  const buffer = model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength)
  return ort.InferenceSession.create(buffer, options)
}

function compare(goldenFile, tensor) {
  const gold = readArray(goldenFile, 'float32')
  const got = tensor.data
  let max = 0
  let sum = 0
  let goldMax = 0
  let dot = 0
  let normGold = 0
  let normGot = 0
  for (let i = 0; i < gold.length; i++) {
    const d = Math.abs(gold[i] - got[i])
    if (d > max) max = d
    sum += d
    if (Math.abs(gold[i]) > goldMax) goldMax = Math.abs(gold[i])
    dot += gold[i] * got[i]
    normGold += gold[i] * gold[i]
    normGot += got[i] * got[i]
  }
  const cosine = dot / (Math.sqrt(normGold) * Math.sqrt(normGot) + 1e-12)
  return { max, mean: sum / gold.length, goldMax, relMax: max / (goldMax + 1e-12), cosine }
}

function argmaxAgreement(goldenFile, dims, tensor) {
  const gold = readArray(goldenFile, 'float32')
  const got = tensor.data
  const last = dims[dims.length - 1]
  const rows = gold.length / last
  let agree = 0
  for (let r = 0; r < rows; r++) {
    let gi = 0
    let oi = 0
    let gv = -Infinity
    let ov = -Infinity
    for (let c = 0; c < last; c++) {
      const idx = r * last + c
      if (gold[idx] > gv) { gv = gold[idx]; gi = c }
      if (got[idx] > ov) { ov = got[idx]; oi = c }
    }
    if (gi === oi) agree++
  }
  return agree / rows
}

const REL_TOL = Number(process.env.PARITY_REL_TOL ?? 1e-2)
let failed = false

for (const spec of specs.cases) {
  const session = await loadSession(spec.model)
  const feeds = {}
  for (const input of spec.inputs) feeds[input.name] = makeTensor(input)
  const outputNames = spec.outputs.map((o) => o.name)
  const results = await session.run(feeds, outputNames)

  console.log(`\n[${spec.model}]`)
  for (const output of spec.outputs) {
    const res = results[output.name]
    if (output.dtype === 'int64') {
      const gold = readArray(output.file, 'int64')
      const got = res.data
      let equal = 0
      for (let i = 0; i < gold.length; i++) if (gold[i] === got[i]) equal++
      const rate = equal / gold.length
      const ok = rate >= 0.999
      if (!ok) failed = true
      console.log(
        `  ${output.name.padEnd(16)} shape=${JSON.stringify(res.dims)}  exact-match=${(rate * 100).toFixed(2)}%  ${ok ? 'PASS' : 'FAIL'}`,
      )
      continue
    }
    const { max, relMax, cosine } = compare(output.file, res)
    const kind = spec.outputs.length === 1 && res.dims.length === 4 ? 'logits' : null
    const agree = kind ? argmaxAgreement(output.file, res.dims, res) : null
    const ok = relMax <= REL_TOL || cosine >= 0.999 || (agree !== null && agree >= 0.999)
    if (!ok) failed = true
    const extra = agree !== null ? `  argmax=${(agree * 100).toFixed(2)}%` : ''
    console.log(
      `  ${output.name.padEnd(16)} shape=${JSON.stringify(res.dims)}  max|Δ|=${max.toExponential(2)}  rel=${relMax.toExponential(2)}  cos=${cosine.toFixed(5)}${extra}  ${ok ? 'PASS' : 'FAIL'}`,
    )
  }
}

if (failed) {
  console.error(`\nPARITY FAILED (rel_tol ${REL_TOL})`)
  process.exit(1)
}
console.log(`\nPARITY PASSED (rel_tol ${REL_TOL})`)
