import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = new URL('../public/ort/', import.meta.url).pathname
ort.env.logLevel = 'error'
const base = 'tools/models/audio_embeddings_encoder.onnx'
const session = await ort.InferenceSession.create(readFileSync(base), {
  externalData: [{ path: 'audio_embeddings_encoder.onnx.data', data: readFileSync(base + '.data') }],
})
console.log('embeddings session created; running…')
const S = 12
const ids = new BigInt64Array(8 * S); ids.fill(1024n)
const mask = new Uint8Array(S); mask.fill(1)
const t0 = Date.now()
const out = await session.run({ input_ids: new ort.Tensor('int64', ids, [1, 8, S]), audio_mask: new ort.Tensor('bool', mask, [1, S]) })
console.log('embeddings forward OK', out.inputs_embeds.dims, Date.now() - t0, 'ms')
