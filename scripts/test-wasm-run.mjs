import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = new URL('../public/ort/', import.meta.url).pathname
ort.env.logLevel = 'error'
const base = 'tools/models/bidir/llm_decoder_int4.onnx'
const session = await ort.InferenceSession.create(readFileSync(base), {
  externalData: [{ path: 'llm_decoder_int4.onnx.data', data: readFileSync(base + '.data') }],
})
console.log('session created; running one forward…')
const t = new ort.Tensor('float32', new Float32Array(8 * 1024).fill(0.01), [1, 8, 1024])
const out = await session.run({ inputs_embeds: t })
console.log('LLM forward OK, out dims', out.hidden_states.dims)
