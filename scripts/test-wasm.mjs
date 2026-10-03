import { readFileSync } from 'node:fs'
import * as ort from 'onnxruntime-web/wasm'

ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = new URL('../public/ort/', import.meta.url).pathname
ort.env.logLevel = 'error'

async function tryCreate(name, modelPath, external) {
  try {
    const model = readFileSync(modelPath)
    const options = {}
    if (external) {
      const base = modelPath.split('/').pop()
      options.externalData = [{ path: `${base}.data`, data: readFileSync(`${modelPath}.data`) }]
    }
    const session = await ort.InferenceSession.create(model, options)
    console.log(`OK    ${name}  inputs=${session.inputNames.join(',')}`)
    return session
  } catch (error) {
    console.log(`FAIL  ${name}  ${String(error.message).split('\n')[0]}`)
    return null
  }
}

await tryCreate('heads int4 (MatMulNBits)', 'tools/models/int4/audio_heads_decoder.onnx')
await tryCreate('embeddings int4 (GatherBlockQuantized)', 'tools/models/int4/audio_embeddings_encoder.onnx', true)
await tryCreate('embeddings fp32 (root)', 'tools/models/audio_embeddings_encoder.onnx', true)
await tryCreate('llm int4 bidir (MatMulNBits)', 'tools/models/bidir/llm_decoder_int4.onnx', true)
