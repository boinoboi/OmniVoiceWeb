import { fileByPath } from './manifest'
import { ensureCached } from './cache'
import { createCachedSession, releaseSession, type Device } from './ort'

export interface ProbeResult {
  device: Device
  inputNames: string[]
  outputNames: string[]
  modelMB: number
  outputDims: readonly number[]
  latencyMs: number
}

export async function runHeadsProbe(
  device: Device,
  onStatus?: (status: string) => void,
): Promise<ProbeResult> {
  const file = fileByPath('int4/audio_heads_decoder.onnx')
  if (!file) throw new Error('audio_heads_decoder is missing from the model manifest')

  onStatus?.('checking model cache…')
  await ensureCached([file])

  onStatus?.(`creating ${device.toUpperCase()} session…`)
  const handle = await createCachedSession(file.path, device)

  try {
    const seq = 16
    const hidden = 1024
    const data = new Float32Array(seq * hidden)
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 0.1
    const input = new handle.ort.Tensor('float32', data, [1, seq, hidden])

    onStatus?.('running forward pass…')
    const start = performance.now()
    const output = await handle.session.run({ hidden_states: input })
    const latencyMs = performance.now() - start

    return {
      device,
      inputNames: handle.inputNames,
      outputNames: handle.outputNames,
      modelMB: handle.modelBytes / 1_000_000,
      outputDims: output.logits.dims,
      latencyMs,
    }
  } finally {
    await releaseSession(handle)
  }
}
