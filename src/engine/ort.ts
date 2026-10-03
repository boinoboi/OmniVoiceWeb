import * as ort from 'onnxruntime-web/webgpu'
import { getModelBlob } from './cache'

export type Device = 'webgpu' | 'wasm'

let configured = false

export function configureOrt(): void {
  if (configured) return
  const base = import.meta.env.BASE_URL
  ort.env.wasm.wasmPaths = `${base}ort/`
  ort.env.wasm.numThreads = 1
  ort.env.wasm.simd = true
  ort.env.logLevel = 'warning'
  configured = true
}

export interface SessionHandle {
  session: ort.InferenceSession
  device: Device
  inputNames: string[]
  outputNames: string[]
  modelBytes: number
}

async function loadModel(modelPath: string): Promise<{ model: ArrayBuffer; bytes: number }> {
  const blob = await getModelBlob(modelPath)
  if (!blob) throw new Error(`Model is not cached: ${modelPath}`)
  return { model: await blob.arrayBuffer(), bytes: blob.size }
}

export async function createCachedSession(
  onnxPath: string,
  device: Device,
): Promise<SessionHandle> {
  configureOrt()

  const { model, bytes } = await loadModel(onnxPath)
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: device === 'webgpu' ? ['webgpu'] : ['wasm'],
    graphOptimizationLevel: 'all',
  }

  const dataBlob = await getModelBlob(`${onnxPath}.data`)
  if (dataBlob) {
    const baseName = onnxPath.split('/').pop() ?? onnxPath
    options.externalData = [
      { path: `${baseName}.data`, data: new Uint8Array(await dataBlob.arrayBuffer()) },
    ]
  }

  const session = await ort.InferenceSession.create(model, options)
  return {
    session,
    device,
    inputNames: [...session.inputNames],
    outputNames: [...session.outputNames],
    modelBytes: bytes,
  }
}

export async function releaseSession(handle: SessionHandle | null): Promise<void> {
  if (!handle) return
  try {
    await handle.session.release()
  } catch {
    void 0
  }
}

export { ort }
