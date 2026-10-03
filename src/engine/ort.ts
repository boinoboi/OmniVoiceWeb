import * as ortWebgpu from 'onnxruntime-web/webgpu'
import { getModelBlob } from './cache'

export type Device = 'webgpu' | 'wasm'
export type OrtModule = typeof ortWebgpu

const configured = new WeakSet<object>()
let wasmModule: OrtModule | undefined

export async function moduleFor(device: Device): Promise<OrtModule> {
  if (device === 'webgpu') return ortWebgpu
  if (!wasmModule) wasmModule = (await import('onnxruntime-web/wasm')) as unknown as OrtModule
  return wasmModule
}

interface NavigatorGpuLike {
  requestAdapter(options?: {
    powerPreference?: string
    forceFallbackAdapter?: boolean
    compatibilityMode?: boolean
    featureLevel?: string
  }): Promise<object | null>
}

export async function ensureWebGpuAdapter(module: OrtModule): Promise<void> {
  const gpu = (navigator as unknown as { gpu?: NavigatorGpuLike }).gpu
  if (!gpu) throw new Error('navigator.gpu is undefined')
  const env = module.env as unknown as { webgpu?: { adapter?: object } }
  env.webgpu = env.webgpu ?? {}
  if (env.webgpu.adapter) return

  const strategies: Array<[string, Parameters<NavigatorGpuLike['requestAdapter']>[0]]> = [
    ['default', undefined],
    ['high-performance', { powerPreference: 'high-performance' }],
    ['low-power', { powerPreference: 'low-power' }],
    ['featureLevel-compat', { featureLevel: 'compatibility' }],
    ['compatibilityMode-legacy', { compatibilityMode: true }],
    ['fallback', { forceFallbackAdapter: true }],
  ]
  const errors: string[] = []
  for (let round = 0; round < 3; round++) {
    for (const [name, options] of strategies) {
      try {
        const adapter = await gpu.requestAdapter(options)
        if (adapter) {
          env.webgpu.adapter = adapter
          return
        }
      } catch (error) {
        errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 300))
  }
  throw new Error(
    `No WebGPU adapter returned for any request (tried default/high-power/low-power/compatibility/fallback). ${errors.slice(-2).join('; ')}`,
  )
}

export function configureOrt(module: OrtModule): void {
  if (configured.has(module)) return
  module.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
  module.env.wasm.numThreads = isolated
    ? Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1))
    : 1
  module.env.wasm.simd = true
  module.env.logLevel = 'warning'
  configured.add(module)
}

export interface SessionHandle {
  session: ortWebgpu.InferenceSession
  device: Device
  inputNames: string[]
  outputNames: string[]
  modelBytes: number
  ort: OrtModule
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
  const module = await moduleFor(device)
  configureOrt(module)
  if (device === 'webgpu') await ensureWebGpuAdapter(module)

  const { model, bytes } = await loadModel(onnxPath)
  const options: ortWebgpu.InferenceSession.SessionOptions = {
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

  const session = await module.InferenceSession.create(model, options)
  return {
    session,
    device,
    inputNames: [...session.inputNames],
    outputNames: [...session.outputNames],
    modelBytes: bytes,
    ort: module,
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
