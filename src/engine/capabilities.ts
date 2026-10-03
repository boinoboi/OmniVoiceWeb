export interface GpuInfo {
  available: boolean
  vendor?: string
  architecture?: string
  device?: string
  description?: string
  maxBufferSize?: number
  maxStorageBufferBindingSize?: number
  error?: string
}

interface AdapterInfoLike {
  vendor?: string
  architecture?: string
  device?: string
  description?: string
}

interface AdapterLike {
  info?: AdapterInfoLike
  limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number }
}

interface GpuLike {
  requestAdapter(): Promise<AdapterLike | null>
}

export async function detectGpu(): Promise<GpuInfo> {
  if (typeof navigator === 'undefined') {
    return { available: false, error: 'No browser environment' }
  }
  const gpu = (navigator as unknown as { gpu?: GpuLike }).gpu
  if (!gpu) {
    return { available: false, error: 'WebGPU is not supported in this browser' }
  }
  try {
    const adapter = await gpu.requestAdapter()
    if (!adapter) return { available: false, error: 'No WebGPU adapter available' }
    return {
      available: true,
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
      device: adapter.info?.device,
      description: adapter.info?.description,
      maxBufferSize: adapter.limits?.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize,
    }
  } catch (error) {
    return { available: false, error: error instanceof Error ? error.message : String(error) }
  }
}
