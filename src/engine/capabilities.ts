export interface GpuInfo {
  available: boolean
  navigatorGpu: boolean
  vendor?: string
  architecture?: string
  device?: string
  description?: string
  maxBufferSize?: number
  maxStorageBufferBindingSize?: number
  shaderF16?: boolean
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
  features?: { has(feature: string): boolean }
}

interface AdapterRequestOptions {
  powerPreference?: string
  forceFallbackAdapter?: boolean
  compatibilityMode?: boolean
  featureLevel?: string
}

interface GpuLike {
  requestAdapter(options?: AdapterRequestOptions): Promise<AdapterLike | null>
}

export async function detectGpu(): Promise<GpuInfo> {
  if (typeof navigator === 'undefined') {
    return { available: false, navigatorGpu: false, error: 'No browser environment' }
  }
  const gpu = (navigator as unknown as { gpu?: GpuLike }).gpu
  if (!gpu) {
    return {
      available: false,
      navigatorGpu: false,
      error: 'navigator.gpu is undefined — this browser does not expose WebGPU',
    }
  }

  const request = async (options?: AdapterRequestOptions): Promise<AdapterLike | null | undefined> => {
    try {
      return await gpu.requestAdapter(options)
    } catch {
      return undefined
    }
  }

  for (let attempt = 0; attempt < 5; attempt++) {
    const adapter =
      (await request()) ??
      (await request({ powerPreference: 'high-performance' })) ??
      (await request({ featureLevel: 'compatibility' })) ??
      (await request({ forceFallbackAdapter: true }))
    if (adapter) {
      return {
        available: true,
        navigatorGpu: true,
        vendor: adapter.info?.vendor,
        architecture: adapter.info?.architecture,
        device: adapter.info?.device,
        description: adapter.info?.description,
        maxBufferSize: adapter.limits?.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits?.maxStorageBufferBindingSize,
        shaderF16: adapter.features?.has('shader-f16') ?? false,
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 400))
  }
  return {
    available: false,
    navigatorGpu: true,
    error: 'requestAdapter() returned no adapter (WebGPU present but no compatible GPU/driver)',
  }
}

export interface DeviceInfo {
  mobile: boolean
  ios: boolean
  android: boolean
  touch: boolean
  cores?: number
  memoryGB?: number
  platform: string
}

export function detectDevice(): DeviceInfo {
  if (typeof navigator === 'undefined') {
    return { mobile: false, ios: false, android: false, touch: false, platform: 'unknown' }
  }
  const ua = navigator.userAgent
  const maxTouch = navigator.maxTouchPoints ?? 0
  const ios =
    /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === 'MacIntel' && maxTouch > 1)
  const android = /Android/i.test(ua)
  const mobile = ios || android || /Mobile|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua)
  const nav = navigator as Navigator & { deviceMemory?: number }
  return {
    mobile,
    ios,
    android,
    touch: maxTouch > 0 || 'ontouchstart' in window,
    cores: navigator.hardwareConcurrency,
    memoryGB: nav.deviceMemory,
    platform: navigator.platform || 'unknown',
  }
}

export type DeviceTier = 'phone' | 'laptop' | 'desktop'

export function classifyTier(gpu: GpuInfo | null, device: DeviceInfo): DeviceTier {
  if (device.mobile) return 'phone'
  const descriptor = `${gpu?.vendor ?? ''} ${gpu?.architecture ?? ''} ${gpu?.description ?? ''}`
  const discrete = /nvidia|geforce|rtx|radeon|amd|arc|apple m\d/i.test(descriptor)
  const bigBuffer = (gpu?.maxBufferSize ?? 0) >= 1_500_000_000
  const manyCores = (device.cores ?? 0) >= 8
  if (discrete && (bigBuffer || manyCores)) return 'desktop'
  return 'laptop'
}

export type FeasibilityLevel = 'good' | 'caution' | 'warn'

export interface Feasibility {
  level: FeasibilityLevel
  messages: string[]
}

export function assessFeasibility(
  gpu: GpuInfo | null,
  device: DeviceInfo,
  profileBytes: number,
): Feasibility {
  const messages: string[] = []
  let level: FeasibilityLevel = 'good'

  if (!gpu) {
    return { level: 'caution', messages: ['Checking device capabilities…'] }
  }

  if (!gpu.available) {
    level = 'caution'
    messages.push(
      'WebGPU is unavailable. Generation will fall back to CPU (WASM) and be far slower.',
    )
    if (device.ios) {
      messages.push('On iPhone/iPad, WebGPU needs iOS 18 or newer in Safari.')
    }
  }

  const profileMB = profileBytes / 1_000_000
  const maxBuffer = gpu.maxBufferSize ?? 0
  if (maxBuffer && profileMB > 0 && maxBuffer < 268_435_456) {
    level = 'warn'
    messages.push(
      'This GPU reports a small max buffer size. Large int4 weights may fail to load on this device.',
    )
  }

  if (device.memoryGB !== undefined && device.memoryGB <= 4) {
    level = 'warn'
    messages.push(
      `This device reports ~${device.memoryGB} GB RAM. Downloading hundreds of MB of weights may exhaust memory.`,
    )
  }

  if (device.mobile && profileBytes > 500_000_000) {
    level = level === 'warn' ? 'warn' : 'caution'
    messages.push(
      'The Full (voice cloning) profile is large and may be unstable on phones. Lite is recommended on mobile.',
    )
  }

  if (messages.length === 0) {
    messages.push('This device looks capable of running the engine locally.')
  }

  return { level, messages }
}
