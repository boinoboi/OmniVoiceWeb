import { useEffect, useState } from 'react'
import {
  classifyTier,
  detectDevice,
  detectGpu,
  type DeviceInfo,
  type DeviceTier,
  type GpuInfo,
} from '../engine/capabilities'
import { estimateStorage, type StorageEstimate } from '../engine/cache'

export interface Capabilities {
  gpu: GpuInfo | null
  device: DeviceInfo
  tier: DeviceTier
  storage?: StorageEstimate
}

export function useCapabilities(): Capabilities {
  const [gpu, setGpu] = useState<GpuInfo | null>(null)
  const [storage, setStorage] = useState<StorageEstimate | undefined>()
  const [device] = useState<DeviceInfo>(() => detectDevice())

  useEffect(() => {
    let active = true
    void detectGpu().then((info) => {
      if (active) setGpu(info)
    })
    void estimateStorage().then((info) => {
      if (active) setStorage(info)
    })
    return () => {
      active = false
    }
  }, [])

  return { gpu, device, tier: classifyTier(gpu, device), storage }
}
