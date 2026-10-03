import type { DeviceInfo, DeviceTier, GpuInfo } from '../engine/capabilities'
import type { StorageEstimate } from '../engine/cache'
import { formatBytes } from '../engine/manifest'

interface Props {
  gpu: GpuInfo | null
  device: DeviceInfo
  tier: DeviceTier
  storage?: StorageEstimate
}

export function SystemCheck({ gpu, device, tier, storage }: Props) {
  const deviceLabel =
    device.platform === 'unknown'
      ? device.mobile
        ? 'Mobile'
        : 'Desktop'
      : device.platform

  return (
    <section className="card">
      <div className="card-head">
        <h2>This device</h2>
        <span className={`badge ${gpu?.available ? 'ok' : gpu ? 'warn' : 'idle'}`}>
          {gpu === null ? 'checking…' : gpu.available ? 'WebGPU ready' : 'CPU fallback'}
        </span>
      </div>
      <div className="stats">
        <div className="stat">
          <span className="stat-label">GPU</span>
          <span className="stat-value mono">
            {gpu?.vendor || gpu?.architecture || (gpu?.available ? 'detected' : '—')}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Max buffer</span>
          <span className="stat-value mono">
            {gpu?.maxBufferSize ? formatBytes(gpu.maxBufferSize) : '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Device</span>
          <span className="stat-value">
            {deviceLabel}
            {device.cores ? ` · ${device.cores} cores` : ''}
            {device.memoryGB ? ` · ~${device.memoryGB} GB` : ''}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Tier</span>
          <span className="stat-value" style={{ textTransform: 'capitalize' }}>
            {tier} · int4{gpu?.shaderF16 ? ' / f16' : ''}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Storage used</span>
          <span className="stat-value mono">
            {storage ? `${formatBytes(storage.usage)} / ${formatBytes(storage.quota)}` : '—'}
          </span>
        </div>
      </div>
    </section>
  )
}
