import { useEffect, useState } from 'react'
import { detectGpu, type GpuInfo } from '../engine/capabilities'
import { estimateStorage, type StorageEstimate } from '../engine/cache'
import { formatBytes } from '../engine/manifest'

export function SystemCheck() {
  const [gpu, setGpu] = useState<GpuInfo | null>(null)
  const [storage, setStorage] = useState<StorageEstimate | undefined>()

  useEffect(() => {
    void detectGpu().then(setGpu)
    void estimateStorage().then(setStorage)
  }, [])

  return (
    <section className="card">
      <div className="card-head">
        <h2>System check</h2>
        <span className={`badge ${gpu?.available ? 'ok' : 'warn'}`}>
          {gpu === null ? 'checking…' : gpu.available ? 'WebGPU ready' : 'WASM fallback'}
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
          <span className="stat-label">Storage used</span>
          <span className="stat-value mono">
            {storage ? `${formatBytes(storage.usage)} / ${formatBytes(storage.quota)}` : '—'}
          </span>
        </div>
      </div>
      {gpu && !gpu.available && (
        <p className="muted small">{gpu.error} — inference will run on CPU via WASM and be much slower.</p>
      )}
    </section>
  )
}
