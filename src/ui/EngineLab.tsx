import { useEffect, useState } from 'react'
import { runHeadsProbe, type ProbeResult } from '../engine/probe'
import type { Device } from '../engine/ort'
import type { GpuInfo } from '../engine/capabilities'

export function EngineLab({ gpu }: { gpu: GpuInfo | null }) {
  const [device, setDevice] = useState<Device>('wasm')
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('')
  const [result, setResult] = useState<ProbeResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (gpu?.available) setDevice('webgpu')
  }, [gpu])

  const run = async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      setResult(await runHeadsProbe(device, setStatus))
      setStatus('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('')
    } finally {
      setRunning(false)
    }
  }

  return (
    <section className="card">
      <div className="card-head">
        <h2>Engine lab</h2>
        <span className="badge idle">audio_heads_decoder · 4.5 MB</span>
      </div>
      <p className="muted small">
        Creates a real ONNX Runtime session from the browser cache and runs one forward pass. This
        validates the runtime path before the full pipeline is wired up.
      </p>

      <div className="segmented" role="tablist" aria-label="Execution provider">
        {(['webgpu', 'wasm'] as Device[]).map((d) => (
          <button
            key={d}
            role="tab"
            aria-selected={device === d}
            className={device === d ? 'active' : ''}
            disabled={running}
            onClick={() => setDevice(d)}
          >
            <strong>{d === 'webgpu' ? 'WebGPU' : 'WASM (CPU)'}</strong>
            <span className="muted">
              {d === 'webgpu'
                ? gpu?.available
                  ? 'hardware accelerated'
                  : 'unavailable on this device'
                : 'fallback backend'}
            </span>
          </button>
        ))}
      </div>

      <div className="actions">
        <button className="btn primary" onClick={() => void run()} disabled={running}>
          {running ? 'Running…' : 'Load & run probe'}
        </button>
        {status && <span className="muted small">{status}</span>}
      </div>

      {error && <p className="notice err">{error}</p>}

      {result && (
        <div className="stats" style={{ marginTop: 14 }}>
          <div className="stat">
            <span className="stat-label">Provider</span>
            <span className="stat-value mono">{result.device}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Forward latency</span>
            <span className="stat-value mono">{result.latencyMs.toFixed(1)} ms</span>
          </div>
          <div className="stat">
            <span className="stat-label">Output shape</span>
            <span className="stat-value mono">[{result.outputDims.join(' × ')}]</span>
          </div>
          <div className="stat">
            <span className="stat-label">Inputs → Outputs</span>
            <span className="stat-value mono">
              {result.inputNames.join(', ')} → {result.outputNames.join(', ')}
            </span>
          </div>
        </div>
      )}
    </section>
  )
}
