import { useMemo } from 'react'
import { useModelManager } from './useModelManager'
import { assessFeasibility, type DeviceInfo, type GpuInfo } from '../engine/capabilities'
import {
  KIND_LABELS,
  bytesForProfile,
  formatBytes,
  type AssetKind,
  type Profile,
} from '../engine/manifest'

interface Props {
  gpu: GpuInfo | null
  device: DeviceInfo
}

function groupStatuses(
  statuses: { file: { kind: AssetKind; bytes: number }; cached: boolean }[],
) {
  const order: AssetKind[] = ['tokenizer', 'backbone', 'decoder', 'encoder']
  const seen = new Map<AssetKind, { bytes: number; cachedBytes: number; count: number }>()
  for (const s of statuses) {
    const entry = seen.get(s.file.kind) ?? { bytes: 0, cachedBytes: 0, count: 0 }
    entry.bytes += s.file.bytes
    entry.count += 1
    if (s.cached) entry.cachedBytes += s.file.bytes
    seen.set(s.file.kind, entry)
  }
  return order.filter((k) => seen.has(k)).map((kind) => ({ kind, ...seen.get(kind)! }))
}

export function ModelManager({ gpu, device }: Props) {
  const manager = useModelManager()
  const { profile, setProfile, progress, downloading, error, allReady, cacheSupported } = manager

  const groups = useMemo(() => groupStatuses(manager.statuses), [manager.statuses])
  const pct = progress.total > 0 ? Math.min(100, (progress.loaded / progress.total) * 100) : 0
  const feasibility = useMemo(
    () => assessFeasibility(gpu, device, progress.total),
    [gpu, device, progress.total],
  )

  return (
    <section className="card">
      <div className="card-head">
        <h2>Models</h2>
        <span className={`badge ${allReady ? 'ok' : 'idle'}`}>
          {allReady
            ? 'cached & ready'
            : `${formatBytes(progress.loaded)} / ${formatBytes(progress.total)}`}
        </span>
      </div>

      <div className="segmented" role="tablist" aria-label="Model profile">
        {(['lite', 'full'] as Profile[]).map((p) => (
          <button
            key={p}
            role="tab"
            aria-selected={profile === p}
            className={profile === p ? 'active' : ''}
            disabled={downloading}
            onClick={() => setProfile(p)}
          >
            <strong>{p === 'lite' ? 'Lite' : 'Full (cloning)'}</strong>
            <span className="muted">{formatBytes(bytesForProfile(p))}</span>
          </button>
        ))}
      </div>

      <p className="muted small">
        {profile === 'lite'
          ? 'Auto-voice only. Downloads the backbone, tokenizer and audio decoder.'
          : 'Adds the Higgs encoders so you can clone a voice from a short reference clip.'}
      </p>

      <div className="progress">
        <span style={{ width: `${pct}%` }} />
      </div>

      <ul className="filelist">
        {groups.map((g) => {
          const done = g.cachedBytes >= g.bytes
          return (
            <li key={g.kind} className="file">
              <span className={`dot ${done ? 'ok' : 'idle'}`} />
              <span className="file-name">{KIND_LABELS[g.kind]}</span>
              <span className="muted mono">
                {done ? 'ready' : `${formatBytes(g.cachedBytes)} / ${formatBytes(g.bytes)}`}
              </span>
            </li>
          )
        })}
      </ul>

      <p className={`notice ${feasibility.level}`}>
        {feasibility.messages.map((m, i) => (
          <span key={i} className="notice-line">
            {m}
          </span>
        ))}
      </p>

      {error && <p className="notice err">{error}</p>}
      {!cacheSupported && (
        <p className="notice err">
          Cache Storage is unavailable. Models will not persist between reloads. Serve over HTTPS or
          localhost.
        </p>
      )}

      <div className="actions">
        {!allReady && !downloading && (
          <button className="btn primary" onClick={() => void manager.start()}>
            Download {formatBytes(progress.total)}
          </button>
        )}
        {downloading && (
          <button className="btn danger" onClick={manager.cancel}>
            Cancel
          </button>
        )}
        <button className="btn ghost" onClick={() => void manager.clear()} disabled={downloading}>
          Clear cache
        </button>
      </div>
    </section>
  )
}
