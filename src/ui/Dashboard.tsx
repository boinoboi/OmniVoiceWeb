import { formatBytes } from '../engine/manifest'

interface Variant {
  name: string
  size: number
  device: string
  wer: number
  speechBand: number
  duration: number
}

const FOX = 'The quick brown fox jumps over the lazy dog.'

const VARIANTS: Variant[] = [
  { name: 'fp32 bidirectional', size: 1_766_000_000, device: 'WebGPU / CUDA', wer: 0, speechBand: 0.87, duration: 2.56 },
  { name: 'fp16 bidirectional', size: 885_000_000, device: 'f16-capable GPUs', wer: 0, speechBand: 0.687, duration: 2.64 },
  { name: 'int4 bidirectional', size: 280_000_000, device: 'WebGPU / CUDA', wer: 0, speechBand: 0.859, duration: 2.56 },
]

export function Dashboard() {
  return (
    <section className="card">
      <div className="card-head">
        <h2>Results</h2>
        <span className="badge ok">WER 0.00 on "{FOX.slice(0, 18)}…"</span>
      </div>
      <p className="muted small">
        Precision ladder measured with round-trip Whisper WER (lower is better) and speech-band
        energy. The browser default is int4 — the smallest variant that still scores WER 0.
      </p>
      <table className="results">
        <thead>
          <tr>
            <th>variant</th>
            <th>size</th>
            <th>device</th>
            <th>WER</th>
            <th>speech band</th>
          </tr>
        </thead>
        <tbody>
          {VARIANTS.map((v) => (
            <tr key={v.name} className={v.name.startsWith('int4') ? 'highlight' : ''}>
              <td>{v.name}</td>
              <td className="mono">{formatBytes(v.size)}</td>
              <td className="muted">{v.device}</td>
              <td className="mono">{v.wer.toFixed(2)}</td>
              <td className="mono">{(v.speechBand * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Evaluation: {FOX} (2.56–2.64 s). int4 is 6.3× smaller than fp32 at equal task quality.
      </p>
    </section>
  )
}
