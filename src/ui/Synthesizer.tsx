import { useState } from 'react'
import { useSynthesizer } from './useSynthesizer'
import type { Device } from '../engine/ort'

const EXAMPLES = [
  'Hola, how are you doing today? 你今天怎么样？',
  'Hello, kaise ho tum? Let us test the system.',
  'Bonjour, je vais bien — thanks for asking.',
]

export function Synthesizer({ device }: { device: Device }) {
  const [text, setText] = useState(EXAMPLES[1])
  const { view, generate } = useSynthesizer(device)
  const busy = view.phase === 'loading' || view.phase === 'generating'

  return (
    <section className="card">
      <div className="card-head">
        <h2>Synthesize</h2>
        <span className={`badge ${view.phase === 'ready' ? 'ok' : 'idle'}`}>
          {view.phase === 'ready' ? `${view.frames} frames` : device === 'webgpu' ? 'WebGPU' : 'WASM'}
        </span>
      </div>

      <textarea
        className="textarea"
        rows={4}
        value={text}
        disabled={busy}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type something to speak — mix languages freely…"
      />

      <div className="chips">
        {EXAMPLES.map((example, i) => (
          <button
            key={i}
            type="button"
            className="chip"
            disabled={busy}
            onClick={() => setText(example)}
          >
            {example.length > 34 ? `${example.slice(0, 34)}…` : example}
          </button>
        ))}
      </div>

      <p className="muted small">
        Code-switching: write two languages in one sentence and the model keeps a single,
        consistent voice across the language boundary.
      </p>

      <div className="progress">
        <span style={{ width: `${Math.round(view.ratio * 100)}%` }} />
      </div>

      <div className="actions">
        <button className="btn primary" disabled={busy || !text.trim()} onClick={() => void generate(text)}>
          {busy ? 'Working…' : 'Generate'}
        </button>
        {view.audioUrl && (
          <a className="btn ghost" href={view.audioUrl} download="omnivoice.wav">
            Download WAV
          </a>
        )}
        {busy && <span className="muted small">{view.stage}{view.detail ? ` · ${view.detail}` : ''}</span>}
      </div>

      {view.audioUrl && <audio className="player" controls src={view.audioUrl} />}

      {view.phase === 'ready' && (
        <p className="muted small">
          Generated {view.frames} frames in {(view.milliseconds / 1000).toFixed(1)}s on {device}.
          Audio never left your device.
        </p>
      )}
      {view.notice && <p className="notice caution">{view.notice}</p>}
      {view.error && <p className="notice err">{view.error}</p>}
    </section>
  )
}
