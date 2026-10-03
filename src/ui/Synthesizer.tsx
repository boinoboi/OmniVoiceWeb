import { useRef, useState } from 'react'
import { resample } from '../engine/audio'
import { useSynthesizer } from './useSynthesizer'
import type { Device } from '../engine/ort'

const EXAMPLES = [
  'Hola, how are you doing today? 你今天怎么样？',
  'Hello, kaise ho tum? Let us test the system.',
  'Bonjour, je vais bien — thanks for asking.',
]

async function decodeToMono(file: File): Promise<Float32Array> {
  const data = await file.arrayBuffer()
  const context = new AudioContext()
  try {
    const decoded = await context.decodeAudioData(data)
    const mono = new Float32Array(decoded.length)
    const channels = decoded.numberOfChannels
    for (let ch = 0; ch < channels; ch++) {
      const channel = decoded.getChannelData(ch)
      for (let i = 0; i < channel.length; i++) mono[i] += channel[i] / channels
    }
    return resample(mono, decoded.sampleRate, 24000)
  } finally {
    await context.close()
  }
}

export function Synthesizer({ device }: { device: Device }) {
  const [text, setText] = useState(EXAMPLES[1])
  const { view, generate, encodeReference, reference, clearReference } = useSynthesizer(device)
  const [refText, setRefText] = useState('')
  const [refStatus, setRefStatus] = useState<string | null>(null)
  const [consented, setConsented] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const busy = view.phase === 'loading' || view.phase === 'generating'

  const onPickFile = async (file: File): Promise<void> => {
    setRefStatus(`decoding ${file.name}…`)
    try {
      const samples = await decodeToMono(file)
      setRefStatus(`encoding ${(samples.length / 24000).toFixed(1)}s reference…`)
      await encodeReference(samples, file.name, refText)
      setRefStatus(null)
    } catch (error) {
      setRefStatus(error instanceof Error ? error.message : String(error))
    }
  }

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

      <details className="cloner">
        <summary>Clone a voice (optional)</summary>
        <p className="muted small">
          Upload a 3–10s clip. Needs the Full profile (Higgs encoders, ~654 MB, downloaded on first
          use). Only clone voices you have consent to use.
        </p>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void onPickFile(file)
            e.target.value = ''
          }}
        />
        <input
          className="textarea"
          placeholder="Transcript of the clip (optional, improves fidelity)"
          value={refText}
          disabled={busy}
          onChange={(e) => setRefText(e.target.value)}
        />
        <label className="consent">
          <input
            type="checkbox"
            checked={consented}
            onChange={(e) => setConsented(e.target.checked)}
          />
          <span>I have the speaker's consent to clone this voice.</span>
        </label>
        <div className="actions">
          <button
            className="btn ghost"
            type="button"
            disabled={busy || !consented}
            onClick={() => fileRef.current?.click()}
          >
            Choose reference clip
          </button>
          {reference && (
            <button className="btn ghost" type="button" onClick={clearReference}>
              Clear voice
            </button>
          )}
        </div>
        {reference && (
          <p className="muted small">
            Active voice: {reference.name ?? 'reference'} · {reference.frames} frames
          </p>
        )}
        {refStatus && <p className="muted small">{refStatus}</p>}
      </details>

      {view.phase === 'ready' && (
        <p className="muted small">
          Generated {view.frames} frames in {(view.milliseconds / 1000).toFixed(1)}s on {device}.
          {view.metrics &&
            ` Speech band ${(view.metrics.speechBand * 100).toFixed(0)}% · centroid ${view.metrics.centroidHz.toFixed(0)} Hz · rms ${view.metrics.rms.toFixed(2)}.`}{' '}
          Audio never left your device.
        </p>
      )}
      {view.notice && <p className="notice caution">{view.notice}</p>}
      {view.error && <p className="notice err">{view.error}</p>}
    </section>
  )
}
