import { useEffect, useMemo, useRef, useState } from 'react'
import { encodeWav } from '../engine/audio'
import type { Precision } from '../engine/manifest'
import { sanitizeForTts } from '../engine/sanitize'
import { SUPPORTED_SCRIPTS, transliterate } from '../engine/transliterate'
import { loadVoices, type PrecomputedVoice } from '../engine/voices'
import { useSynthesizer } from './useSynthesizer'
import type { Device } from '../engine/ort'

const TARGET_RATE = 24000
const MAX_REFERENCE_SECONDS = 30

const EXAMPLES = [
  'Hola, how are you doing today? 你今天怎么样？',
  'Hello, kaise ho tum? Let us test the system.',
  'Bonjour, je vais bien — thanks for asking.',
]

async function decodeToMono(file: File): Promise<{ samples: Float32Array; seconds: number }> {
  const data = await file.arrayBuffer()
  const context = new AudioContext()
  try {
    const decoded = await context.decodeAudioData(data)
    const seconds = Math.min(decoded.duration, MAX_REFERENCE_SECONDS)
    const length = Math.max(1, Math.ceil(seconds * TARGET_RATE))
    const offline = new OfflineAudioContext(1, length, TARGET_RATE)
    const source = offline.createBufferSource()
    source.buffer = decoded
    source.connect(offline.destination)
    source.start()
    const rendered = await offline.startRendering()
    return { samples: rendered.getChannelData(0), seconds }
  } finally {
    await context.close()
  }
}

export function Synthesizer({ device }: { device: Device }) {
  const [text, setText] = useState(EXAMPLES[1])
  const [precision, setPrecision] = useState<Precision>('int4')
  const [safeDecoder, setSafeDecoder] = useState(false)
  const { view, generate, encodeReference, setReference, reference, clearReference, selfTest } =
    useSynthesizer(device, precision, safeDecoder)

  const playTone = (): void => {
    const rate = 24000
    const samples = new Float32Array(rate)
    for (let i = 0; i < samples.length; i++) samples[i] = 0.3 * Math.sin((2 * Math.PI * 440 * i) / rate)
    const url = URL.createObjectURL(new Blob([encodeWav(samples, rate)], { type: 'audio/wav' }))
    void new Audio(url).play()
  }
  const [refText, setRefText] = useState('')
  const [refStatus, setRefStatus] = useState<string | null>(null)
  const [consented, setConsented] = useState(false)
  const [translit, setTranslit] = useState(true)
  const [cleanup, setCleanup] = useState(true)
  const [script, setScript] = useState('devanagari')
  const [voices, setVoices] = useState<PrecomputedVoice[]>([])
  const [elapsed, setElapsed] = useState(0)
  const prepared = useMemo(() => {
    const sanitized = cleanup ? sanitizeForTts(text) : text
    return translit ? transliterate(sanitized, { keepEnglish: true, script }) : sanitized
  }, [text, cleanup, translit, script])
  const busy = view.phase === 'loading' || view.phase === 'generating'
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!busy) return
    const start = performance.now()
    const id = setInterval(() => setElapsed((performance.now() - start) / 1000), 200)
    return () => clearInterval(id)
  }, [busy])

  useEffect(() => {
    void loadVoices(`${import.meta.env.BASE_URL}voices.json`).then(setVoices)
  }, [])

  const onPickFile = async (file: File): Promise<void> => {
    setRefStatus(`decoding ${file.name}…`)
    try {
      const { samples, seconds } = await decodeToMono(file)
      setRefStatus(`encoding ${seconds.toFixed(1)}s reference…`)
      await encodeReference(samples, file.name, refText)
      setRefStatus(null)
    } catch (error) {
      setRefStatus(
        `Could not decode "${file.name}" (${file.type || 'unknown type'}). Try WAV, MP3, OGG/Opus, M4A or FLAC.`,
      )
      void error
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

      <div className="controls-row">
        <label className="consent">
          <input type="checkbox" checked={translit} onChange={(e) => setTranslit(e.target.checked)} />
          <span>Romanized → native script</span>
        </label>
        <select
          className="select"
          value={script}
          disabled={!translit}
          onChange={(e) => setScript(e.target.value)}
          aria-label="Target script"
        >
          {SUPPORTED_SCRIPTS.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
        <label className="consent">
          <input type="checkbox" checked={cleanup} onChange={(e) => setCleanup(e.target.checked)} />
          <span>Sanitize input</span>
        </label>
        <select
          className="select"
          value={precision}
          disabled={busy}
          onChange={(e) => setPrecision(e.target.value as Precision)}
          aria-label="Backbone precision"
        >
          <option value="int4">Backbone: int4 (fast)</option>
          <option value="fp16">Backbone: fp16</option>
          <option value="fp32">Backbone: fp32 (accurate)</option>
        </select>
        <label className="consent">
          <input
            type="checkbox"
            checked={safeDecoder}
            onChange={(e) => setSafeDecoder(e.target.checked)}
          />
          <span>Audio decoder on CPU (fixes blank audio)</span>
        </label>
      </div>
      {view.metrics && view.metrics.rms < 0.005 && (
        <p className="notice warn">
          Output looks like silence. If you're on WebGPU, this GPU may miscompute the int4 weights —
          set <strong>Backbone</strong> to <strong>fp16</strong> or <strong>fp32</strong> and
          regenerate.
        </p>
      )}
      {prepared !== text && (
        <div className="preview">
          <span className="preview-label">Model input preview — this is what OmniVoice speaks</span>
          <p>{prepared}</p>
        </div>
      )}

      {device === 'wasm' && (
        <p className="notice caution">
          WebGPU isn't available, so generation runs on the CPU (WASM) — slow and lower quality. For
          good audio, get WebGPU working: on Android Chrome only Qualcomm/ARM GPUs are enabled by
          default, so AMD/Exynos devices must enable{' '}
          <span className="mono">chrome://flags/#enable-unsafe-webgpu</span> and{' '}
          <span className="mono">#ignore-gpu-blocklist</span>, then relaunch Chrome.
        </p>
      )}

      <div className="progress">
        <span style={{ width: `${Math.round(view.ratio * 100)}%` }} />
      </div>

      <div className="actions">
        <button
          className="btn primary"
          disabled={busy || !text.trim()}
          onClick={() => void generate(prepared)}
        >
          {busy ? 'Working…' : 'Generate'}
        </button>
        {view.audioUrl && (
          <a className="btn ghost" href={view.audioUrl} download="omnivoice.wav">
            Download WAV
          </a>
        )}
        <button className="btn ghost" type="button" onClick={playTone}>
          Test sound
        </button>
        <button className="btn ghost" type="button" disabled={busy} onClick={() => void selfTest()}>
          Test decoder
        </button>
        {busy && (
          <span className="muted small">
            {view.stage}
            {view.detail ? ` · ${view.detail}` : ''} · {elapsed.toFixed(1)}s
          </span>
        )}
      </div>

      {view.audioUrl && <audio className="player" controls src={view.audioUrl} />}

      <details className="cloner">
        <summary>Clone a voice (optional)</summary>
        <p className="muted small">
          Pick a precomputed voice (instant) or upload a 3–10s clip in any format — decoding happens
          locally in your browser, nothing is uploaded. Uploading a new clip needs the Full profile
          (Higgs encoders, ~654 MB, downloaded on first use). Only clone voices you have consent to
          use.
        </p>
        {voices.length > 0 && (
          <div className="chips">
            {voices.map((voice) => (
              <button
                key={voice.id}
                type="button"
                className="chip"
                disabled={busy}
                onClick={() =>
                  setReference({
                    data: voice.codes,
                    frames: voice.frames,
                    rms: voice.rms,
                    text: voice.text,
                    name: voice.name,
                  })
                }
              >
                Use {voice.name}
              </button>
            ))}
          </div>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,.mp3,.wav,.ogg,.opus,.m4a,.aac,.flac,.webm"
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
          Generated {view.frames} frames in {(view.milliseconds / 1000).toFixed(2)}s on {device}.
          {view.ttfa !== null && ` TTFA ${view.ttfa.toFixed(2)}s.`}
          {view.rtf !== null &&
            ` RTF ${view.rtf.toFixed(2)}× (${view.rtf < 1 ? 'faster' : 'slower'} than real-time).`}
          {view.metrics &&
            ` RMS ${view.metrics.rms.toFixed(3)} · peak ${view.metrics.peak.toFixed(3)} · speech band ${(view.metrics.speechBand * 100).toFixed(0)}%.`}{' '}
          Audio never left your device.
        </p>
      )}
      {view.notice && <p className="notice caution">{view.notice}</p>}
      {view.error && <p className="notice err">{view.error}</p>}
    </section>
  )
}
