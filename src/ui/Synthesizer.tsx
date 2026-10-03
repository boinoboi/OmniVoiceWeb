import { useState } from 'react'

export function Synthesizer({ ready }: { ready: boolean }) {
  const [text, setText] = useState('Hello, this is OmniVoice running entirely in your browser on WebGPU.')

  return (
    <section className="card">
      <div className="card-head">
        <h2>Synthesize</h2>
        <span className={`badge ${ready ? 'ok' : 'idle'}`}>{ready ? 'engine ready' : 'models required'}</span>
      </div>
      <textarea
        className="textarea"
        rows={5}
        value={text}
        disabled={!ready}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type something to speak…"
      />
      <div className="actions">
        <button className="btn primary" disabled>
          Generate
        </button>
        <span className="muted small">Inference pipeline lands in the next milestone.</span>
      </div>
    </section>
  )
}
