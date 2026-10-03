import { useState } from 'react'

const EXAMPLES = [
  'Hola, how are you doing today? 你今天怎么样？',
  'Bonjour, je vais bien — thanks for asking.',
  'Dime, what is the plan for mañana?',
]

export function Synthesizer({ ready }: { ready: boolean }) {
  const [text, setText] = useState(EXAMPLES[0])

  return (
    <section className="card">
      <div className="card-head">
        <h2>Synthesize</h2>
        <span className={`badge ${ready ? 'ok' : 'idle'}`}>
          {ready ? 'engine ready' : 'models required'}
        </span>
      </div>

      <textarea
        className="textarea"
        rows={4}
        value={text}
        disabled={!ready}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type something to speak — mix languages freely…"
      />

      <div className="chips">
        {EXAMPLES.map((example, i) => (
          <button
            key={i}
            type="button"
            className="chip"
            disabled={!ready}
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

      <div className="actions">
        <button className="btn primary" disabled>
          Generate
        </button>
        <span className="muted small">Inference pipeline lands in the next milestone.</span>
      </div>
    </section>
  )
}
