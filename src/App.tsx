import { SystemCheck } from './ui/SystemCheck'
import { ModelManager } from './ui/ModelManager'
import { EngineLab } from './ui/EngineLab'
import { Synthesizer } from './ui/Synthesizer'
import { Dashboard } from './ui/Dashboard'
import { useCapabilities } from './ui/useCapabilities'

export default function App() {
  const { gpu, device, storage } = useCapabilities()

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">OV</span>
          <div>
            <h1>OmniVoice Web</h1>
            <p className="tagline">Mix languages mid-sentence · 600+ languages · local WebGPU</p>
          </div>
        </div>
        <a
          className="ghost-link"
          href="https://github.com/boinoboi/OmniVoiceWeb"
          target="_blank"
          rel="noreferrer"
        >
          Source
        </a>
      </header>

      <div className="hero-strip">
        <span className="pill">Code-switching ready</span>
        <span className="muted small">
          Speak English and Spanish (or any of 600+ languages) in the same sentence — OmniVoice
          follows without switching voices.
        </span>
      </div>

      <main className="grid">
        <SystemCheck gpu={gpu} device={device} storage={storage} />
        <ModelManager gpu={gpu} device={device} />
        <EngineLab gpu={gpu} />
        <Dashboard />
        <Synthesizer device={gpu?.available ? 'webgpu' : 'wasm'} />
      </main>

      <footer className="footer">
        <p className="muted small">
          Model weights are CC-BY-NC. Never use voice cloning without consent. All audio is generated
          on your device and never uploaded.
        </p>
      </footer>
    </div>
  )
}
