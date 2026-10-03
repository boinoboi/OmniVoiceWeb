import { SystemCheck } from './ui/SystemCheck'
import { ModelManager } from './ui/ModelManager'
import { Synthesizer } from './ui/Synthesizer'

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">OV</span>
          <div>
            <h1>OmniVoice Web</h1>
            <p className="tagline">Multilingual zero-shot TTS · 100% local · WebGPU</p>
          </div>
        </div>
        <a className="ghost-link" href="https://huggingface.co/onnx-community/OmniVoice-Onnx" target="_blank" rel="noreferrer">
          onnx-community/OmniVoice-Onnx
        </a>
      </header>

      <main className="grid">
        <SystemCheck />
        <ModelManager />
        <Synthesizer ready={false} />
      </main>

      <footer className="footer">
        <p className="muted small">
          Model weights are CC-BY-NC. Never use voice cloning without consent. No audio leaves your device.
        </p>
      </footer>
    </div>
  )
}
