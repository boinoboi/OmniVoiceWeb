import { Synthesizer, type SynthProgress } from '../engine/pipeline'
import type { Device } from '../engine/ort'

interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void
  onmessage: ((event: MessageEvent) => void) | null
}

const scope = self as unknown as WorkerScope
const synth = new Synthesizer()

type Incoming =
  | { type: 'load'; device: Device }
  | { type: 'generate'; text: string; numAudioTokens?: number; numSteps?: number }
  | { type: 'dispose' }

const progress = (p: SynthProgress) => scope.postMessage({ type: 'progress', progress: p })

scope.onmessage = async (event: MessageEvent<Incoming>) => {
  const message = event.data
  try {
    if (message.type === 'load') {
      try {
        await synth.load(message.device, 'lite', progress)
      } catch (error) {
        if (message.device !== 'wasm') {
          scope.postMessage({ type: 'notice', message: 'WebGPU failed, falling back to WASM' })
          await synth.dispose()
          await synth.load('wasm', 'lite', progress)
        } else {
          throw error
        }
      }
      scope.postMessage({ type: 'loaded', device: synth.device })
    } else if (message.type === 'generate') {
      const result = await synth.generate(
        message.text,
        { numAudioTokens: message.numAudioTokens, numSteps: message.numSteps },
        progress,
      )
      scope.postMessage(
        {
          type: 'result',
          samples: result.samples,
          sampleRate: result.sampleRate,
          frames: result.frames,
          milliseconds: result.milliseconds,
        },
        [result.samples.buffer],
      )
    } else if (message.type === 'dispose') {
      await synth.dispose()
    }
  } catch (error) {
    scope.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    })
  }
}
