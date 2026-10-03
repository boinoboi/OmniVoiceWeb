import { IdleReleaser } from '../engine/memory'
import type { Precision } from '../engine/manifest'
import { Synthesizer, type SynthProgress } from '../engine/pipeline'
import { getAdapterStrategy, type Device } from '../engine/ort'

interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void
  onmessage: ((event: MessageEvent) => void) | null
}

const scope = self as unknown as WorkerScope
const synth = new Synthesizer()

const idle = new IdleReleaser({
  timeoutMs: 5 * 60_000,
  onRelease: async () => {
    if (!synth.loaded) return
    await synth.dispose()
    scope.postMessage({ type: 'notice', message: 'Released models — GPU memory freed after idling' })
  },
})

type Incoming =
  | { type: 'load'; device: Device; precision?: Precision; safeDecoder?: boolean }
  | {
      type: 'generate'
      text: string
      numAudioTokens?: number
      numSteps?: number
      maxChars?: number
      refCodes?: Int32Array
      refFrames?: number
      refRms?: number
      refText?: string
    }
  | { type: 'encode'; samples: Float32Array; device: Device }
  | { type: 'selftest' }
  | { type: 'cancel' }
  | { type: 'dispose' }

const progress = (p: SynthProgress) => scope.postMessage({ type: 'progress', progress: p })
let cancelRequested = false

scope.onmessage = async (event: MessageEvent<Incoming>) => {
  const message = event.data
  if (message.type === 'cancel') {
    cancelRequested = true
    return
  }
    const isWork =
      message.type === 'load' ||
      message.type === 'generate' ||
      message.type === 'encode' ||
      message.type === 'selftest'
  if (isWork) idle.cancel()
  try {
    if (message.type === 'load') {
      const precision: Precision = message.precision ?? 'int4'
      const safeDecoder = message.safeDecoder ?? false
      if (
        synth.loaded &&
        (synth.device !== message.device ||
          synth.precision !== precision ||
          synth.safeDecoder !== safeDecoder)
      ) {
        await synth.dispose()
      }
      try {
        await synth.load(message.device, 'lite', progress, precision, safeDecoder)
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error)
        if (message.device !== 'wasm') {
          scope.postMessage({
            type: 'notice',
            message: `WebGPU failed (${detail.slice(0, 160)}) — falling back to WASM`,
          })
          await synth.dispose()
          await synth.load('wasm', 'lite', progress, 'int4')
        } else {
          throw error
        }
      }
      if (synth.device === 'webgpu') {
        scope.postMessage({ type: 'notice', message: `WebGPU adapter: ${getAdapterStrategy() ?? 'default'}` })
      }
      scope.postMessage({ type: 'loaded', device: synth.device })
    } else if (message.type === 'generate') {
      cancelRequested = false
      const result = await synth.generateStream(
        message.text,
        {
          numAudioTokens: message.numAudioTokens,
          numSteps: message.numSteps,
          maxChars: message.maxChars,
          refText: message.refText,
          refCodes:
            message.refCodes && message.refFrames
              ? { data: message.refCodes, frames: message.refFrames, rms: message.refRms }
              : undefined,
        },
        progress,
        (chunk) =>
          scope.postMessage({
            type: 'chunk',
            samples: chunk.samples,
            sampleRate: chunk.sampleRate,
            frames: chunk.frames,
            index: chunk.index,
            total: chunk.total,
          }),
        () => cancelRequested,
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
    } else if (message.type === 'encode') {
      await synth.loadEncoders(progress, message.device)
      progress({ stage: 'clone', ratio: 0, detail: 'encoding reference audio' })
      const reference = await synth.encodeReference(message.samples)
      scope.postMessage(
        { type: 'encoded', codes: reference.data, frames: reference.frames, rms: reference.rms },
        [reference.data.buffer],
      )
    } else if (message.type === 'selftest') {
      const result = await synth.decoderSelfTest()
      scope.postMessage(
        {
          type: 'result',
          samples: result.samples,
          sampleRate: result.sampleRate,
          frames: result.frames,
          milliseconds: 0,
        },
        [result.samples.buffer],
      )
    } else if (message.type === 'dispose') {
      await synth.dispose()
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      scope.postMessage({ type: 'cancelled' })
    } else {
      scope.postMessage({
        type: 'error',
        message: error instanceof Error ? error.message : String(error),
      })
    }
  } finally {
    if (isWork) idle.touch()
  }
}
