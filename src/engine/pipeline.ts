import { iterativeUnmask } from './algorithm'
import { createBackboneStep } from './backbone'
import { ensureCached } from './cache'
import { toFloat32 } from './dtype'
import { filesForProfile, type Profile } from './manifest'
import { createCachedSession, ort, releaseSession, type Device, type SessionHandle } from './ort'
import { encodeText } from './tokenizer'

export interface SynthProgress {
  stage: 'download' | 'load' | 'tokenize' | 'generate' | 'decode' | 'done'
  ratio: number
  detail?: string
}

export interface SynthOptions {
  numSteps?: number
  numAudioTokens?: number
}


export interface SynthResult {
  samples: Float32Array
  sampleRate: number
  frames: number
  textTokens: number[]
  milliseconds: number
}

const PATHS = {
  embeddings: 'int4/audio_embeddings_encoder.onnx',
  llm: 'int4/llm_decoder.onnx',
  heads: 'int4/audio_heads_decoder.onnx',
  decoder: 'audio_tokenizer/higgs_decoder.onnx',
} as const

export function estimateFrames(text: string): number {
  return Math.max(48, Math.min(600, Math.round(text.length * 1.7) + 20))
}

function toBigInt64(input: Int32Array): BigInt64Array {
  const out = new BigInt64Array(input.length)
  for (let i = 0; i < input.length; i++) out[i] = BigInt(input[i])
  return out
}

export class Synthesizer {
  private embeddings?: SessionHandle
  private llm?: SessionHandle
  private heads?: SessionHandle
  private decoder?: SessionHandle
  device: Device = 'wasm'

  get loaded(): boolean {
    return Boolean(this.embeddings && this.llm && this.heads && this.decoder)
  }

  async load(
    device: Device,
    profile: Profile = 'lite',
    onProgress?: (progress: SynthProgress) => void,
  ): Promise<void> {
    this.device = device
    if (this.loaded) return

    const files = filesForProfile(profile)
    await ensureCached(files, {
      concurrency: 2,
      onUpdate: (update) =>
        onProgress?.({
          stage: 'download',
          ratio: update.overallTotal ? update.overallLoaded / update.overallTotal : 0,
          detail: update.file.path,
        }),
    })

    onProgress?.({ stage: 'load', ratio: 0 })
    // Session creation must be sequential: ORT mounts external data globally.
    this.embeddings = await createCachedSession(PATHS.embeddings, device)
    this.llm = await createCachedSession(PATHS.llm, device)
    this.heads = await createCachedSession(PATHS.heads, device)
    this.decoder = await createCachedSession(PATHS.decoder, device)
  }

  async generate(
    text: string,
    options: SynthOptions = {},
    onProgress?: (progress: SynthProgress) => void,
  ): Promise<SynthResult> {
    if (!this.embeddings || !this.llm || !this.heads || !this.decoder) {
      throw new Error('Synthesizer is not loaded')
    }
    const start = performance.now()

    onProgress?.({ stage: 'tokenize', ratio: 0 })
    const textTokens = await encodeText(text)
    const frames = options.numAudioTokens ?? estimateFrames(text)
    const steps = options.numSteps ?? 32

    const step = createBackboneStep(ort, this.embeddings, this.llm, this.heads)
    onProgress?.({ stage: 'generate', ratio: 0 })
    const generation = await iterativeUnmask(
      { textTokens, numAudioTokens: frames, numSteps: steps },
      step,
      (index, remaining) =>
        onProgress?.({
          stage: 'generate',
          ratio: (index + 1) / steps,
          detail: `${remaining} frames masked`,
        }),
    )

    onProgress?.({ stage: 'decode', ratio: 0 })
    const codes = new ort.Tensor('int64', toBigInt64(generation.codes), [8, 1, frames])
    const output = await this.decoder.session.run({ codes })
    const samples = toFloat32(output.waveform_24k)

    onProgress?.({ stage: 'done', ratio: 1 })
    return {
      samples,
      sampleRate: 24000,
      frames,
      textTokens,
      milliseconds: performance.now() - start,
    }
  }

  async dispose(): Promise<void> {
    await releaseSession(this.embeddings ?? null)
    await releaseSession(this.llm ?? null)
    await releaseSession(this.heads ?? null)
    await releaseSession(this.decoder ?? null)
    this.embeddings = undefined
    this.llm = undefined
    this.heads = undefined
    this.decoder = undefined
  }
}
