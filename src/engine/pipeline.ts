import { iterativeUnmaskCfg } from './algorithm'
import { concatFloat32 } from './audio'
import { createBackboneStep } from './backbone'
import { ensureCached } from './cache'
import { encodeReference, type EncoderSet, type ReferenceCodes } from './cloning'
import { estimateTargetTokens } from './duration'
import { toFloat32 } from './dtype'
import { filesForDevice, filesForProfile, type Profile } from './manifest'
import { createCachedSession, releaseSession, type Device, type SessionHandle } from './ort'
import { prepareInputs } from './prompt'
import { chunkText } from './streaming'

export interface SynthProgress {
  stage: 'download' | 'load' | 'tokenize' | 'generate' | 'decode' | 'clone' | 'done'
  ratio: number
  detail?: string
}

export interface SynthOptions {
  numSteps?: number
  numAudioTokens?: number
  guidanceScale?: number
  lang?: string
  instruct?: string
  refText?: string
  refCodes?: { data: Int32Array; frames: number; rms?: number }
  maxChars?: number
}


export interface SynthResult {
  samples: Float32Array
  sampleRate: number
  frames: number
  textTokens: number[]
  milliseconds: number
}

export interface SynthChunk {
  samples: Float32Array
  sampleRate: number
  frames: number
  index: number
  total: number
  text: string
}

const PATHS = {
  embeddings: 'int4/audio_embeddings_encoder.onnx',
  embeddingsFp32: 'audio_embeddings_encoder.onnx',
  llm: 'llm_decoder_int4.onnx',
  heads: 'int4/audio_heads_decoder.onnx',
  decoder: 'audio_tokenizer/higgs_decoder.onnx',
  acoustic: 'audio_tokenizer/acoustic_encoder.onnx',
  semantic: 'audio_tokenizer/semantic_encoder.onnx',
  quantizer: 'audio_tokenizer/quantizer_encoder.onnx',
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
  private encoders?: EncoderSet
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

    const files = filesForDevice(device, profile)
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
    const embeddingsPath = device === 'webgpu' ? PATHS.embeddings : PATHS.embeddingsFp32
    this.embeddings = await createCachedSession(embeddingsPath, device)
    this.llm = await createCachedSession(PATHS.llm, device)
    this.heads = await createCachedSession(PATHS.heads, device)
    this.decoder = await createCachedSession(PATHS.decoder, device)
    await this.warmup()
  }

  private async warmup(): Promise<void> {
    if (!this.embeddings || !this.llm || !this.heads || !this.decoder) return
    try {
      const step = createBackboneStep(this.embeddings.ort, this.embeddings, this.llm, this.heads, {
        llmFloat16: PATHS.llm.includes('fp16'),
      })
      await step(new Int32Array(8 * 4).fill(1024), new Uint8Array(4).fill(1), 4, 0, 1)
      await this.decoder.session.run({
        codes: new this.embeddings.ort.Tensor('int64', new BigInt64Array(8 * 2), [8, 1, 2]),
      })
    } catch {
      void 0
    }
  }

  get cloningReady(): boolean {
    return Boolean(this.encoders)
  }

  async loadEncoders(onProgress?: (progress: SynthProgress) => void, device?: Device): Promise<void> {
    if (this.encoders) return
    if (device) this.device = device
    const files = filesForProfile('full').filter((file) => file.kind === 'encoder')
    await ensureCached(files, {
      concurrency: 2,
      onUpdate: (update) =>
        onProgress?.({
          stage: 'download',
          ratio: update.overallTotal ? update.overallLoaded / update.overallTotal : 0,
          detail: update.file.path,
        }),
    })
    const loadOne = async (name: string, path: string): Promise<SessionHandle> => {
      onProgress?.({ stage: 'load', ratio: 0, detail: `${name} encoder` })
      return createCachedSession(path, this.device)
    }
    const acoustic = await loadOne('acoustic', PATHS.acoustic)
    const semantic = await loadOne('semantic', PATHS.semantic)
    const quantizer = await loadOne('quantizer', PATHS.quantizer)
    this.encoders = { acoustic, semantic, quantizer }
  }

  async encodeReference(waveform24k: Float32Array): Promise<ReferenceCodes> {
    if (!this.encoders) throw new Error('Cloning encoders are not loaded')
    return encodeReference(waveform24k, this.encoders)
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

    const cpu = this.device === 'wasm'
    const estimated = options.numAudioTokens ?? estimateTargetTokens(text)
    const frames = cpu ? Math.min(estimated, 220) : estimated
    const steps = options.numSteps ?? (cpu ? 12 : 32)
    const guidanceScale = options.guidanceScale ?? (cpu ? 0 : 2)

    onProgress?.({ stage: 'tokenize', ratio: 0 })
    const prepared = await prepareInputs(text, frames, {
      lang: options.lang,
      instruct: options.instruct,
      refText: options.refText,
      refCodes: options.refCodes,
    })

    const step = createBackboneStep(this.embeddings.ort, this.embeddings, this.llm, this.heads, {
      llmFloat16: PATHS.llm.includes('fp16'),
    })
    onProgress?.({ stage: 'generate', ratio: 0 })
    const codes = await iterativeUnmaskCfg(
      prepared,
      frames,
      steps,
      { guidanceScale, tShift: 0.1, layerPenalty: 5, positionTemperature: 5, seed: 0 },
      step,
      (index, remaining) =>
        onProgress?.({
          stage: 'generate',
          ratio: index / steps,
          detail: `${remaining} frames masked`,
        }),
    )

    onProgress?.({ stage: 'decode', ratio: 0 })
    const codesTensor = new this.embeddings.ort.Tensor('int64', toBigInt64(codes), [8, 1, frames])
    const output = await this.decoder.session.run({ codes: codesTensor })
    const samples = toFloat32(output.waveform_24k)

    const refRms = options.refCodes?.rms
    if (refRms !== undefined && refRms < 0.1) {
      const gain = refRms / 0.1
      for (let i = 0; i < samples.length; i++) samples[i] *= gain
    }

    onProgress?.({ stage: 'done', ratio: 1 })
    return {
      samples,
      sampleRate: 24000,
      frames,
      textTokens: [],
      milliseconds: performance.now() - start,
    }
  }

  async generateStream(
    text: string,
    options: SynthOptions = {},
    onProgress?: (progress: SynthProgress) => void,
    onChunk?: (chunk: SynthChunk) => void,
  ): Promise<SynthResult> {
    const chunks = chunkText(text, { maxChars: options.maxChars ?? 240 })
    if (chunks.length <= 1) return this.generate(text, options, onProgress)

    const parts: Float32Array[] = []
    let frames = 0
    const start = performance.now()
    for (let i = 0; i < chunks.length; i++) {
      const result = await this.generate(chunks[i], options, (progress) =>
        onProgress?.({ ...progress, ratio: (i + progress.ratio) / chunks.length }),
      )
      const chunk: SynthChunk = {
        samples: result.samples,
        sampleRate: result.sampleRate,
        frames: result.frames,
        index: i,
        total: chunks.length,
        text: chunks[i],
      }
      parts.push(result.samples)
      frames += result.frames
      onChunk?.(chunk)
      onProgress?.({
        stage: 'generate',
        ratio: (i + 1) / chunks.length,
        detail: `part ${i + 1}/${chunks.length}`,
      })
    }
    onProgress?.({ stage: 'done', ratio: 1 })
    return {
      samples: concatFloat32(parts),
      sampleRate: 24000,
      frames,
      textTokens: [],
      milliseconds: performance.now() - start,
    }
  }

  async dispose(): Promise<void> {
    await releaseSession(this.embeddings ?? null)
    await releaseSession(this.llm ?? null)
    await releaseSession(this.heads ?? null)
    await releaseSession(this.decoder ?? null)
    if (this.encoders) {
      await releaseSession(this.encoders.acoustic)
      await releaseSession(this.encoders.semantic)
      await releaseSession(this.encoders.quantizer)
    }
    this.embeddings = undefined
    this.llm = undefined
    this.heads = undefined
    this.decoder = undefined
    this.encoders = undefined
  }
}
