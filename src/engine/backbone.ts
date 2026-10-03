import type { InferenceSession, Tensor } from 'onnxruntime-web'
import { AUDIO_VOCAB, NUM_CODEBOOKS, type BackboneStep } from './algorithm'
import { toFloat16Data, toFloat32 } from './dtype'

const HIDDEN = 1024

export interface OrtLike {
  Tensor: new (
    type: string,
    data: Float32Array | Uint16Array | BigInt64Array | Uint8Array,
    dims: readonly number[],
  ) => Tensor
}

export interface BackboneOptions {
  llmFloat16?: boolean
}

export interface BackboneSession {
  session: InferenceSession
  inputNames: readonly string[]
}

export function createBackboneStep(
  ortModule: OrtLike,
  embeddings: BackboneSession,
  llm: BackboneSession,
  heads: BackboneSession,
  options: BackboneOptions = {},
): BackboneStep {
  const llmNames = llm.inputNames
  const hasAttentionMask = llmNames.includes('attention_mask')
  const pastNames = llmNames.filter((name) => name.includes('past'))
  const llmFloat16 = options.llmFloat16 ?? false

  return async (inputIds, audioMask, seq, genStart, genFrames) => {
    const ids = new BigInt64Array(inputIds.length)
    for (let i = 0; i < inputIds.length; i++) ids[i] = BigInt(inputIds[i])

    const embedsOut = await embeddings.session.run({
      input_ids: new ortModule.Tensor('int64', ids, [1, NUM_CODEBOOKS, seq]),
      audio_mask: new ortModule.Tensor('bool', audioMask, [1, seq]),
    })
    const embeds = embedsOut.inputs_embeds
    const embedsData = embeds.data as Float32Array

    const feed: Record<string, Tensor> = {
      inputs_embeds: llmFloat16
        ? new ortModule.Tensor('float16', toFloat16Data(embedsData), [1, seq, HIDDEN])
        : new ortModule.Tensor('float32', embedsData, [1, seq, HIDDEN]),
    }
    if (hasAttentionMask) {
      feed.attention_mask = new ortModule.Tensor('int64', new BigInt64Array(seq).fill(1n), [1, seq])
    }
    for (const name of pastNames) {
      feed[name] = new ortModule.Tensor('float32', new Float32Array(0), [1, 8, 0, 128])
    }

    const hiddenOut = await llm.session.run(feed)
    const hidden = hiddenOut.hidden_states
    const hiddenData = llmFloat16 ? toFloat32(hidden) : (hidden.data as Float32Array)
    const headsOut = await heads.session.run({
      hidden_states: new ortModule.Tensor('float32', hiddenData, [1, seq, HIDDEN]),
    })

    const logits = headsOut.logits
    const data = logits.data as Float32Array
    const out = new Float32Array(NUM_CODEBOOKS * genFrames * AUDIO_VOCAB)
    for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
      for (let t = 0; t < genFrames; t++) {
        const src = (cb * seq + genStart + t) * AUDIO_VOCAB
        const dst = (cb * genFrames + t) * AUDIO_VOCAB
        out.set(data.subarray(src, src + AUDIO_VOCAB), dst)
      }
    }
    return out
  }
}
