import type { Tensor } from 'onnxruntime-web'
import { AUDIO_VOCAB, NUM_CODEBOOKS, type BackboneStep } from './algorithm'

const HIDDEN = 1024

export interface OrtLike {
  Tensor: new (
    type: string,
    data: Float32Array | BigInt64Array | Uint8Array,
    dims: readonly number[],
  ) => Tensor
}

export interface RunnableSession {
  run(
    feeds: Record<string, Tensor>,
    outputNames?: readonly string[],
  ): Promise<Record<string, Tensor>>
  inputNames: readonly string[]
}

export function createBackboneStep(
  ortModule: OrtLike,
  embeddings: RunnableSession,
  llm: RunnableSession,
  heads: RunnableSession,
): BackboneStep {
  const llmNames = llm.inputNames
  const hasAttentionMask = llmNames.includes('attention_mask')
  const pastNames = llmNames.filter((name) => name.includes('past'))

  return async (inputIds, audioMask, seq, genStart, genFrames) => {
    const ids = new BigInt64Array(inputIds.length)
    for (let i = 0; i < inputIds.length; i++) ids[i] = BigInt(inputIds[i])

    const embedsOut = await embeddings.run({
      input_ids: new ortModule.Tensor('int64', ids, [1, NUM_CODEBOOKS, seq]),
      audio_mask: new ortModule.Tensor('bool', audioMask, [1, seq]),
    })
    const embeds = embedsOut.inputs_embeds

    const feed: Record<string, Tensor> = {
      inputs_embeds: new ortModule.Tensor('float32', embeds.data as Float32Array, [1, seq, HIDDEN]),
    }
    if (hasAttentionMask) {
      feed.attention_mask = new ortModule.Tensor('int64', new BigInt64Array(seq).fill(1n), [1, seq])
    }
    for (const name of pastNames) {
      feed[name] = new ortModule.Tensor('float32', new Float32Array(0), [1, 8, 0, 128])
    }

    const hiddenOut = await llm.run(feed)
    const hidden = hiddenOut.hidden_states
    const headsOut = await heads.run({
      hidden_states: new ortModule.Tensor('float32', hidden.data as Float32Array, [1, seq, HIDDEN]),
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
