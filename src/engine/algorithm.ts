export const NUM_CODEBOOKS = 8
export const AUDIO_VOCAB = 1025
export const AUDIO_MASK_ID = 1024
export const CODEBOOK_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]

export interface PrefixCodes {
  data: ArrayLike<number>
  frames: number
}

export interface GenerationRequest {
  textTokens: readonly number[]
  numAudioTokens: number
  numSteps: number
  prefixCodes?: PrefixCodes
}

export type BackboneStep = (
  inputIds: Int32Array,
  audioMask: Uint8Array,
  seq: number,
  genStart: number,
  genFrames: number,
) => Promise<Float32Array>

export interface GenerationResult {
  codes: Int32Array
  frames: number
  textLength: number
  refFrames: number
}

function argmax(logits: Float32Array, offset: number, count: number): number {
  let best = 0
  let bestValue = -Infinity
  for (let v = 0; v < count; v++) {
    const value = logits[offset + v]
    if (value > bestValue) {
      bestValue = value
      best = v
    }
  }
  return best
}

export async function iterativeUnmask(
  request: GenerationRequest,
  step: BackboneStep,
  onProgress?: (step: number, remaining: number) => void,
  onStep?: (stepIndex: number, logits: Float32Array) => void,
): Promise<GenerationResult> {
  const { textTokens, numAudioTokens, numSteps, prefixCodes } = request
  const textLength = textTokens.length
  const refFrames = prefixCodes?.frames ?? 0
  const genStart = textLength + refFrames
  const seq = genStart + numAudioTokens

  const inputIds = new Int32Array(NUM_CODEBOOKS * seq)
  const audioMask = new Uint8Array(seq)

  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    const row = cb * seq
    for (let i = 0; i < textLength; i++) inputIds[row + i] = textTokens[i]
  }
  if (prefixCodes) {
    for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
      const row = cb * seq
      for (let t = 0; t < refFrames; t++) {
        inputIds[row + genStart - refFrames + t] = prefixCodes.data[cb * refFrames + t]
      }
    }
  }
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    const row = cb * seq
    for (let t = genStart; t < seq; t++) inputIds[row + t] = AUDIO_MASK_ID
  }
  for (let t = genStart; t < seq; t++) audioMask[t] = 1

  const weights = CODEBOOK_WEIGHTS.slice()
  const weightSum = weights.reduce((sum, w) => sum + w, 0)
  for (let i = 0; i < weights.length; i++) weights[i] /= weightSum

  const confidence = new Float32Array(numAudioTokens)
  const masked: number[] = []
  let remaining = numAudioTokens

  for (let s = 0; s < numSteps && remaining > 0; s++) {
    const logits = await step(inputIds, audioMask, seq, genStart, numAudioTokens)
    onStep?.(s, logits)

    confidence.fill(0)
    for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
      const base = cb * numAudioTokens * AUDIO_VOCAB
      for (let t = 0; t < numAudioTokens; t++) {
        const off = base + t * AUDIO_VOCAB
        let max = -Infinity
        for (let v = 0; v < AUDIO_MASK_ID; v++) if (logits[off + v] > max) max = logits[off + v]
        let sum = 0
        for (let v = 0; v < AUDIO_MASK_ID; v++) sum += Math.exp(logits[off + v] - max)
        confidence[t] += (1 / sum) * weights[cb]
      }
    }

    masked.length = 0
    for (let t = 0; t < numAudioTokens; t++) {
      if (inputIds[genStart + t] === AUDIO_MASK_ID) masked.push(t)
    }
    if (masked.length === 0) break

    const remainingSteps = Math.max(1, numSteps - s)
    const nThis = Math.max(1, Math.ceil(masked.length / remainingSteps))
    masked.sort((a, b) => confidence[b] - confidence[a])
    const chosen = masked.slice(0, nThis)

    for (const t of chosen) {
      for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
        const off = (cb * numAudioTokens + t) * AUDIO_VOCAB
        inputIds[cb * seq + genStart + t] = argmax(logits, off, AUDIO_MASK_ID)
      }
    }
    remaining -= chosen.length
    onProgress?.(s + 1, remaining)
  }

  const leftover: number[] = []
  for (let t = 0; t < numAudioTokens; t++) {
    if (inputIds[genStart + t] === AUDIO_MASK_ID) leftover.push(t)
  }
  if (leftover.length) {
    const logits = await step(inputIds, audioMask, seq, genStart, numAudioTokens)
    for (const t of leftover) {
      for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
        const off = (cb * numAudioTokens + t) * AUDIO_VOCAB
        inputIds[cb * seq + genStart + t] = argmax(logits, off, AUDIO_MASK_ID)
      }
    }
  }

  const codes = new Int32Array(NUM_CODEBOOKS * numAudioTokens)
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    for (let t = 0; t < numAudioTokens; t++) {
      codes[cb * numAudioTokens + t] = inputIds[cb * seq + genStart + t]
    }
  }
  return { codes, frames: numAudioTokens, textLength, refFrames }
}
