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

export interface CfgConfig {
  guidanceScale: number
  tShift: number
  layerPenalty: number
  positionTemperature: number
  seed: number
}

export interface PreparedCfgInputs {
  condIds: Int32Array
  condMask: Uint8Array
  condSeq: number
  genStart: number
  uncondIds: Int32Array
  uncondMask: Uint8Array
}

export function timeSteps(numStep: number, tShift: number): Float64Array {
  const out = new Float64Array(numStep + 1)
  for (let i = 0; i <= numStep; i++) {
    const t = i / numStep
    out[i] = (tShift * t) / (1 + (tShift - 1) * t)
  }
  return out
}

export function unmaskSchedule(totalMask: number, numStep: number, tShift: number): number[] {
  const ts = timeSteps(numStep, tShift)
  let remaining = totalMask
  const out: number[] = []
  for (let step = 0; step < numStep; step++) {
    const num =
      step === numStep - 1
        ? remaining
        : Math.min(Math.ceil(totalMask * (ts[step + 1] - ts[step])), remaining)
    out.push(num)
    remaining -= num
  }
  return out
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function logSoftmaxInto(logits: Float32Array, offset: number, out: Float32Array): void {
  let max = -Infinity
  for (let v = 0; v < AUDIO_VOCAB; v++) if (logits[offset + v] > max) max = logits[offset + v]
  let sum = 0
  for (let v = 0; v < AUDIO_VOCAB; v++) sum += Math.exp(logits[offset + v] - max)
  const lse = max + Math.log(sum)
  for (let v = 0; v < AUDIO_VOCAB; v++) out[v] = logits[offset + v] - lse
}

function topKIndices(scores: Float32Array, k: number): number[] {
  const indices = Array.from({ length: scores.length }, (_, i) => i)
  indices.sort((a, b) => scores[b] - scores[a])
  return indices.slice(0, k)
}

export async function iterativeUnmaskCfg(
  inputs: PreparedCfgInputs,
  target: number,
  numSteps: number,
  cfg: CfgConfig,
  step: BackboneStep,
  onProgress?: (step: number, remaining: number) => void,
): Promise<Int32Array> {
  const { condIds, condMask, condSeq, genStart, uncondIds, uncondMask } = inputs
  const total = NUM_CODEBOOKS * target
  const schedule = unmaskSchedule(total, numSteps, cfg.tShift)
  const rng = mulberry32(cfg.seed)

  const targetIds = new Int32Array(total).fill(AUDIO_MASK_ID)
  const pred = new Int32Array(total)
  const score = new Float32Array(total)
  const clp = new Float32Array(AUDIO_VOCAB)
  const ulp = new Float32Array(AUDIO_VOCAB)
  const comb = new Float32Array(AUDIO_VOCAB)

  for (let s = 0; s < numSteps; s++) {
    const k = schedule[s]
    if (k <= 0) continue

    const cLogits = await step(condIds, condMask, condSeq, genStart, target)
    const useGuidance = cfg.guidanceScale > 0
    const uLogits = useGuidance ? await step(uncondIds, uncondMask, target, 0, target) : null

    for (let r = 0; r < total; r++) {
      logSoftmaxInto(cLogits, r * AUDIO_VOCAB, clp)
      if (uLogits) {
        logSoftmaxInto(uLogits, r * AUDIO_VOCAB, ulp)
        for (let v = 0; v < AUDIO_VOCAB; v++) comb[v] = clp[v] + cfg.guidanceScale * (clp[v] - ulp[v])
        logSoftmaxInto(comb, 0, clp)
      }
      clp[AUDIO_MASK_ID] = -Infinity

      let best = 0
      let bestValue = -Infinity
      for (let v = 0; v < AUDIO_VOCAB; v++) {
        if (clp[v] > bestValue) {
          bestValue = clp[v]
          best = v
        }
      }
      pred[r] = best
      const layer = Math.floor(r / target)
      let value = bestValue - layer * cfg.layerPenalty
      if (cfg.positionTemperature > 0) {
        const u = rng()
        const gumbel = -Math.log(-Math.log(u + 1e-10) + 1e-10)
        value = value / cfg.positionTemperature + gumbel
      }
      score[r] = value
    }

    for (let r = 0; r < total; r++) if (targetIds[r] !== AUDIO_MASK_ID) score[r] = -Infinity
    for (const index of topKIndices(score, k)) targetIds[index] = pred[index]

    for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
      for (let t = 0; t < target; t++) {
        const value = targetIds[cb * target + t]
        condIds[cb * condSeq + genStart + t] = value
        uncondIds[cb * target + t] = value
      }
    }
    let remaining = 0
    for (let r = 0; r < total; r++) if (targetIds[r] === AUDIO_MASK_ID) remaining++
    onProgress?.(s + 1, remaining)
  }

  return targetIds
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
