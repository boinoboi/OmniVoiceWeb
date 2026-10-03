import { describe, expect, it } from 'vitest'
import {
  AUDIO_MASK_ID,
  AUDIO_VOCAB,
  NUM_CODEBOOKS,
  type BackboneStep,
  iterativeUnmaskCfg,
} from './algorithm'

const deterministicStep: BackboneStep = async (_ids, _mask, _seq, _genStart, genFrames) => {
  const logits = new Float32Array(NUM_CODEBOOKS * genFrames * AUDIO_VOCAB)
  for (let r = 0; r < NUM_CODEBOOKS * genFrames; r++) logits[r * AUDIO_VOCAB + (r % 1000)] = 10
  return logits
}

function makeInputs(genStart: number, target: number) {
  const condSeq = genStart + target
  const condIds = new Int32Array(NUM_CODEBOOKS * condSeq)
  const condMask = new Uint8Array(condSeq)
  const uncondIds = new Int32Array(NUM_CODEBOOKS * target)
  const uncondMask = new Uint8Array(target)
  return { condIds, condMask, condSeq, genStart, uncondIds, uncondMask }
}

describe('iterativeUnmaskCfg', () => {
  it('unmasks every position and returns valid codebook ids', async () => {
    const target = 6
    const inputs = makeInputs(2, target)
    const codes = await iterativeUnmaskCfg(inputs, target, 4, {
      guidanceScale: 2,
      tShift: 0.1,
      layerPenalty: 5,
      positionTemperature: 5,
      seed: 0,
    }, deterministicStep)

    expect(codes.length).toBe(NUM_CODEBOOKS * target)
    for (let r = 0; r < codes.length; r++) {
      expect(codes[r]).toBe(r % 1000)
      expect(codes[r]).toBeLessThan(AUDIO_MASK_ID)
    }
  })

  it('writes generated codes back into the conditional and unconditional sequences', async () => {
    const target = 6
    const inputs = makeInputs(2, target)
    const codes = await iterativeUnmaskCfg(inputs, target, 4, {
      guidanceScale: 2,
      tShift: 0.1,
      layerPenalty: 5,
      positionTemperature: 5,
      seed: 1,
    }, deterministicStep)

    for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
      for (let t = 0; t < target; t++) {
        const value = codes[cb * target + t]
        expect(inputs.condIds[cb * inputs.condSeq + inputs.genStart + t]).toBe(value)
        expect(inputs.uncondIds[cb * target + t]).toBe(value)
      }
    }
  })

  it('reports monotone progress down to zero remaining', async () => {
    const target = 8
    const inputs = makeInputs(1, target)
    const remaining: number[] = []
    await iterativeUnmaskCfg(inputs, target, 5, {
      guidanceScale: 2,
      tShift: 0.1,
      layerPenalty: 5,
      positionTemperature: 0,
      seed: 2,
    }, deterministicStep, (_step, left) => remaining.push(left))

    expect(remaining[remaining.length - 1]).toBe(0)
    for (let i = 1; i < remaining.length; i++) expect(remaining[i]).toBeLessThanOrEqual(remaining[i - 1])
  })
})
