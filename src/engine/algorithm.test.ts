import { describe, expect, it } from 'vitest'
import { AUDIO_MASK_ID, NUM_CODEBOOKS, iterativeUnmask, type BackboneStep } from './algorithm'

const FRAMES = 5
const VOCAB = 1025

const mockStep: BackboneStep = async (inputIds, _mask, _seq, genStart, genFrames) => {
  const logits = new Float32Array(NUM_CODEBOOKS * genFrames * VOCAB)
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    for (let t = 0; t < genFrames; t++) {
      const off = (cb * genFrames + t) * VOCAB
      const token = (inputIds[genStart + t] + cb + t + 1) % 1024
      logits[off + token] = 10
    }
  }
  return logits
}

describe('iterativeUnmask', () => {
  it('unmasks every frame and never returns the mask id', async () => {
    const result = await iterativeUnmask(
      { textTokens: [1, 2, 3], numAudioTokens: FRAMES, numSteps: 2 },
      mockStep,
    )
    expect(result.codes.length).toBe(NUM_CODEBOOKS * FRAMES)
    for (const code of result.codes) {
      expect(code).toBeGreaterThanOrEqual(0)
      expect(code).toBeLessThan(AUDIO_MASK_ID)
    }
    expect(result.textLength).toBe(3)
    expect(result.refFrames).toBe(0)
  })

  it('places prefix codes and masks only the generated region', async () => {
    const prefix = { data: [7, 7, 7, 7, 7, 7, 7, 7], frames: 1 }
    let sawMaskedPrefix = false
    const probe: BackboneStep = async (inputIds, mask, seq, genStart) => {
      if (inputIds[0] === AUDIO_MASK_ID) sawMaskedPrefix = true
      expect(mask[genStart - 1]).toBe(0)
      const logits = new Float32Array(NUM_CODEBOOKS * (seq - genStart) * VOCAB)
      logits[0] = 1
      return logits
    }
    const result = await iterativeUnmask(
      { textTokens: [1], numAudioTokens: 2, numSteps: 1, prefixCodes: prefix },
      probe,
    )
    expect(sawMaskedPrefix).toBe(false)
    expect(result.refFrames).toBe(1)
  })
})
