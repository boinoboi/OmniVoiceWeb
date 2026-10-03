import { describe, expect, it } from 'vitest'
import {
  HOP_LENGTH,
  computeRms,
  normalizeReference,
  sliceFeatureFrames,
  trimEdgeSilence,
  trimToHop,
} from './cloning'

describe('sliceFeatureFrames', () => {
  it('trims the last (time) dimension per channel', () => {
    const data = new Float32Array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    const out = sliceFeatureFrames(data, [1, 2, 5], 3)
    expect(Array.from(out)).toEqual([0, 1, 2, 10, 11, 12])
  })

  it('handles batch dimension > 1', () => {
    const data = new Float32Array([0, 1, 2, 3, 100, 101, 102, 103])
    const out = sliceFeatureFrames(data, [2, 1, 4], 2)
    expect(Array.from(out)).toEqual([0, 1, 100, 101])
  })
})

describe('reference preprocessing', () => {
  it('normalizes quiet audio up to the minimum rms', () => {
    const quiet = new Float32Array(1000).fill(0.01)
    const out = normalizeReference(quiet, 0.1)
    expect(computeRms(out)).toBeCloseTo(0.1, 5)
  })

  it('leaves loud audio untouched', () => {
    const loud = new Float32Array(1000).fill(0.5)
    expect(normalizeReference(loud, 0.1)).toBe(loud)
  })

  it('trims to a whole number of hops', () => {
    const samples = new Float32Array(HOP_LENGTH * 3 + 500)
    expect(trimToHop(samples).length).toBe(HOP_LENGTH * 3)
  })

  it('trims leading and trailing near-silence', () => {
    const samples = new Float32Array(100)
    samples.fill(0.001)
    for (let i = 20; i < 80; i++) samples[i] = 0.5
    const out = trimEdgeSilence(samples, 0.01)
    expect(out.length).toBe(60)
    expect(out[0]).toBeCloseTo(0.5, 5)
  })
})
