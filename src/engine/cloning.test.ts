import { describe, expect, it } from 'vitest'
import { sliceFeatureFrames } from './cloning'

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
