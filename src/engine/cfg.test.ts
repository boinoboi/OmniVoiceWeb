import { describe, expect, it } from 'vitest'
import { unmaskSchedule, timeSteps } from './algorithm'
import { combineText } from './prompt'
import { estimateTargetTokens } from './duration'

describe('cfg scheduling', () => {
  it('time steps are monotone from 0 to 1 with t_shift', () => {
    const ts = timeSteps(32, 0.1)
    expect(ts[0]).toBeCloseTo(0, 6)
    expect(ts[ts.length - 1]).toBeCloseTo(1, 6)
    for (let i = 1; i < ts.length; i++) expect(ts[i]).toBeGreaterThan(ts[i - 1])
  })

  it('schedule sums exactly to the total mask count', () => {
    for (const steps of [1, 4, 16, 32]) {
      const schedule = unmaskSchedule(8 * 80, steps, 0.1)
      expect(schedule.reduce((a, b) => a + b, 0)).toBe(8 * 80)
      expect(schedule.every((k) => k >= 0)).toBe(true)
    }
  })
})

describe('prompt helpers', () => {
  it('combines reference and target text', () => {
    expect(combineText('hello', 'ref text')).toBe('ref text hello')
  })

  it('strips spaces around CJK', () => {
    expect(combineText('你好 世界')).toBe('你好世界')
  })
})

describe('duration', () => {
  it('scales with text length and clamps', () => {
    expect(estimateTargetTokens('a')).toBe(8)
    expect(estimateTargetTokens('The quick brown fox jumps over the lazy dog.')).toBeGreaterThan(20)
  })
})
