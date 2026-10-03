import { describe, expect, it } from 'vitest'
import { filesToPrune } from './cache'
import { filesForProfile } from './manifest'

describe('filesToPrune', () => {
  it('keeps everything needed for the lite profile', () => {
    const lite = filesForProfile('lite').map((file) => file.path)
    expect(filesToPrune('lite', lite)).toEqual([])
    expect(filesToPrune('full', lite)).toEqual([])
  })

  it('evicts encoder files when downgrading to lite', () => {
    const full = filesForProfile('full').map((file) => file.path)
    const victims = filesToPrune('lite', full)
    expect(victims.length).toBe(full.length - filesForProfile('lite').length)
    expect(victims.every((path) => path.includes('audio_tokenizer/'))).toBe(true)
  })

  it('ignores paths that are not part of any profile', () => {
    expect(filesToPrune('full', ['stale/model.onnx'])).toEqual(['stale/model.onnx'])
  })
})
