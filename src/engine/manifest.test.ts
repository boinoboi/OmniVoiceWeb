import { describe, expect, it } from 'vitest'
import { MODEL_FILES, bytesForProfile, filesForProfile } from './manifest'

describe('model manifest', () => {
  it('exposes unique file paths', () => {
    const paths = MODEL_FILES.map((f) => f.path)
    expect(new Set(paths).size).toBe(paths.length)
  })

  it('lite files are a subset of full files', () => {
    const full = new Set(filesForProfile('full').map((f) => f.path))
    for (const file of filesForProfile('lite')) {
      expect(full.has(file.path)).toBe(true)
    }
  })

  it('full profile is larger than lite profile', () => {
    expect(bytesForProfile('full')).toBeGreaterThan(bytesForProfile('lite'))
  })

  it('every file has a positive byte size and a hugging face url', () => {
    for (const file of MODEL_FILES) {
      expect(file.bytes).toBeGreaterThan(0)
      expect(file.url.startsWith('https://huggingface.co/')).toBe(true)
    }
  })
})
