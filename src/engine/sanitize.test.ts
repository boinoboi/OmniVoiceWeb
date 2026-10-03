import { describe, expect, it } from 'vitest'
import { sanitizeForTts } from './sanitize'

describe('sanitizeForTts', () => {
  it('strips emojis, urls, html and markdown', () => {
    const out = sanitizeForTts('Hello **world** 😀 visit https://x.com/a <b>now</b>')
    expect(out).not.toMatch(/\*\*|😀|https?|<b>/)
    expect(out).toContain('Hello')
    expect(out).toContain('world')
    expect(out).toContain('now')
  })

  it('collapses whitespace and adds terminal punctuation', () => {
    expect(sanitizeForTts('  hello    world  ')).toBe('hello world.')
  })

  it('normalizes repeated punctuation', () => {
    expect(sanitizeForTts('wow!!! really???')).toBe('wow! really?')
  })

  it('keeps Devanagari and danda', () => {
    expect(sanitizeForTts('नमस्ते दोस्तो।')).toBe('नमस्ते दोस्तो।')
  })

  it('turns newlines into sentence breaks', () => {
    expect(sanitizeForTts('first line\nsecond line')).toBe('first line. second line.')
  })
})
