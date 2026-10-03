import { describe, expect, it } from 'vitest'
import { chunkText, splitSentences } from './streaming'

describe('splitSentences', () => {
  it('splits on latin and devanagari terminators', () => {
    expect(splitSentences('One. Two! Three?')).toEqual(['One.', 'Two!', 'Three?'])
    expect(splitSentences('नमस्ते। आप कैसे हैं？')).toEqual(['नमस्ते।', 'आप कैसे हैं？'])
  })

  it('splits on newlines', () => {
    expect(splitSentences('He said hi.\nNext')).toEqual(['He said hi.', 'Next'])
  })

  it('drops empty fragments', () => {
    expect(splitSentences('\n\n  A  \n\n')).toEqual(['A'])
  })
})

describe('chunkText', () => {
  it('merges short sentences up to maxChars', () => {
    expect(chunkText('Hi. Ok. Bye.', { maxChars: 240, minChars: 40 })).toEqual(['Hi. Ok. Bye.'])
  })

  it('splits when exceeding maxChars', () => {
    const text = 'aaaaaaaaaa. bbbbbbbbbb. cccccccccc.'
    const chunks = chunkText(text, { maxChars: 24, minChars: 0 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) expect(chunk.length).toBeLessThanOrEqual(24)
  })

  it('hard splits a single over-long sentence at word boundaries', () => {
    const words = Array.from({ length: 60 }, () => 'word').join(' ')
    const chunks = chunkText(words, { maxChars: 40, minChars: 0 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) expect(chunk.length).toBeLessThanOrEqual(40)
    expect(chunks.join(' ')).toBe(words)
  })

  it('never returns empty chunks for whitespace', () => {
    expect(chunkText('   \n  ', {})).toEqual([])
  })

  it('preserves code-switched content', () => {
    const chunks = chunkText('I am going to the बाज़ार. Then I will call you.', { maxChars: 240 })
    expect(chunks.join(' ')).toContain('बाज़ार')
  })
})
