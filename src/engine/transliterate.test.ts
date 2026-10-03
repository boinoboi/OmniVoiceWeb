import { describe, expect, it } from 'vitest'
import { hasIndicScript, transliterate } from './transliterate'

describe('transliterate', () => {
  it('converts Hinglish to Devanagari while preserving English words', () => {
    const out = transliterate(
      'hi Kaise ho bhai, chalo ajj kuch karte hai. Lets try some new ai stuff shall we.',
    )
    expect(hasIndicScript(out)).toBe(true)
    expect(out).toContain('कैसे')
    expect(out).toContain('भाई')
    expect(out).toContain('चलो')
    expect(out).toContain('Lets try some new ai stuff shall we')
  })

  it('uses the dictionary for known words', () => {
    expect(transliterate('namaste dosto')).toContain('नमस्ते')
  })

  it('can keep English off', () => {
    const out = transliterate('hello try some', false)
    expect(hasIndicScript(out)).toBe(true)
  })

  it('leaves pure Indic text alone', () => {
    expect(transliterate('नमस्ते दोस्तो')).toBe('नमस्ते दोस्तो')
  })

  it('keeps common English words that collide with the Hindi dictionary', () => {
    const out = transliterate('Let us test the system. bhai aaj toh hum kuch karenge')
    expect(out).toContain('Let us test the system')
    expect(out).toContain('भाई')
    expect(out).toContain('करेंगे')
  })

  it('handles semi-Devanagari + semi-English input', () => {
    const out = transliterate('नमस्ते bhai this is a test')
    expect(out).toContain('नमस्ते')
    expect(out).toContain('भाई')
    expect(out).toContain('this is a test')
  })

  it('keeps fully Devanagari sentences unchanged', () => {
    expect(transliterate('नमस्ते, आप कैसे हैं?')).toBe('नमस्ते, आप कैसे हैं?')
  })

  it('keeps Latin languages (English/Spanish) and converts Hinglish', () => {
    const out = transliterate('Hola amigo, kaise ho? vamos a la playa')
    expect(out).toContain('Hola amigo')
    expect(out).toContain('vamos')
    expect(out).toContain('playa')
    expect(out).toContain('कैसे')
    expect(out).toContain('हो')
  })

  it('supports other Indic scripts', () => {
    expect(transliterate('namaste', { script: 'telugu' })).toMatch(/[\u0c00-\u0c7f]/)
    expect(transliterate('namaste', { script: 'tamil' })).toMatch(/[\u0b80-\u0bff]/)
    expect(transliterate('namaste', { script: 'bengali' })).toMatch(/[\u0980-\u09ff]/)
  })
})
