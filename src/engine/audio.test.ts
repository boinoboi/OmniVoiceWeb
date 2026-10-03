import { describe, expect, it } from 'vitest'
import { decodeWav, encodeWav, resample } from './audio'

describe('audio', () => {
  it('round-trips WAV encode/decode within 16-bit tolerance', () => {
    const n = 1000
    const samples = new Float32Array(n)
    for (let i = 0; i < n; i++) samples[i] = Math.sin((2 * Math.PI * 440 * i) / 24000) * 0.8
    const decoded = decodeWav(encodeWav(samples, 24000))
    expect(decoded.sampleRate).toBe(24000)
    expect(decoded.samples.length).toBe(n)
    let max = 0
    for (let i = 0; i < n; i++) max = Math.max(max, Math.abs(decoded.samples[i] - samples[i]))
    expect(max).toBeLessThan(2 / 32767 + 1e-6)
  })

  it('downsamples 24k -> 16k at the expected length', () => {
    const samples = new Float32Array(24000)
    const out = resample(samples, 24000, 16000)
    expect(out.length).toBe(16000)
  })

  it('preserves a constant signal', () => {
    const samples = new Float32Array(5000).fill(0.5)
    const out = resample(samples, 24000, 16000)
    for (let i = 100; i < out.length - 100; i++) expect(Math.abs(out[i] - 0.5)).toBeLessThan(1e-3)
  })
})
