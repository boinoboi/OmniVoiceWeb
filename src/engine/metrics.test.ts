import { describe, expect, it } from 'vitest'
import { analyzeAudio } from './metrics'

function tone(frequency: number, sampleRate: number, seconds: number, amplitude = 0.5): Float32Array {
  const out = new Float32Array(Math.floor(sampleRate * seconds))
  for (let i = 0; i < out.length; i++) out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / sampleRate)
  return out
}

describe('analyzeAudio', () => {
  it('finds silence', () => {
    const metrics = analyzeAudio(new Float32Array(4800), 24000)
    expect(metrics.rms).toBe(0)
    expect(metrics.peak).toBe(0)
    expect(metrics.speechBand).toBe(0)
  })

  it('estimates the centroid of a pure tone near its frequency', () => {
    const metrics = analyzeAudio(tone(1000, 24000, 1), 24000)
    expect(metrics.centroidHz).toBeGreaterThan(850)
    expect(metrics.centroidHz).toBeLessThan(1150)
    expect(metrics.speechBand).toBeGreaterThan(0.9)
  })

  it('puts most energy in the speech band for speech-like content', () => {
    const metrics = analyzeAudio(tone(800, 24000, 1), 24000)
    expect(metrics.speechBand).toBeGreaterThan(0.85)
  })

  it('reports low band energy for a very low tone', () => {
    const metrics = analyzeAudio(tone(60, 24000, 1), 24000)
    expect(metrics.speechBand).toBeLessThan(0.1)
  })

  it('reports duration from length and sample rate', () => {
    expect(analyzeAudio(new Float32Array(24000), 24000).durationMs).toBeCloseTo(1000, 3)
  })
})
