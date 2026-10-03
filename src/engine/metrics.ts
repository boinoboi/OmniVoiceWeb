export interface AudioMetrics {
  durationMs: number
  rms: number
  peak: number
  centroidHz: number
  speechBand: number
}

const FFT_SIZE = 1024
const SPEECH_LOW = 300
const SPEECH_HIGH = 3400

function hann(index: number, size: number): number {
  return 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (size - 1))
}

export function analyzeAudio(samples: Float32Array, sampleRate: number): AudioMetrics {
  const durationMs = (samples.length / sampleRate) * 1000
  if (samples.length === 0) return { durationMs, rms: 0, peak: 0, centroidHz: 0, speechBand: 0 }

  let peak = 0
  let sumSq = 0
  for (let i = 0; i < samples.length; i++) {
    const value = samples[i]
    const abs = Math.abs(value)
    if (abs > peak) peak = abs
    sumSq += value * value
  }
  const rms = Math.sqrt(sumSq / samples.length)

  const size = Math.min(FFT_SIZE, samples.length)
  const start = Math.max(0, Math.floor((samples.length - size) / 2))
  const bins = size / 2
  const power = new Float64Array(bins + 1)

  for (let k = 0; k <= bins; k++) {
    let re = 0
    let im = 0
    const step = (2 * Math.PI * k) / size
    for (let n = 0; n < size; n++) {
      const value = samples[start + n] * hann(n, size)
      const angle = step * n
      re += value * Math.cos(angle)
      im -= value * Math.sin(angle)
    }
    power[k] = re * re + im * im
  }

  let centroidNum = 0
  let centroidDen = 0
  let speechEnergy = 0
  let totalEnergy = 0
  for (let k = 1; k <= bins; k++) {
    const hz = (k * sampleRate) / size
    const magnitude = Math.sqrt(power[k])
    centroidNum += hz * magnitude
    centroidDen += magnitude
    if (hz >= 20 && hz <= sampleRate * 0.45) totalEnergy += power[k]
    if (hz >= SPEECH_LOW && hz <= SPEECH_HIGH) speechEnergy += power[k]
  }

  return {
    durationMs,
    rms,
    peak,
    centroidHz: centroidDen > 0 ? centroidNum / centroidDen : 0,
    speechBand: totalEnergy > 0 ? speechEnergy / totalEnergy : 0,
  }
}
