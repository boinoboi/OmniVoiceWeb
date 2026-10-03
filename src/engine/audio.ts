export interface DecodedAudio {
  sampleRate: number
  samples: Float32Array
}

function readString(view: DataView, offset: number, length: number): string {
  let out = ''
  for (let i = 0; i < length; i++) out += String.fromCharCode(view.getUint8(offset + i))
  return out
}

export function decodeWav(data: ArrayBuffer): DecodedAudio {
  const view = new DataView(data)
  if (readString(view, 0, 4) !== 'RIFF' || readString(view, 8, 4) !== 'WAVE') {
    throw new Error('Not a RIFF/WAVE file')
  }

  let offset = 12
  let format = 1
  let channels = 1
  let sampleRate = 24000
  let bitsPerSample = 16
  let dataOffset = -1
  let dataLength = 0

  while (offset + 8 <= view.byteLength) {
    const id = readString(view, offset, 4)
    const size = view.getUint32(offset + 4, true)
    const body = offset + 8
    if (id === 'fmt ') {
      format = view.getUint16(body, true)
      channels = view.getUint16(body + 2, true)
      sampleRate = view.getUint32(body + 4, true)
      bitsPerSample = view.getUint16(body + 14, true)
    } else if (id === 'data') {
      dataOffset = body
      dataLength = size
    }
    offset = body + size + (size % 2)
  }

  if (dataOffset < 0) throw new Error('WAVE data chunk not found')

  const bytesPerSample = bitsPerSample / 8
  const frames = Math.floor(dataLength / (bytesPerSample * channels))
  const mono = new Float32Array(frames)

  for (let frame = 0; frame < frames; frame++) {
    let sum = 0
    for (let ch = 0; ch < channels; ch++) {
      const pos = dataOffset + (frame * channels + ch) * bytesPerSample
      if (format === 3 && bitsPerSample === 32) sum += view.getFloat32(pos, true)
      else if (bitsPerSample === 16) sum += view.getInt16(pos, true) / 32768
      else if (bitsPerSample === 24) {
        const b0 = view.getUint8(pos)
        const b1 = view.getUint8(pos + 1)
        const b2 = view.getInt8(pos + 2)
        sum += ((b2 << 16) | (b1 << 8) | b0) / 8388608
      } else if (bitsPerSample === 32) sum += view.getInt32(pos, true) / 2147483648
      else if (bitsPerSample === 8) sum += (view.getUint8(pos) - 128) / 128
      else throw new Error(`Unsupported WAV bit depth: ${bitsPerSample}`)
    }
    mono[frame] = sum / channels
  }

  return { sampleRate, samples: mono }
}

export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
  }

  write(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, Math.round(clamped * 32767), true)
  }
  return buffer
}

const sinc = (x: number): number => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))
const blackman = (i: number, n: number): number =>
  0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1))

export function resample(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to || input.length === 0) return input.slice()
  const ratio = to / from
  const outLength = Math.round(input.length * ratio)
  const output = new Float32Array(outLength)
  const taps = 16
  const half = taps / 2
  const cutoff = Math.min(1, ratio)

  for (let i = 0; i < outLength; i++) {
    const src = i / ratio
    const base = Math.floor(src)
    const frac = src - base
    let acc = 0
    let weight = 0
    for (let j = -half + 1; j <= half; j++) {
      const idx = base + j
      if (idx < 0 || idx >= input.length) continue
      const w = blackman(j + half, taps) * sinc((j - frac) * cutoff)
      acc += input[idx] * w
      weight += w
    }
    output[i] = weight !== 0 ? acc / weight : 0
  }
  return output
}
