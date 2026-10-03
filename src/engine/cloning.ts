import { resample } from './audio'
import { toFloat32, toInt32 } from './dtype'
import type { SessionHandle } from './ort'

export const HOP_LENGTH = 960
export const MIN_REF_RMS = 0.1

export interface ReferenceCodes {
  data: Int32Array
  frames: number
  rms: number
}

export function computeRms(samples: Float32Array): number {
  let sumSq = 0
  for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i]
  return samples.length ? Math.sqrt(sumSq / samples.length) : 0
}

export function normalizeReference(samples: Float32Array, minRms = MIN_REF_RMS): Float32Array {
  const rms = computeRms(samples)
  if (rms <= 0 || rms >= minRms) return samples
  const gain = minRms / rms
  const out = new Float32Array(samples.length)
  for (let i = 0; i < samples.length; i++) out[i] = samples[i] * gain
  return out
}

export function trimToHop(samples: Float32Array, hop = HOP_LENGTH): Float32Array {
  const usable = samples.length - (samples.length % hop)
  return usable > 0 ? samples.subarray(0, usable) : samples
}

export function trimEdgeSilence(
  samples: Float32Array,
  threshold = 0.01,
): Float32Array {
  let peak = 0
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]))
  if (peak <= 0) return samples
  const level = threshold * peak
  let start = 0
  while (start < samples.length && Math.abs(samples[start]) < level) start++
  let end = samples.length
  while (end > start && Math.abs(samples[end - 1]) < level) end--
  return start === 0 && end === samples.length ? samples : samples.subarray(start, end)
}

export interface EncoderSet {
  acoustic: SessionHandle
  semantic: SessionHandle
  quantizer: SessionHandle
}

export function sliceFeatureFrames(
  data: Float32Array,
  dims: readonly number[],
  frames: number,
): Float32Array {
  const channels = dims[0] * dims[1]
  const total = dims[2]
  const out = new Float32Array(channels * frames)
  for (let channel = 0; channel < channels; channel++) {
    out.set(data.subarray(channel * total, channel * total + frames), channel * frames)
  }
  return out
}

export async function encodeReference(
  input: Float32Array,
  encoders: EncoderSet,
): Promise<ReferenceCodes> {
  const normalized = normalizeReference(input)
  const rms = computeRms(normalized)
  const waveform24k = trimToHop(trimEdgeSilence(normalized))
  const waveform16k = resample(waveform24k, 24000, 16000)

  const module = encoders.acoustic.ort
  const acoustic = await encoders.acoustic.session.run({
    waveform_24k: new module.Tensor('float32', waveform24k, [1, 1, waveform24k.length]),
  })
  const semantic = await encoders.semantic.session.run({
    waveform_16k: new module.Tensor('float32', waveform16k, [1, waveform16k.length]),
  })

  const acousticTensor = acoustic.acoustic_features
  const semanticTensor = semantic.semantic_features
  const frames = Math.min(acousticTensor.dims[2], semanticTensor.dims[2])
  const acousticData = sliceFeatureFrames(toFloat32(acousticTensor), acousticTensor.dims, frames)
  const semanticData = sliceFeatureFrames(toFloat32(semanticTensor), semanticTensor.dims, frames)

  const quantized = await encoders.quantizer.session.run({
    acoustic_features: new module.Tensor('float32', acousticData, [1, acousticTensor.dims[1], frames]),
    semantic_features: new module.Tensor('float32', semanticData, [1, semanticTensor.dims[1], frames]),
  })

  const codes = quantized.codes
  return { data: toInt32(codes), frames: codes.dims[codes.dims.length - 1], rms }
}
