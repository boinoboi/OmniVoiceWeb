import { resample } from './audio'
import { toFloat32, toInt32 } from './dtype'
import { ort, type SessionHandle } from './ort'

export interface ReferenceCodes {
  data: Int32Array
  frames: number
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
  waveform24k: Float32Array,
  encoders: EncoderSet,
): Promise<ReferenceCodes> {
  const waveform16k = resample(waveform24k, 24000, 16000)

  const acoustic = await encoders.acoustic.session.run({
    waveform_24k: new ort.Tensor('float32', waveform24k, [1, 1, waveform24k.length]),
  })
  const semantic = await encoders.semantic.session.run({
    waveform_16k: new ort.Tensor('float32', waveform16k, [1, waveform16k.length]),
  })

  const acousticTensor = acoustic.acoustic_features
  const semanticTensor = semantic.semantic_features
  const frames = Math.min(acousticTensor.dims[2], semanticTensor.dims[2])
  const acousticData = sliceFeatureFrames(toFloat32(acousticTensor), acousticTensor.dims, frames)
  const semanticData = sliceFeatureFrames(toFloat32(semanticTensor), semanticTensor.dims, frames)

  const quantized = await encoders.quantizer.session.run({
    acoustic_features: new ort.Tensor('float32', acousticData, [1, acousticTensor.dims[1], frames]),
    semantic_features: new ort.Tensor('float32', semanticData, [1, semanticTensor.dims[1], frames]),
  })

  const codes = quantized.codes
  return { data: toInt32(codes), frames: codes.dims[codes.dims.length - 1] }
}
