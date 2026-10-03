import type { Tensor } from 'onnxruntime-web'

export function f16ToF32(input: Uint16Array): Float32Array {
  const out = new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const h = input[i]
    const sign = (h & 0x8000) >> 15
    const exponent = (h & 0x7c00) >> 10
    const fraction = h & 0x03ff
    let value: number
    if (exponent === 0) value = fraction * 2 ** -24
    else if (exponent === 0x1f) value = fraction ? NaN : Infinity
    else value = (1 + fraction / 1024) * 2 ** (exponent - 15)
    out[i] = sign ? -value : value
  }
  return out
}

export function f32ToF16(input: Float32Array): Uint16Array {
  const out = new Uint16Array(input.length)
  const scratch = new Float32Array(1)
  const asInt = new Int32Array(scratch.buffer)
  for (let i = 0; i < input.length; i++) {
    scratch[0] = input[i]
    const bits = asInt[0]
    const sign = (bits >> 16) & 0x8000
    const exponent = ((bits >> 23) & 0xff) - 127 + 15
    const mantissa = bits & 0x7fffff
    if (exponent <= 0) out[i] = sign
    else if (exponent >= 0x1f) out[i] = sign | 0x7c00
    else out[i] = sign | (exponent << 10) | (mantissa >> 13)
  }
  return out
}

export function toFloat32(tensor: Tensor): Float32Array {
  if (tensor.type === 'float32') return tensor.data as Float32Array
  if (tensor.type === 'float16') {
    const data = tensor.data as Uint16Array
    if (typeof Float16Array !== 'undefined' && data instanceof Float16Array) {
      return Float32Array.from(data as unknown as ArrayLike<number>)
    }
    return f16ToF32(data)
  }
  throw new Error(`Unsupported tensor type: ${tensor.type}`)
}

export function toInt32(tensor: Tensor): Int32Array {
  if (tensor.type === 'int32') return tensor.data as Int32Array
  const data = tensor.data as ArrayLike<bigint | number>
  const out = new Int32Array(data.length)
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i])
  return out
}
