import { encodeTokens } from './tokenizer'

export const AUDIO_MASK_ID = 1024
export const NUM_CODEBOOKS = 8

export interface ReferenceCodes {
  data: Int32Array
  frames: number
}

export interface PromptOptions {
  lang?: string
  instruct?: string
  refText?: string
  refCodes?: ReferenceCodes
}

export interface PreparedInputs {
  condIds: Int32Array
  condMask: Uint8Array
  condSeq: number
  genStart: number
  uncondIds: Int32Array
  uncondMask: Uint8Array
}

export function combineText(text: string, refText?: string): string {
  let full = refText ? `${refText.trim()} ${text.trim()}` : text.trim()
  full = full.replace(/[\r\n]+/g, '')
  full = full.replace(/\uff08/g, '(').replace(/\uff09/g, ')')
  full = full.replace(/[ \t]+/g, ' ')
  const cjk = '\\u4e00-\\u9fff'
  full = full.replace(new RegExp(`([${cjk}])\\s+|\\s+([${cjk}])`, 'g'), '$1$2')
  return full
}

async function styleTokens(options: PromptOptions): Promise<number[]> {
  const ids: number[] = []
  if (options.refCodes) ids.push(...(await encodeTokens('<|denoise|>')))
  ids.push(...(await encodeTokens('<|lang_start|>')))
  ids.push(...(await encodeTokens(options.lang ?? 'None')))
  ids.push(...(await encodeTokens('<|lang_end|>')))
  ids.push(...(await encodeTokens('<|instruct_start|>')))
  ids.push(...(await encodeTokens(options.instruct ?? 'None')))
  ids.push(...(await encodeTokens('<|instruct_end|>')))
  return ids
}

export async function prepareInputs(
  text: string,
  target: number,
  options: PromptOptions = {},
): Promise<PreparedInputs> {
  const prefix = await styleTokens(options)
  prefix.push(...(await encodeTokens('<|text_start|>')))
  prefix.push(...(await encodeTokens(combineText(text, options.refText))))
  prefix.push(...(await encodeTokens('<|text_end|>')))

  const refFrames = options.refCodes?.frames ?? 0
  const genStart = prefix.length + refFrames
  const condSeq = genStart + target

  const condIds = new Int32Array(NUM_CODEBOOKS * condSeq)
  const condMask = new Uint8Array(condSeq)
  for (let cb = 0; cb < NUM_CODEBOOKS; cb++) {
    const row = cb * condSeq
    for (let i = 0; i < prefix.length; i++) condIds[row + i] = prefix[i]
    if (options.refCodes) {
      for (let t = 0; t < refFrames; t++) {
        condIds[row + prefix.length + t] = options.refCodes.data[cb * refFrames + t]
      }
    }
    for (let t = 0; t < target; t++) condIds[row + genStart + t] = AUDIO_MASK_ID
  }
  for (let t = genStart; t < condSeq; t++) condMask[t] = 1

  const uncondIds = new Int32Array(NUM_CODEBOOKS * target)
  const uncondMask = new Uint8Array(target)
  uncondIds.fill(AUDIO_MASK_ID)
  uncondMask.fill(1)

  return { condIds, condMask, condSeq, genStart, uncondIds, uncondMask }
}
