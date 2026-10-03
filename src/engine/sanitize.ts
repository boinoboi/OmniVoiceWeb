export interface SanitizeOptions {
  terminalPunctuation?: string
}

const EMOJI = /\p{Extended_Pictographic}/gu
const URL = /\b(?:https?:\/\/|www\.)\S+/gi
const EMAIL = /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/gi
const HTML = /<[^>]*>/g

export function sanitizeForTts(text: string, options: SanitizeOptions = {}): string {
  const terminal = options.terminalPunctuation ?? '.'

  let out = text.normalize('NFKC')
  out = out.replace(HTML, ' ')
  out = out.replace(URL, ' ')
  out = out.replace(EMAIL, ' ')
  out = out.replace(EMOJI, ' ')
  out = out.replace(/\u{fe0f}/gu, '')
  out = out.replace(/[\u2018\u2019\u201b]/g, "'")
  out = out.replace(/[\u201c\u201d\u201f]/g, '"')
  out = out.replace(/[\u2013\u2014\u2212]/g, '-')
  out = out.replace(/[\u2026]/g, '.')
  out = out.replace(/[\u00a0\u2007\u202f\t]+/g, ' ')
  out = out.replace(/[\r\n]+/g, '. ')

  out = out.replace(/[^\p{L}\p{M}\p{N}\s.,!?;:'"%&()\-।]/gu, ' ')

  out = out.replace(/([.,!?;:])\1+/g, '$1')
  out = out.replace(/\s+([.,!?;:।])/g, '$1')
  out = out.replace(/([.,!?;:।])(?=[^\s.,!?;:।])/g, '$1 ')
  out = out.replace(/\.\s*\./g, '.')
  out = out.replace(/\s{2,}/g, ' ').trim()

  if (!out) return ''
  if (!/[.!?;:।]$/.test(out)) out += terminal
  return out
}
