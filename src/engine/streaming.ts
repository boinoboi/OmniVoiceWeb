export interface ChunkOptions {
  maxChars?: number
  minChars?: number
}

const TERMINATORS = new Set(['.', '!', '?', '。', '！', '？', '।', '॥', ';', '；'])

export function splitSentences(text: string): string[] {
  const sentences: string[] = []
  let current = ''
  for (const ch of text) {
    if (ch === '\n') {
      if (current.trim()) sentences.push(current.trim())
      current = ''
      continue
    }
    current += ch
    if (TERMINATORS.has(ch)) {
      if (current.trim()) sentences.push(current.trim())
      current = ''
    }
  }
  if (current.trim()) sentences.push(current.trim())
  return sentences
}

function hardSplit(text: string, maxChars: number): string[] {
  const parts: string[] = []
  let rest = text.trim()
  while (rest.length > maxChars) {
    const floor = Math.floor(maxChars * 0.5)
    let cut = rest.lastIndexOf(' ', maxChars)
    if (cut < floor) cut = rest.lastIndexOf(',', maxChars)
    if (cut < floor) cut = rest.lastIndexOf('，', maxChars)
    if (cut < floor) cut = maxChars
    parts.push(rest.slice(0, cut).trim())
    rest = rest.slice(cut).trim()
  }
  if (rest) parts.push(rest)
  return parts
}

export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  const maxChars = Math.max(1, options.maxChars ?? 240)
  const minChars = Math.max(0, options.minChars ?? 40)
  const sentences = splitSentences(text)

  const chunks: string[] = []
  let buffer = ''
  const flush = (): void => {
    if (buffer.trim()) chunks.push(buffer.trim())
    buffer = ''
  }

  for (const sentence of sentences) {
    if (sentence.length > maxChars) {
      flush()
      for (const part of hardSplit(sentence, maxChars)) chunks.push(part)
      continue
    }
    if (!buffer) {
      buffer = sentence
    } else if (buffer.length + 1 + sentence.length <= maxChars) {
      buffer = `${buffer} ${sentence}`
    } else {
      flush()
      buffer = sentence
    }
  }
  flush()

  const merged: string[] = []
  for (const chunk of chunks) {
    const last = merged[merged.length - 1]
    if (last !== undefined && last.length < minChars && last.length + 1 + chunk.length <= maxChars) {
      merged[merged.length - 1] = `${last} ${chunk}`
    } else {
      merged.push(chunk)
    }
  }
  return merged
}
