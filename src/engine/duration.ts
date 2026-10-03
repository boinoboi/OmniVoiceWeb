const CJK = (code: number): boolean =>
  (code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3040 && code <= 0x30ff) || (code >= 0xac00 && code <= 0xd7af)

function charWeight(text: string): number {
  let weight = 0
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0
    if (code === 32) weight += 0.3
    else if (CJK(code)) weight += 1.6
    else if (code < 128) weight += 1.0
    else weight += 1.3
  }
  return weight
}

const LOW_THRESHOLD = 50
const BOOST_STRENGTH = 3

export function estimateTargetTokens(text: string, refFrames?: number): number {
  const weight = charWeight(text)
  const refWeight = charWeight('Nice to meet you.')
  const refTokens = refFrames && refFrames > 0 ? refFrames : 25
  let tokens = weight * (refTokens / refWeight)
  if (tokens < LOW_THRESHOLD) {
    tokens = LOW_THRESHOLD * (tokens / LOW_THRESHOLD) ** (1 / BOOST_STRENGTH)
  }
  return Math.max(8, Math.min(1200, Math.round(tokens)))
}
