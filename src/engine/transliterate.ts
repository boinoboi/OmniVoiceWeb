import { ENGLISH_WORDS, ROMAN_TO_HINDI, SPECIALS } from './transliterateData'

const VOWELS: Record<string, string> = {
  aa: 'आ', ai: 'ऐ', au: 'औ', ee: 'ई', oo: 'ऊ', ou: 'औ',
  a: 'अ', e: 'ए', i: 'इ', o: 'ओ', u: 'उ',
}

const MATRAS: Record<string, string> = {
  aa: 'ा', ai: 'ै', au: 'ौ', ee: 'ी', oo: 'ू', ou: 'ौ',
  a: '', e: 'े', i: 'ि', o: 'ो', u: 'ु',
}

const CONSONANTS: Record<string, string> = {
  shree: 'श्री', ksha: 'क्ष', gnya: 'ज्ञ', dnya: 'ज्ञ', shra: 'श्र',
  thra: 'थ्र', ttra: 'त्र', bha: 'भ', cha: 'च', chh: 'छ', dha: 'ध',
  gha: 'घ', jha: 'झ', kha: 'ख', nka: 'ंक', pha: 'फ', sha: 'श',
  shh: 'ष', tha: 'थ', tra: 'त्र', nga: 'ंग', nha: 'न्ह', lha: 'ल्ह',
  bh: 'भ', ch: 'च', dh: 'ध', gh: 'घ', jh: 'झ', kh: 'ख', ph: 'फ',
  sh: 'श', th: 'थ', tr: 'त्र', ng: 'ंग', nn: 'ण', ny: 'ञ', rh: 'ढ',
  b: 'ब', c: 'क', d: 'द', f: 'फ', g: 'ग', h: 'ह', j: 'ज', k: 'क',
  l: 'ल', m: 'म', n: 'न', p: 'प', q: 'क़', r: 'र', s: 'स', t: 'त',
  v: 'व', w: 'व', x: 'क्स', y: 'य', z: 'ज़',
}

const DIGITS: Record<string, string> = {
  '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
  '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
}

const PUNCTUATION: Record<string, string> = { '.': '।', '|': '।', '||': '॥' }

const ENGLISH_PRIORITY = new Set([
  'a', 'i', 'hi', 'hello', 'hey', 'ok', 'okay', 'bye', 'ai', 'app', 'web', 'api', 'ok',
])

const isAlpha = (ch: string): boolean => /[a-z]/.test(ch)
const isDigit = (ch: string): boolean => ch >= '0' && ch <= '9'

function stripPunctuation(word: string): [string, string, string] {
  let prefix = ''
  let suffix = ''
  let start = 0
  let end = word.length
  while (start < end && !/[a-z0-9]/i.test(word[start])) prefix += word[start++]
  while (end > start && !/[a-z0-9]/i.test(word[end - 1])) suffix = word[--end] + suffix
  return [prefix, word.slice(start, end), suffix]
}

function match(map: Record<string, string>, text: string, pos: number, lengths: number[]): [number, string] | null {
  for (const length of lengths) {
    const chunk = text.slice(pos, pos + length)
    if (map[chunk] !== undefined) return [length, map[chunk]]
  }
  return null
}

function convertWord(word: string): string {
  const lower = word.toLowerCase().trim()
  if (!lower) return word
  if (ROMAN_TO_HINDI[lower] !== undefined) return ROMAN_TO_HINDI[lower]
  if (SPECIALS[lower] !== undefined) return SPECIALS[lower]

  const out: string[] = []
  let i = 0
  let lastWasConsonant = false
  while (i < lower.length) {
    const ch = lower[i]
    if (isDigit(ch)) {
      out.push(DIGITS[ch] ?? ch)
      lastWasConsonant = false
      i++
      continue
    }
    if (!isAlpha(ch)) {
      out.push(ch)
      lastWasConsonant = false
      i++
      continue
    }
    if (ch === 'n' && i + 1 < lower.length && !'aeiou'.includes(lower[i + 1]) && isAlpha(lower[i + 1]) && 'gkcdjtpb'.includes(lower[i + 1])) {
      out.push('ं')
      lastWasConsonant = false
      i++
      continue
    }
    const cons = match(CONSONANTS, lower, i, [5, 4, 3, 2, 1])
    if (cons) {
      if (lastWasConsonant) out.push('्')
      out.push(cons[1])
      i += cons[0]
      if (i < lower.length) {
        const matra = match(MATRAS, lower, i, [2, 1])
        if (matra) {
          out.push(matra[1])
          i += matra[0]
          lastWasConsonant = false
          continue
        }
      }
      lastWasConsonant = true
      continue
    }
    const vowel = match(VOWELS, lower, i, [2, 1])
    if (vowel) {
      if (lastWasConsonant) {
        const matra = match(MATRAS, lower, i, [2, 1])
        if (matra) {
          out.push(matra[1])
          i += matra[0]
          lastWasConsonant = false
          continue
        }
      }
      out.push(vowel[1])
      i += vowel[0]
      lastWasConsonant = false
      continue
    }
    out.push(ch)
    lastWasConsonant = false
    i++
  }
  return out.join('')
}

function convertToken(token: string, keepEnglish: boolean): string {
  const [prefix, core, suffix] = stripPunctuation(token)
  if (!core) return token
  const lower = core.toLowerCase()
  const knownHindi = ROMAN_TO_HINDI[lower] !== undefined || SPECIALS[lower] !== undefined
  const keep =
    keepEnglish && (ENGLISH_PRIORITY.has(lower) || (!knownHindi && ENGLISH_WORDS[lower] === true))
  const converted = keep ? core : convertWord(core)
  const convertedSuffix = [...suffix].map((c) => PUNCTUATION[c] ?? c).join('')
  return `${prefix}${converted}${convertedSuffix}`
}

export function transliterate(text: string, keepEnglish = true): string {
  if (!text) return ''
  return text
    .split(/(\s+)/)
    .map((part) => (/\s/.test(part) || part === '' ? part : convertToken(part, keepEnglish)))
    .join('')
}

export function hasIndicScript(text: string): boolean {
  return /[\u0900-\u097f]/.test(text)
}
