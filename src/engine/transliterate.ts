import { ENGLISH_WORDS } from './transliterateData'
import { SCRIPTS, type ScriptMaps } from './transliterateScripts'

export interface TransliterateOptions {
  keepEnglish?: boolean
  script?: string
}

export const SUPPORTED_SCRIPTS: { id: string; label: string }[] = [
  { id: 'devanagari', label: 'Hindi / Marathi (Devanagari)' },
  { id: 'bengali', label: 'Bengali' },
  { id: 'gujarati', label: 'Gujarati' },
  { id: 'gurmukhi', label: 'Punjabi (Gurmukhi)' },
  { id: 'tamil', label: 'Tamil' },
  { id: 'telugu', label: 'Telugu' },
  { id: 'kannada', label: 'Kannada' },
  { id: 'malayalam', label: 'Malayalam' },
  { id: 'oriya', label: 'Odia' },
]

const ENGLISH_PRIORITY = new Set([
  'a', 'i', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'so', 'of', 'in', 'on', 'at', 'by',
  'for', 'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'do',
  'does', 'did', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 'should', 'may', 'might',
  'must', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
  'our', 'you', 'your', 'my', 'not', 'no', 'yes', 'let', 'get', 'got', 'make', 'made', 'take',
  'test', 'system', 'data', 'model', 'code', 'ai', 'app', 'web', 'api', 'hello', 'hi', 'hey',
  'ok', 'okay', 'bye', 'please', 'thanks', 'thank', 'new', 'good', 'bad', 'fast', 'slow', 'time',
  'work', 'use', 'using', 'user', 'file', 'type', 'name', 'text', 'audio', 'voice', 'video',
  'image', 'link', 'run', 'try', 'also', 'just', 'only', 'very', 'more', 'most', 'much', 'many',
  'some', 'any', 'all', 'one', 'two', 'three', 'here', 'there', 'now', 'next', 'last', 'first',
  'well', 'way', 'thing', 'things', 'people', 'sorry', 'sure', 'about', 'after', 'before',
  'because', 'what', 'when', 'where', 'who', 'how', 'why', 'which', 'while', 'into', 'over',
  'under', 'up', 'down', 'out', 'off', 'again', 'still', 'even', 'back', 'same', 'other', 'each',
  'every',
])

const INDIC_RANGES =
  /[\u0900-\u097f\u0980-\u09ff\u0a00-\u0a7f\u0a80-\u0aff\u0b00-\u0b7f\u0b80-\u0bff\u0c00-\u0c7f\u0c80-\u0cff\u0d00-\u0d7f]/

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

function match(
  map: Record<string, string>,
  text: string,
  pos: number,
  lengths: number[],
): [number, string] | null {
  for (const length of lengths) {
    const chunk = text.slice(pos, pos + length)
    if (map[chunk] !== undefined) return [length, map[chunk]]
  }
  return null
}

function convertWord(word: string, maps: ScriptMaps): string {
  const lower = word.toLowerCase().trim()
  if (!lower) return word
  if (maps.dictionary[lower] !== undefined) return maps.dictionary[lower]
  if (maps.specials[lower] !== undefined) return maps.specials[lower]

  const out: string[] = []
  let i = 0
  let lastWasConsonant = false
  while (i < lower.length) {
    const ch = lower[i]
    if (isDigit(ch)) {
      out.push(maps.digits[ch] ?? ch)
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
    if (
      ch === 'n' &&
      i + 1 < lower.length &&
      !'aeiou'.includes(lower[i + 1]) &&
      isAlpha(lower[i + 1]) &&
      'gkcdjtpb'.includes(lower[i + 1])
    ) {
      out.push(maps.anusvara)
      lastWasConsonant = false
      i++
      continue
    }
    const cons = match(maps.consonants, lower, i, [5, 4, 3, 2, 1])
    if (cons) {
      if (lastWasConsonant) out.push(maps.virama)
      out.push(cons[1])
      i += cons[0]
      if (i < lower.length) {
        const matra = match(maps.matras, lower, i, [2, 1])
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
    const vowel = match(maps.vowels, lower, i, [2, 1])
    if (vowel) {
      if (lastWasConsonant) {
        const matra = match(maps.matras, lower, i, [2, 1])
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

function convertToken(token: string, keepEnglish: boolean, maps: ScriptMaps): string {
  const [prefix, core, suffix] = stripPunctuation(token)
  if (!core) return token
  const lower = core.toLowerCase()
  const knownIndic = maps.dictionary[lower] !== undefined || maps.specials[lower] !== undefined
  const keep =
    keepEnglish && (ENGLISH_PRIORITY.has(lower) || (!knownIndic && ENGLISH_WORDS[lower] === true))
  const converted = keep ? core : convertWord(core, maps)
  const convertedSuffix = [...suffix].map((c) => (c === '.' || c === '|' ? maps.danda : c)).join('')
  return `${prefix}${converted}${convertedSuffix}`
}

export function transliterate(
  text: string,
  keepEnglishOrOptions: boolean | TransliterateOptions = true,
  script = 'devanagari',
): string {
  const options: TransliterateOptions =
    typeof keepEnglishOrOptions === 'boolean'
      ? { keepEnglish: keepEnglishOrOptions, script }
      : keepEnglishOrOptions
  const keepEnglish = options.keepEnglish ?? true
  const maps = SCRIPTS[options.script ?? 'devanagari'] ?? SCRIPTS.devanagari
  if (!text) return ''
  return text
    .split(/(\s+)/)
    .map((part) => (part === '' || /\s/.test(part) ? part : convertToken(part, keepEnglish, maps)))
    .join('')
}

export function hasIndicScript(text: string): boolean {
  return INDIC_RANGES.test(text)
}
