export interface PrecomputedVoice {
  id: string
  name: string
  text: string
  frames: number
  rms: number
  codes: Int32Array
}

interface RawVoice {
  id: string
  name: string
  text: string
  frames: number
  rms: number
  codes: string
}

function decodeCodes(base64: string): Int32Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Int32Array(bytes.buffer, 0, bytes.byteLength >> 2)
}

export async function loadVoices(url: string): Promise<PrecomputedVoice[]> {
  try {
    const response = await fetch(url)
    if (!response.ok) return []
    const data = (await response.json()) as { voices?: RawVoice[] }
    return (data.voices ?? []).map((voice) => ({ ...voice, codes: decodeCodes(voice.codes) }))
  } catch {
    return []
  }
}
