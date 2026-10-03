export const TOKENIZER_MODEL_ID = 'onnx-community/OmniVoice-Onnx'

interface TokenizerLike {
  encode(text: string, options?: { add_special_tokens?: boolean }): Promise<ArrayLike<number>> | ArrayLike<number>
}

let tokenizerPromise: Promise<TokenizerLike> | null = null

async function getTokenizer(): Promise<TokenizerLike> {
  if (!tokenizerPromise) {
    tokenizerPromise = import('@huggingface/transformers').then(
      ({ AutoTokenizer }) =>
        AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID) as unknown as TokenizerLike,
    )
  }
  return tokenizerPromise
}

export async function encodeTokens(text: string, addSpecial = false): Promise<number[]> {
  const tokenizer = await getTokenizer()
  const ids = await tokenizer.encode(text, { add_special_tokens: addSpecial })
  return Array.from(ids, (id) => Number(id))
}

export async function encodeText(text: string): Promise<number[]> {
  return encodeTokens(text, true)
}
