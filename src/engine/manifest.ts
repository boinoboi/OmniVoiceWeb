export const HF_REPO = 'onnx-community/OmniVoice-Onnx'
export const HF_REVISION = 'main'
export const HF_BASE = `https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/`

export type Profile = 'lite' | 'full'
export type AssetKind = 'backbone' | 'tokenizer' | 'decoder' | 'encoder'

export interface ModelFile {
  path: string
  url: string
  bytes: number
  kind: AssetKind
  requires: Profile
}

const file = (
  path: string,
  bytes: number,
  kind: AssetKind,
  requires: Profile = 'lite',
): ModelFile => ({ path, url: HF_BASE + path, bytes, kind, requires })

export const MODEL_FILES: ModelFile[] = [
  file('int4/audio_embeddings_encoder.onnx', 2363, 'backbone'),
  file('int4/audio_embeddings_encoder.onnx.data', 87160832, 'backbone'),
  file('int4/audio_heads_decoder.onnx', 4462676, 'backbone'),
  file('int4/llm_decoder.onnx', 298798, 'backbone'),
  file('int4/llm_decoder.onnx.data', 296484864, 'backbone'),

  file('int4/tokenizer.json', 11423986, 'tokenizer'),
  file('int4/tokenizer_config.json', 533, 'tokenizer'),
  file('int4/config.json', 1416, 'tokenizer'),
  file('int4/genai_config.json', 1501, 'tokenizer'),
  file('int4/model_config.json', 341, 'tokenizer'),
  file('int4/omnivoice_manifest.json', 3517, 'tokenizer'),
  file('int4/chat_template.jinja', 4168, 'tokenizer'),

  file('audio_tokenizer/fp16/higgs_decoder.onnx', 308494, 'decoder'),
  file('audio_tokenizer/fp16/higgs_decoder.onnx.data', 43100160, 'decoder'),
  file('audio_tokenizer/fp16/model_config.json', 348, 'decoder'),

  file('audio_tokenizer/fp16/acoustic_encoder.onnx', 295846, 'encoder', 'full'),
  file('audio_tokenizer/fp16/acoustic_encoder.onnx.data', 102615040, 'encoder', 'full'),
  file('audio_tokenizer/fp16/semantic_encoder.onnx', 334295, 'encoder', 'full'),
  file('audio_tokenizer/fp16/semantic_encoder.onnx.data', 218235904, 'encoder', 'full'),
  file('audio_tokenizer/fp16/quantizer_encoder.onnx', 6091791, 'encoder', 'full'),
]

export const filesForProfile = (profile: Profile): ModelFile[] =>
  MODEL_FILES.filter((f) => f.requires === 'lite' || profile === 'full')

export const bytesForProfile = (profile: Profile): number =>
  filesForProfile(profile).reduce((total, f) => total + f.bytes, 0)

export const fileByPath = (path: string): ModelFile | undefined =>
  MODEL_FILES.find((f) => f.path === path)

export const KIND_LABELS: Record<AssetKind, string> = {
  backbone: 'Qwen3 backbone (int4)',
  tokenizer: 'Tokenizer + config',
  decoder: 'Higgs audio decoder (fp16)',
  encoder: 'Higgs cloning encoders (fp16)',
}

export const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i++
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[i]}`
}
