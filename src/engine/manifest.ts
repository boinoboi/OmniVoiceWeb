export const HF_REPO = 'onnx-community/OmniVoice-Onnx'
export const HF_REVISION = 'main'
export const HF_BASE = `https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/`
export const BIDIR_BASE =
  'https://huggingface.co/asdasdsadscxzxc/omnivoice-web-bidir/resolve/main/'

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

const bidir = (
  path: string,
  bytes: number,
  kind: AssetKind,
  requires: Profile = 'lite',
): ModelFile => ({ path, url: BIDIR_BASE + path, bytes, kind, requires })

export const MODEL_FILES: ModelFile[] = [
  file('int4/audio_embeddings_encoder.onnx', 2363, 'backbone'),
  file('int4/audio_embeddings_encoder.onnx.data', 87160832, 'backbone'),
  file('int4/audio_heads_decoder.onnx', 4462676, 'backbone'),
  bidir('llm_decoder.onnx', 4380033, 'backbone'),
  bidir('llm_decoder.onnx.data', 1761869824, 'backbone'),

  file('int4/tokenizer.json', 11423986, 'tokenizer'),
  file('int4/tokenizer_config.json', 533, 'tokenizer'),
  file('int4/config.json', 1416, 'tokenizer'),
  file('int4/genai_config.json', 1501, 'tokenizer'),
  file('int4/model_config.json', 341, 'tokenizer'),
  file('int4/omnivoice_manifest.json', 3517, 'tokenizer'),
  file('int4/chat_template.jinja', 4168, 'tokenizer'),

  file('audio_tokenizer/higgs_decoder.onnx', 86500102, 'decoder'),

  file('audio_tokenizer/acoustic_encoder.onnx', 205546480, 'encoder', 'full'),
  file('audio_tokenizer/semantic_encoder.onnx', 436736856, 'encoder', 'full'),
  file('audio_tokenizer/quantizer_encoder.onnx', 12131293, 'encoder', 'full'),
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
  decoder: 'Higgs audio decoder (fp32)',
  encoder: 'Higgs cloning encoders (fp32)',
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
