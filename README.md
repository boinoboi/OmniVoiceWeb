# OmniVoice Web

Multilingual zero-shot text-to-speech that runs **entirely in your browser** on **WebGPU**.
No server, no uploads, no API keys — the model runs on your own device and audio never leaves it.

Built on [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa) via the
[onnx-community/OmniVoice-Onnx](https://huggingface.co/onnx-community/OmniVoice-Onnx) export,
using a custom TypeScript inference runtime on top of `onnxruntime-web`.

## Features

- 600+ languages, zero-shot TTS
- Voice cloning from a short reference clip
- 100% local inference (WebGPU, WASM fallback)
- One-time model download, cached in the browser (Cache Storage)

## How it works

OmniVoice is a Qwen3-0.6B backbone that drives an 8-codebook audio codec through a 32-step
non-autoregressive unmasking loop. The ONNX export ships as separate sub-models, which we run
and orchestrate in the browser:

```
text → tokenize → iterative unmasking (32 steps) → audio codes → Higgs decoder → 24 kHz WAV
                     │
                     ├── audio_embeddings_encoder
                     ├── llm_decoder        (Qwen3, int4)
                     └── audio_heads_decoder
```

Voice cloning adds the Higgs encoders (`acoustic`, `semantic`, `quantizer`) to encode a
reference clip into prefix codes for the loop.

## Development

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # production build
npm test          # unit tests
npm run lint
```

The ONNX Runtime WebAssembly files are copied into `public/ort/` by `scripts/copy-ort.mjs`
(`npm run ort:copy`). They are git-ignored and copied automatically in CI before the build.

## Deployment

Pushing to `main` builds and deploys to GitHub Pages via `.github/workflows/deploy.yml`.
In the repository settings, set **Pages → Build and deployment → Source → GitHub Actions**.

GitHub Pages cannot set COOP/COEP headers, so the WASM backend uses a single thread. WebGPU is
unaffected.

## Model license

The application code in this repository is **MIT**. The OmniVoice **model weights are licensed
under CC-BY-NC** (non-commercial) due to their training data. Weights are downloaded directly
from the Hugging Face Hub and are never bundled here.

**Do not use voice cloning without the explicit consent of the person whose voice is being
cloned.** This project is intended for research and personal use.
