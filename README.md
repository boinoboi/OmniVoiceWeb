# OmniVoice Web

Multilingual zero-shot text-to-speech that runs **entirely in your browser** on **WebGPU**.
No server, no uploads, no API keys — the model runs on your own device and audio never leaves it.

Built on [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa), using a custom TypeScript
inference runtime on top of `onnxruntime-web`. The shipped `llm_decoder` is **re-exported with
bidirectional attention** — the public ONNX export is causal, which makes the model ignore the text
prompt (see [Results](#results)).

## Features

- **Code-switching**: mix two (or more) languages inside one sentence and keep a single,
  consistent voice across every language boundary
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
                     ├── llm_decoder        (Qwen3-0.6B, bidirectional attention, int4)
                     └── audio_heads_decoder
```

The loop uses **classifier-free guidance** (cond + uncond), a shifted-timestep schedule, a
per-codebook layer penalty and Gumbel position sampling — ported from the reference. Voice cloning
adds the Higgs encoders (`acoustic`, `semantic`, `quantizer`) to encode a reference clip into
prefix codes for the loop.

## Results

Round-trip Whisper WER and speech-band energy for *"The quick brown fox jumps over the lazy dog."*,
measured across the precision ladder in Python and in the browser (`tools/evaluate.py`):

| variant | size | runtime | WER |
|---|---|---|---|
| PyTorch reference | — | CUDA | 0.00 |
| fp32 bidirectional | 1.77 GB | CUDA | 0.00 |
| fp16 bidirectional | 885 MB | CUDA | 0.00 |
| **int4 bidirectional** | **280 MB** | **WebGPU (browser)** | **0.00** |

The browser default is the **int4** backbone (MatMulNBits, f32 activations, no `shader-f16`
required) — 6.3× smaller than fp32 at identical task quality. The critical fix was re-exporting the
LLM with a full bidirectional attention mask (`tools/export_llm.py`); the public causal export
produced non-speech even with the correct algorithm.

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

## Device support

- **Desktop**: Chrome/Edge 113+, Safari 18+, Firefox 141+ (WebGPU required for good speed)
- **iOS/iPadOS**: Safari 18+ (WebGPU). Older iOS falls back to CPU/WASM and is very slow.
- **Android**: Chrome with WebGPU enabled

The site is a responsive SPA with safe-area insets and 16px inputs (no iOS focus zoom), so it
works on phones and tablets. The real constraint on mobile is **RAM**, not the UI: the Lite
profile is ~470 MB and Full (which adds the cloning encoders) is ~1.12 GB. Lite is recommended on
phones; Full voice cloning may be unstable on low-memory devices. The app detects device memory,
GPU buffer limits and `navigator.gpu` availability and warns accordingly.

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
