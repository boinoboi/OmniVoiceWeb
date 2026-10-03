# AGENTS.md — OmniVoice Web

Durable working memory for agents (and humans). If context was compacted, read this file
first, then `docs/PLAN.md` (the live step-by-step plan), then the todo list.

## Project in one line

College capstone: **on-device multilingual code-switched TTS using quantized OmniVoice**, running
fully in the browser on ONNX Runtime Web (WebGPU, WASM fallback). Static site on GitHub Pages.
Not a `transformers.js` pipeline — OmniVoice has no transformers.js architecture; we run the ONNX
sub-models and implement the inference algorithm ourselves.

## Repo / deployment

- Repo: https://github.com/boinoboi/OmniVoiceWeb (public)
- Live: https://boinoboi.github.io/OmniVoiceWeb/ (GitHub Pages, Actions workflow on push to `main`)
- Base path comes from `GITHUB_REPOSITORY` / `BASE_PATH`; GH Pages deploys to `/OmniVoiceWeb/`.
- GH Pages cannot set COOP/COEP → ORT WASM runs single-threaded (`ort.env.wasm.numThreads = 1`).

## Commands

```bash
npm install
npm run dev
npm run build          # tsc -b && vite build
npm run preview
npm run lint           # oxlint
npm test               # vitest
npm run smoke -- <url> # puppeteer-core browser smoke (desktop/iPhone/iPad), uses system Chrome
npm run ort:copy       # copies ORT wasm+mjs into public/ort (git-ignored, run before build in CI)
```

Python harness lives in `tools/` (created during execution). See `docs/PLAN.md`.

## Code conventions (enforced by tsconfig)

- **No comments** unless genuinely necessary.
- `erasableSyntaxOnly`: no `enum`, no `namespace`, no constructor parameter properties.
- `verbatimModuleSyntax`: type-only imports must use `import type`.
- `noUnusedLocals` / `noUnusedParameters` are on.
- `src/engine/*` is framework-free and worker-safe. UI is in `src/ui/*`.

## Test recipes

- **WebGPU without a real GPU** (CI/headless): launch system Chrome with
  `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader`. Gives a software WebGPU adapter with
  `maxBufferSize = 1 GB`. Validates **correctness/op support, not speed**. Requires a secure
  context (`http://localhost` counts).
- Headless `--dump-dom` does not wait for async cache reads; use puppeteer (`scripts/smoke.mjs`)
  when the result depends on async work.

## Dev machine (this box)

- RTX 3090 (24 GB) + RTX 5060 Ti (16 GB), i9-14900K (24C/32T), 125 GB DDR5, 227 GB free on `/home`.
- HF CDN ~32 MB/s. The `cuda/` model dir has an fp16 LLM (~891 MB); root has fp16 embeddings/heads.
- Python 3.14 is installed; torch/omnivoice will likely need a 3.11/3.12 venv (use `uv` if present).

## Model contract (do not rediscover)

OmniVoice = Qwen3-0.6B backbone + 8-codebook codec (Higgs Audio V2) + **32-step masked-diffusion
unmask loop**, 24 kHz. Confirmed ONNX I/O:

| Sub-model | Inputs | Outputs |
|---|---|---|
| `audio_embeddings_encoder` | `input_ids` int64 `[B,8,S]`, `audio_mask` bool `[B,S]` | `inputs_embeds` f32 `[B,S,1024]` |
| `llm_decoder` | `inputs_embeds` `[B,S,1024]` (+ `attention_mask`,`position_ids`, empty `past_key_values`) | `hidden_states` `[B,S,1024]` |
| `audio_heads_decoder` | `hidden_states` `[B,S,1024]` | `logits` `[B,8,S,1025]` |
| Higgs encoders | `waveform_24k` / `waveform_16k` → codes | (cloning) |
| `higgs_decoder` | `codes` `[8,1,T]` int64 | `waveform_24k` |

- Audio mask id = `1024`; codebook weights `[8,8,6,6,4,4,2,2]`; 25 fps (downsample 960).
- External-data sidecar name = onnx basename + `.data`; ORT-web needs `sessionOptions.externalData`
  as an **array** for >2 GB models.
- `ort.env.wasm.wasmPaths = import.meta.env.BASE_URL + 'ort/'`.

## Prior-art gotchas (already paid for by others)

1. **Diffusion loop is precision-sensitive.** Reported: CUDA TF32 → incoherent noise, fp16 →
   different-but-valid, fp32 → faithful. INT8/INT4 quality unverified. This is our #1 risk and our
   research subject.
2. **Precompute reference voice codes offline** into `voices.json` → drops the 654 MB encoder and
   stabilizes short inputs.
3. fp16 is a **WebGPU** rung; ORT WASM kernels are fp32/int8, so fp16 buys size, not speed there.
4. VocoLoco's main model is ~2.3 GB (near fp32/fp16), **not** int4 — the 296 MB int4 export may be
   too lossy. Must A/B.

## Roadmap + active status

Full plan: `docs/PLAN.md`. Differentiation + prior art: `docs/PRIOR_ART.md`.
Architecture: `docs/ARCHITECTURE.md`. Results so far: `docs/RESULTS.md`.
Milestones tracked in the todo list.

**Status:** Phase 1 done (golden vectors + ORT-web parity). Next: Phase 2 Higgs codec.
Python harness: `tools/.venv` (Py 3.12), fixtures in `tools/golden/`, models in `tools/models/`.

**Checkpoints where the agent must message the user:** after parity proof, after first real audio,
after greedy-vs-CFG and int4-vs-fp16 A/B, after quantization variants, and before real-GPU/mobile
testing.

## Licensing

Site code MIT. OmniVoice weights **CC-BY-NC** (non-commercial). Never bundle weights; always fetch
from HF. Voice cloning requires consent — keep the notice in the UI.
