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
Report outline: `docs/REPORT.md`. Milestones tracked in the todo list.

**Done since the milestone:** int4 bidir LLM is the browser default (280 MB, exact ASR);
idle GPU release (`engine/memory.ts`); eval harness (`tools/evaluate.py` → WER/spectral/RTF,
`eval/manifest.json` → `eval/results.json`); in-app Results dashboard; TTFA/RTF surfaced; README +
report outline. **Open:** cloning quality (synthetic refs only), WASM path, stable HF hosting,
progressive sharding, UTMOS.

**Status (latest, resume here):** **browser parity achieved on a real GPU.** Headful Chrome on the
3090 running the full in-browser pipeline (CFG + bidirectional fp32 LLM) gives exact ASR for
"The quick brown fox jumps over the lazy dog." (2.56 s, 87% speech-band). Reproduce:
`E2E_REALGPU=1 E2E_HEADFUL=1 E2E_PROFILE=/tmp/opencode/chrome-profile node scripts/e2e-tts.mjs
http://localhost:4173/` (headful is required; headless exposes no WebGPU adapter).

- **Browser uses the int4 bidir LLM (280 MB), exact ASR.** fp16 was ruled out (this WebGPU adapter
  lacks `shader-f16`); int4 MatMulNBits keeps f32 activations so needs no f16. Hosted at HF
  `asdasdsadscxzxc/omnivoice-web-bidir` (`BIDIR_BASE`, `manifest.ts`, `PATHS.llm`). fp32/fp16 also
  live there if needed.
- int4 made with `tools/.venv-quant` (Python 3.12, `onnxruntime==1.20.1`, `onnx==1.16.2`), legacy
  `MatMul4BitsQuantizer`; pass the raw `ModelProto` (double-wrapping an `ONNXModel` crashes with
  `'method' object is not iterable`). ORT 1.30's new int4 API is broken.
- `createBackboneStep` casts f32→f16→f32 around the LLM when `llmFloat16` (not used for int4).
- Duration now ports the reference boost (`low_threshold=50`, boost 3) — fox → 65 frames vs 66.
- Downloads have retries + HTTP Range resume (HF `ERR_NETWORK_CHANGED` blips).
- Short auto-voice ("Hello world.") still decodes to non-speech — needs a reference voice.

### THE critical finding (do not rediscover)
- The public `onnx-community/OmniVoice-Onnx` `llm_decoder` is **causal** → the model ignores the
  text prompt. OmniVoice is a masked-diffusion LM needing **full bidirectional** attention.
- Fix: `tools/export_llm.py` re-exports the Qwen3 core with a 4-D all-zero additive mask
  (non-causal). Result matches the PyTorch reference exactly.
- `tools/.venv-ref` has torch cu128 + `omnivoice`. `tools/.venv` has ORT-gpu, transformers,
  faster-whisper, onnx, onnxconverter-common.
- The browser fetches the re-exported model from HF `asdasdsadscxzxc/omnivoice-web-bidir`
  (`llm_decoder_fp16.onnx` + `.data`, 885 MB). `BIDIR_BASE` in `manifest.ts`.

### Reference commands (tools/)
```bash
tools/.venv/bin/python tools/generate_cfg.py --llm bidir/llm_decoder_fp16.onnx \
  --provider CUDAExecutionProvider --text "..." --target 66 --scale 2 --out out.wav
tools/.venv-ref/bin/python tools/export_llm.py --dtype fp32|fp16 --name llm_decoder_fp16.onnx
tools/.venv/bin/python tools/convert_fp16.py     # (converter path; prefer native export)
tools/.venv/bin/python tools/quantize_llm.py     # NOTE: ORT 1.30 int4 config is broken
```
ASR/WER check: faster-whisper `WhisperModel('base', cpu, int8)`; resample to 16k; `vad_filter=False`.

### Known gotchas (paid for)
- fp16 & int4 are **WebGPU-only**; ORT WASM lacks `GatherBlockQuantized` (int4 embeddings) and
  fp16 kernels → **the WASM fallback does not run the current backbone. WebGPU is required.**
- `audio_tokenizer/fp16/semantic_encoder.onnx` from onnx-community is malformed; use fp32.
- SwiftShader WebGPU e2e is ~45 min (too slow to iterate). Use real-GPU headless
  (`--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan`) or a short target.

### Next steps (in order)
1. ~~Validate browser CFG+bidir on a real GPU~~ **DONE** (exact ASR). Give the user a one-command
   tester + a phone test.
2. Host/deploy: point `BIDIR_BASE` at a stable repo (currently the throwaway HF `asdasdsadscxzxc`);
   reduce download size — **int4 MatMulNBits is the target** (ORT 1.30 quantizer broken; legacy
   removed — use an older ORT in a throwaway venv or hand-pack).
3. Voice cloning end-to-end test (engine + UI landed); add `voices.json` precompute; stabilises
   short input.
4. Cache/memory manager: cross-profile LRU + session/GPU release + device-tier select; dedupe ORT wasm.
5. Progressive shard loading + surface TTFA.
6. WASM-safe embeddings/backbone if a CPU path is required (currently WebGPU-only).
7. Eval: UTMOS + round-trip WER + RTF/TTFA/peak-mem + in-app Pareto dashboard. Report/demo.

Fixtures `tools/golden/`, models `tools/models/` (git-ignored).

**Checkpoints where the agent must message the user:** after parity proof, after first real audio,
after greedy-vs-CFG and int4-vs-fp16 A/B, after quantization variants, and before real-GPU/mobile
testing.

## Licensing

Site code MIT. OmniVoice weights **CC-BY-NC** (non-commercial). Never bundle weights; always fetch
from HF. Voice cloning requires consent — keep the notice in the UI.
