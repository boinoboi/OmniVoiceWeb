# Architecture

## Layers

```
src/
├─ engine/                 framework-free, worker-safe
│  ├─ manifest.ts          model registry: URLs, exact byte sizes, lite/full profiles
│  ├─ cache.ts             Cache Storage downloader (progress, cancel, verify), profiles, LRU (P5)
│  ├─ capabilities.ts      WebGPU adapter, device info, feasibility assessment
│  ├─ ort.ts               ORT-web factory: wasmPaths, external-data array, session create/release
│  ├─ probe.ts             single-model forward probe (runtime validation)
│  ├─ tokenizer.ts         Qwen2 tokenizer + OmniVoice prompt assembly          [M3]
│  ├─ codec.ts             Higgs encode/decode (24k/16k, resample)              [M3]
│  ├─ algorithm.ts         greedy + CFG diffusion loop, codebook weights         [M4]
│  ├─ pipeline.ts          orchestration + progress events                       [M5]
│  ├─ audio.ts             WAV encode/decode, playback                          [M5]
│  └─ memory.ts            session/GPU-buffer lifecycle, device-tier selection   [P5]
├─ worker/inference.worker.ts   runs the engine off the main thread             [M5]
├─ ui/                     React components (SystemCheck, ModelManager, EngineLab, Synthesizer, Dashboard)
│  └─ useModelManager/useCapabilities hooks
└─ App.tsx
```

## Inference data flow

Auto-voice:
```
text ──tokenizer──▶ text_ids
[MASK=1024] × N (per 8 codebooks)
  loop 32×:
    audio_embeddings_encoder(input_ids[B,8,S], audio_mask[B,S]) ─▶ inputs_embeds[B,S,1024]
    llm_decoder(inputs_embeds, attention_mask, position_ids, empty KV) ─▶ hidden[B,S,1024]
    audio_heads_decoder(hidden) ─▶ logits[B,8,S,1025]
    unmask top-confidence positions (weights [8,8,6,6,4,4,2,2])
  ─▶ audio_codes[8,T]
higgs_decoder(codes[8,1,T]) ─▶ waveform 24 kHz ─▶ WAV
```

Voice cloning: prepend `ref_codes[8,Tref]` (encoded from a reference WAV, or loaded from
`voices.json`) to the audio region; `audio_mask` is True only on the generated region.

## Model profiles (exact sizes)

| Profile | Backbone | Codec | Download |
|---|---|---|---|
| `lite` (auto-voice) | int4 | fp16 decoder only | ~423 MB |
| `full` (cloning) | int4 | + 3 fp16 encoders | ~735 MB |
| `fp16` (quality) | fp16 from `cuda/`+root | fp16 | ~1.2 GB backbone |

Exact file list + sizes live in `src/engine/manifest.ts` (source of truth).

## Threading / lifecycle

- One Web Worker owns all ORT sessions. The main thread only sends commands and receives progress.
- ORT sessions are created lazily per stage and released on idle (`memory.ts`), freeing WASM heap
  and (best-effort) WebGPU buffers.
- `ort.env.wasm.numThreads = 1` (no COOP/COEP on GitHub Pages). WebGPU path unaffected.
- Models are cached in Cache Storage under key `<origin>/__omnivoice_models__/<path>`.

## External data

- Sidecar file = onnx basename + `.data`.
- ORT-web `sessionOptions.externalData` must be an **array** when sharding (>2 GB total).
- `progressive shard loading` (P4g) fetches sidecars in order and reports progress per shard.

## Validation strategy

- **Golden vectors**: run the Python reference (`tools/`) on fixed inputs, store outputs as `.json`
  f16/base64 or `.npy`; the TS runtime must match within tolerance (per-stage `max|Δ|`).
- **Op coverage**: run each sub-model on SwiftShader WebGPU in headless Chrome.
- **Browser smoke**: `scripts/smoke.mjs` for UI/render on desktop + iPhone/iPad viewports.
- **Quality**: UTMOS + Whisper round-trip WER on Hinglish/Marathi-English sets (P6).
