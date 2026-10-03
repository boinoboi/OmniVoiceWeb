# Prior art & differentiation

## What already exists

| Project | Kind | Notes |
|---|---|---|
| **Magkino/vocoloco_tts** | Full browser app (vanilla JS, Apache-2.0) | WebGPU + WASM, sentence streaming, voice design, guided cloning, saved voices, library, MP3 + AI-provenance. Models on `Gigsu/vocoloco-onnx` (~2.3 GB main, 83 MB decoder, 624 MB encoder). Custom WebGPU compute shader for log-softmax/CFG/argmax. Chrome/Edge 113+. |
| **christopherthompson81/vernacula** `web-demo` | React+Vite+`onnxruntime-web` research demo (Netlify) | IPA/TTS pipeline; `tools/logit-diff.mjs` parity, `pw-bench.mjs` bench. Published quantized ONNX. Documented that the diffusion loop is precision-sensitive; INT8 pending listening test. |
| **RabbitDaisuke/typed-voice** | `src/engine/omnivoice-engine.js` | ORT-web engine. |
| **AFun9/Omnivoice-onnx**, **ServeurpersoCom/omnivoice.cpp** (GGML), **FerrisMind/omnivoice-rs** | exports / native ports | Reference. |
| **onnx-community/OmniVoice-Onnx** | ONNX export (int4 + fp16) | What we consume. |

**Conclusion:** "OmniVoice in a browser" is solved. Building only that would not be novel.

## What is still open (our contribution)

1. **Quantization trade-off study for the diffusion loop**, measured end-to-end:
   fp16 / dynamic INT8 / static INT8 / INT4 weight-only / mixed, with size, latency, peak memory,
   **UTMOS** and **round-trip WER** — plus listening. Prior art has only: fp32 reference, an
   unverified INT8, and an unconsumed INT4 export.
2. **Code-switched (Hinglish + Marathi-English) evaluation.** Nobody has measured a multilingual
   code-switched TTS on-device with these metrics. This is the headline of the proposal.
3. **A cache + memory manager with measured behaviour**: storage accounting, LRU eviction, inference
   session + GPU buffer release when idle, and **automatic device-tier variant selection**.
4. **Numerical parity gates** (golden vectors) that make the whole study reproducible.

## How we beat prior art

- **Correct algorithm, then optimize.** Implement the *real* diffusion (CFG, timestep schedule,
  guidance mix, layer penalty, top-k unmask), not just a greedy simplification — then measure.
- **Smaller without losing quality.** Find the Pareto point on the precision ladder instead of
  shipping ~2.3 GB (VocoLoco) or guessing INT8 (vernacula). Target: match near-fp32 quality at a
  fraction of the size, proven by UTMOS/WER on code-switched speech.
- **Honest metrics in the product.** An in-app evaluation dashboard (size / load / TTFA / RTF /
  peak mem / UTMOS / WER), not just a demo.
- **Reproducibility.** Our own export + quantization toolkit with parity checks.
- **Code-switching first-class.** Mixed-language test sets, examples, and results.

## Free wins to adopt

1. Precompute reference voice codes offline (`voices.json`) → drop the 654 MB encoder at runtime.
2. `externalData` as an array for >2 GB sharding.
3. fp16 targets WebGPU, not WASM.
4. Read `vocoloco_tts/workers/tts-worker.js` and `vernacula/web-demo/src/inference/omnivoice.ts`
   for concrete gotchas before porting.
