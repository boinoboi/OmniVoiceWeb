# Results log

Numbers in this file are reproduced by scripts in `tools/` and `scripts/`. Keep it append-only.

## Parity: ORT-web WASM vs ORT Python CPU EP (int4 export)

Command: `node scripts/parity.mjs` (fixtures from `tools/.venv/bin/python tools/golden.py`).
Same random inputs, same ONNX graphs; difference is only the execution provider
(WASM `MatMulNBits` vs CPU EP).

| Sub-model | shape | max abs Δ | relative Δ | cosine | argmax agree | verdict |
|---|---|---|---|---|---|---|
| `audio_embeddings_encoder` | `[1,20,1024]` | 0.0 | 0.0 | 1.00000 | — | exact |
| `audio_heads_decoder` | `[1,8,16,1025]` | 7.25e-3 | 5.8e-3 | 0.99998 | **99.22%** | pass |
| `llm_decoder` (28× int4) | `[1,20,1024]` | 1.04 | 6.4e-3 | 0.99999 | — | pass |

**Finding (supports the project thesis):** int4 weights executed on different execution
providers diverge by ~0.6–0.8% relative, and the single-layer `audio_heads_decoder` already
disagrees on **0.8% of argmax tokens**. In a 32-step loop this can compound — the diffusion
loop is the precision-sensitive stage. End-to-end token agreement is measured in Phase 3.

Tolerance policy: pass if `relMax ≤ 1e-2` **or** cosine ≥ 0.999 **or** argmax ≥ 99.9%.
Exact bitwise equality is neither expected nor required across EPs.

## Codec parity: Higgs Audio V2 (fp32 reference)

Same fixtures, ORT-web WASM vs ORT Python CUDA EP. 1 s synthetic waveform (24 kHz),
25 codec frames.

| Sub-model | shape | max abs Δ | relative Δ | cosine | verdict |
|---|---|---|---|---|---|
| `acoustic_encoder` | `[1,256,25]` | 2.38e-3 | 4.0e-4 | 1.00000 | pass |
| `semantic_encoder` | `[1,768,25]` | 1.96e-2 | 1.4e-3 | 1.00000 | pass |
| `quantizer_encoder` (`codes` int64) | `[8,1,25]` | — | — | **100% exact** | pass |
| `higgs_decoder` | `[1,1,24000]` | 4.68e-4 | 9.0e-4 | 1.00000 | pass |

**Finding:** the discrete quantizer codes match the reference **exactly**, and the decoded
waveform is within 9e-4 relative — so the codec round-trip is faithful through the browser
runtime. Codec precision is *not* the risky stage; the diffusion loop is.

## Diffusion loop fidelity (int4, greedy, 32 steps)

Reference: `tools/generate.py` (Python, CPU EP) for a code-switched sentence
("Hello, kaise ho tum? Let's test.", 11 tokens, 80 audio frames). Compare: the TS greedy loop
through ORT-web WASM (`node scripts/generation-parity.mjs`).

Step-0 logits (single forward, all positions masked):

| metric | value |
|---|---|
| cosine of logits | **0.999996** |
| argmax token agreement | **83.59%** |

So the logits are numerically almost identical, yet **16% of greedy token choices differ** — the
model's per-codebook top-2 logits frequently sit within int4 noise. Fidelity by diffusion steps:

| steps | code agreement | ms/frame (WASM) |
|---|---|---|
| 1 | 83.6% | 30 |
| 4 | 46.6% | 118 |
| 8 | 34.1% | 235 |
| 16 | 46.9% | 469 |
| 32 | 40.5% | 943 |

**Interpretation (this is the project's core result so far):** weight-level error and even
*logit* cosine are poor proxies for TTS quality. A cosine of 0.999996 still yields ~16% token
flips, which the 32-step loop turns into ~60% disagreement. This is exactly why the proposal
insists on **task-level** metrics (UTMOS, round-trip WER) rather than weight error, and why the
precision ladder (fp16/INT8/INT4/mixed) must be measured on audio. Token flips may still decode
to acceptable audio, so listening/UTMOS is the arbiter — the loops and token-flip rate are
reported as diagnostics, not pass/fail gates.

**Algorithm correctness gate:** step-0 logit cosine ≥ 0.999 → PASS (the loop and IO are correct).

## End-to-end browser inference (Phase 4)

- The full pipeline (tokenizer → 3-model backbone → 32-step greedy loop → Higgs decoder → WAV)
  runs in the browser. First real audio produced via WebGPU (SwiftShader) for "Hello world."
  (48 frames → 1.92 s @ 24 kHz), saved at `tools/golden/browser_tts.wav`.
- **WASM fallback cannot run the int4 embeddings**: `GatherBlockQuantized` has no WASM kernel
  (`Could not find an implementation for GatherBlockQuantized`). A WASM path needs a non-int4
  (fp32/int8) embeddings graph.
- WebGPU on the software SwiftShader adapter is slow (~10 min for ~50 frames); real GPUs are
  expected to be orders of magnitude faster (prior art: ~2.5× real-time on a 3090).

## Precision A/B and the auto-voice quality finding

Greedy auto-voice, "Hello world.", 48 frames, 32 steps; spectral metrics of the decoded audio:

| backbone | centroid | speech-band (300–3400 Hz) energy | rms |
|---|---|---|---|
| int4 | 345 Hz | 12.2% | 0.20 |
| fp16 | 20 Hz | **0.1%** | 0.49 |
| Python int4 reference | 227 Hz | 7.0% | 0.55 |

**fp16 is no better than int4**, so the poor output is **algorithmic, not a quantization
artefact**: the simplified *greedy, no-CFG, no-reference* loop is out of distribution for short
auto-voice input. This matches prior art ("without a reference voice, short input can emit noise")
and the diffusion-loop precision literature. The next required work is the **real algorithm**
(classifier-free guidance + reference voice), not a smaller quantization step.

## Ground truth vs ONNX port (the pivotal finding)

Installed the real reference (`omnivoice` package + torch cu128) in `tools/.venv-ref` and ran
the exact CLI (`omnivoice-infer --text "The quick brown fox..."`) on the 3090:

| audio | ASR round-trip | centroid | speech-band | dur |
|---|---|---|---|---|
| **Reference (PyTorch)** | **"The quick brown fox jumps over the lazy dog." (exact)** | 1364 Hz | **95.5%** | 2.64 s |
| Our ONNX CFG port | "Please, please, please, please, please." | 144 Hz | 16.3% | 3.60 s |

So auto-voice works in the reference; our ONNX port does **not condition on the text**.

**Root cause: the onnx-community export uses causal attention.** The exported `llm_decoder`
uses ORT `GroupQueryAttention` with genai's `attn_mask_subformat` reformat (2-D causal mask).
The real OmniVoice loop uses a **full bidirectional block mask** (`batch_attention_mask[...]=True`
over the whole `[c_len, c_len]` block) — the model is a masked-diffusion LM, not a causal LM.
Under causal attention the target positions cannot use the bidirectional context the model was
trained on, so conditioning collapses to generic speech.

**Consequence:** the shipped ONNX backbone cannot be made faithful by prompt/CFG fixes alone.
We must **re-export the LLM with non-causal (bidirectional) attention** — which is exactly the
proposal's ONNX-export deliverable (Phase 2/8), now clearly *necessary*, not optional. The
`omnivoice` + torch env we just installed is the source for that export.

## BREAKTHROUGH: bidirectional LLM export + CFG reproduces the reference

Fix = re-export the Qwen3 core with a 4-D all-zero additive mask (full **non-causal**
attention), then run the real prompt + CFG. `tools/export_llm.py` produces
`tools/models/bidir/llm_decoder.onnx` (1.77 GB fp32, zero `GroupQueryAttention` nodes).
`tools/generate_cfg.py --llm bidir/llm_decoder.onnx` runs it.

| pipeline | ASR round-trip | centroid | speech-band | dur |
|---|---|---|---|---|
| reference (PyTorch) | exact | 1364 Hz | 95.5% | 2.64 s |
| causal ONNX (old) | "Please, please, please…" | 144 Hz | 16.3% | 3.60 s |
| **bidir ONNX + CFG** | **exact** | 878 Hz | **88.1%** | **2.64 s** |

Code-switching (headline): "Hola, kaise ho tum? Aaj hum ek test kar rahe hain." →
ASR `अला कैसे हो तुम आज हम एक टेस्ट कर रहे हैं` (correct Hinglish), 55.9% speech-band.
Generation is ~3.7 s on the 3090 (32 steps, 66 frames).

**The critical fix was the export, not the algorithm** — the onnx-community causal graph
cannot be recovered by prompt/CFG alone. This validates the proposal's export-toolkit
deliverable as essential. Browser path now requires: (a) TS port of prompt+CFG, (b) a
browser-sized (int4/fp16) bidirectional LLM.

## Browser-sized bidirectional LLM (fp16) — exact ASR at 885 MB

Native fp16 export (`tools/export_llm.py --dtype fp16`) makes an **885 MB** external-data graph
(`llm_decoder_fp16.onnx` + `.data`); the earlier "2652 MB" was a size-print bug that summed the
fp32 `.data` too. `tools/generate_cfg.py --llm bidir/llm_decoder_fp16.onnx` on the 3090:

| pipeline | ASR round-trip | centroid | speech-band | dur |
|---|---|---|---|---|
| fp32 bidir + CFG | exact | 878 Hz | 88.1% | 2.64 s |
| **fp16 bidir + CFG** | **exact** | 773 Hz | **81.2%** | 2.64 s |

So fp16 is faithful enough for the backbone at **half the size** — this is the browser default
(`BIDIR_BASE` in `manifest.ts`, hosted at HF `PranavBoi/omnivoice-web`). fp16 is a
**WebGPU-only** rung (ORT WASM has no fp16 kernels).

## Feature work landed (Phases 5–7, engine + UI)

- **CFG in the browser:** `engine/prompt.ts` + `iterativeUnmaskCfg` (`algorithm.ts`) are wired into
  `pipeline.generate`; unit-tested with a mock step (`algorithm.test.ts`) — every position unmasks,
  codes are written back to cond+uncond, progress reaches 0 remaining.
- **Streaming (P7):** `engine/streaming.ts` splits code-switched text (latin + Devanagari danda)
  into sentence chunks; `generateStream` emits progressive chunks; the worker posts `chunk`
  messages and the UI publishes a growing WAV so playback can start early.
- **Cloning (P5):** `engine/cloning.ts` ports the Higgs reference encode path
  (24k acoustic → 16k semantic → frame-aligned quantizer → `[8, t]` codes);
  `Synthesizer.loadEncoders()/encodeReference()` (full profile) + `encode` worker message + a
  reference-clip picker in the UI. Needs listening/A-B at the gate.
- **Cache manager (P6):** `pruneCache`/`filesToPrune`/`cachedBytes`; "Remove other models" button.
- **Real-GPU testing:** headless Chrome returns *no* WebGPU adapter on this box; **headful** gives
  `nvidia | ampere`. `scripts/probe-gpu.mjs` documents the matrix; `scripts/e2e-tts.mjs` gained
  `E2E_REALGPU`, `E2E_HEADFUL`, `E2E_PROFILE` (persistent cache) and fails fast without a GPU.

## Browser parity on a real GPU (fp32 bidirectional + CFG) — THE MILESTONE

Headful Chrome on the 3090 (adapter `nvidia | ampere`), models streamed from HF, running the full
pipeline **in the browser** (tokenizer → int4 embeddings → fp32 bidirectional LLM → int4 heads →
CFG loop → Higgs decoder → WAV). Command:
`E2E_REALGPU=1 E2E_HEADFUL=1 E2E_PROFILE=… node scripts/e2e-tts.mjs http://localhost:4173/`.

| backbone | text | dur | centroid | speech-band | ASR round-trip |
|---|---|---|---|---|---|
| fp32 bidir (1.77 GB) | "The quick brown fox jumps over the lazy dog." | 2.56 s | 2658 Hz | 87.0% | **exact** |
| **int4 bidir (280 MB)** | "The quick brown fox jumps over the lazy dog." | 2.56 s | 2385 Hz | 85.9% | **exact** |

The browser now reproduces the Python reference (also exact ASR). The browser default is **int4
MatMulNBits** (produced with the ORT 1.20 legacy quantizer in `tools/.venv-quant`; ORT 1.30's new
quantizer is broken). int4 keeps f32 activations, so it needs **no `shader-f16`** — this is the
sweet spot: 6.3× smaller than fp32 at equal ASR. Findings:

- **fp16 cannot run on WebGPU here.** This Chrome/Dawn adapter lacks `shader-f16`
  (`scripts/feat.mjs` → `f16:false`), and ORT-web 1.30 needs it for fp16 ops
  (`Cast requires f16 but the device does not support it`). We therefore ship the **fp32**
  bidirectional LLM (1.77 GB) as the browser default. fp16 stays a variant for f16-capable GPUs.
- **int4 is the browser default now:** `tools/quantize_llm_legacy.py` + ORT 1.20
  (`tools/.venv-quant`, `onnx==1.16.2`) produce 280 MB MatMulNBits with exact ASR. ORT 1.30's new
  quantizer errors (`must be 8-bit before packing`) and the legacy one was removed from 1.30.
- **Download robustness matters:** HF Xet emits `ERR_NETWORK_CHANGED`, which aborted multi-GB
  fetches. Fixed with retries + HTTP **Range resume**; the 1.77 GB download now completes with
  monotonic progress.
- **Short auto-voice is still unstable:** "Hello world." (35 frames, 1.44 s) decodes to non-speech
  (`. . . .`), confirming prior art — a reference voice (cloning) is the stabiliser.
- **Duration estimator fixed:** ported the reference `RuleDurationEstimator` power-curve boost
  (`low_threshold=50`, boost 3). The fox sentence now estimates 64–65 frames vs the reference 66.

## Task-level evaluation (round-trip WER + spectral metrics)

`tools/evaluate.py --manifest eval/manifest.json` (faster-whisper `base`, CPU int8). WER is a
normalized word error rate (0 = exact). Rows generated across the precision ladder and both
runtimes for the sentence "The quick brown fox jumps over the lazy dog.".

| variant | dur | centroid | speech-band | ASR round-trip | WER |
|---|---|---|---|---|---|
| PyTorch reference | 2.64 s | 2710 Hz | 90.2% | exact | **0.00** |
| Python fp32 bidir | 2.64 s | 3058 Hz | 76.5% | exact | **0.00** |
| Python fp16 bidir | 2.64 s | 3074 Hz | 68.7% | exact | **0.00** |
| Python int4 bidir | 2.64 s | 2533 Hz | 85.5% | exact | **0.00** |
| Browser fp32 bidir | 2.56 s | 2658 Hz | 87.0% | exact | **0.00** |
| Browser int4 bidir | 2.56 s | 2385 Hz | 85.9% | exact | **0.00** |
| Browser int4, "Hello world." | 1.44 s | 1689 Hz | 4.3% | — | 2.00 |
| Python int4 clone (synthetic ref) | 2.40 s | 1961 Hz | 38.0% | "This is a Clone Fox jumps over speaking now." | 0.57 |

**Reading:** every fp32/fp16/int4 variant, in Python *and* in the browser, reproduces the reference
sentence with **WER 0** — the runtime port and int4 quantization are lossless at the task level for
normal-length input. Two open items match prior art: **short auto-voice** ("Hello world.") collapses
to non-speech, and **cloning from a synthetic (model-generated) reference** leaks reference words
(needs a clean human reference and reference-exact encode alignment).

## Export defects found

- `audio_tokenizer/fp16/semantic_encoder.onnx` is **malformed**: a `LayerNormalization` node is
  bound to both float and float16, so ORT rejects it on *any* execution provider
  (`Type Error: Type parameter (T) bound to different types`). The other three fp16 codec models
  load fine. A correct fp16 semantic encoder must be re-exported in Phase 8.
