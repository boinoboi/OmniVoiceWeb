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

## Export defects found

- `audio_tokenizer/fp16/semantic_encoder.onnx` is **malformed**: a `LayerNormalization` node is
  bound to both float and float16, so ORT rejects it on *any* execution provider
  (`Type Error: Type parameter (T) bound to different types`). The other three fp16 codec models
  load fine. A correct fp16 semantic encoder must be re-exported in Phase 8.
