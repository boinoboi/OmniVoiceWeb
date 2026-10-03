# OmniVoice Web — report / paper outline

Draft outline for the capstone report. Numbers live in `docs/RESULTS.md`; this is the narrative
skeleton. Bracketed items are to be filled by the human author.

## Title
On-device multilingual code-switched TTS in the browser: a quantized, bidirectional OmniVoice
runtime on ONNX Runtime Web.

## Abstract
We implement and evaluate a fully client-side text-to-speech system for multilingual,
code-switched text, running OmniVoice (a Qwen3-0.6B masked-diffusion LM + Higgs audio codec) in the
browser via ONNX Runtime Web on WebGPU. We show that the public ONNX export cannot condition on
text because it uses autoregressive (causal) attention, whereas OmniVoice is a bidirectional
masked-diffusion LM; we re-export the backbone with a non-causal mask, port the full
classifier-free-guidance unmasking loop to TypeScript, and quantize the backbone to int4
(MatMulNBits). The resulting 280 MB browser model reproduces the PyTorch reference with **WER 0** on
a round-trip Whisper evaluation, at 6.3× smaller than fp32. [Add code-switched results.]

## 1. Introduction / motivation
- On-device TTS: privacy, latency, cost, offline.
- Code-switching (Hinglish, Marathi-English) is common and poorly served by monolingual TTS.
- Contribution: first browser-native OmniVoice runtime; a necessary ONNX re-export; a task-level
  evaluation of the precision ladder in-browser.

## 2. Background
- OmniVoice architecture: Qwen3-0.6B backbone, 8-codebook codec (Higgs Audio V2), 32-step
  masked-diffusion unmask loop, 24 kHz.
- ONNX Runtime Web execution providers: WebGPU vs WASM.
- Masked-diffusion decoding and classifier-free guidance (CFG).

## 3. System
- Runtime port: tokenizer, prompt builder (special tokens + ref text), duration estimator,
  backbone step, CFG loop (t-shift schedule, layer penalty, Gumbel positions), Higgs decoder.
- Model export toolkit: causal→bidirectional re-export (`tools/export_llm.py`), native fp16/fp32
  export, int4 MatMulNBits (`tools/quantize_llm_legacy.py`).
- Browser engineering: Cache Storage downloader with retries + HTTP Range resume, external-data
  mounting, sequential session creation, idle GPU release, in-browser metrics, streaming chunks.

## 4. The bidirectional-attention finding (core result)
- Symptom: correct algorithm with the public ONNX export produced non-speech ("Please, please…").
- Diagnosis: genai `GroupQueryAttention` + 2-D causal mask; OmniVoice needs a full bidirectional
  4-D mask.
- Fix and outcome: exact ASR parity with the PyTorch reference.

## 5. Evaluation
- Protocol: round-trip Whisper (`base`, int8) WER; spectral speech-band energy; RTF/TTFA; peak RAM.
- Precision ladder (Python + browser): fp32 / fp16 / int4 — all **WER 0** on the reference sentence.
- int4 is the browser default (280 MB, WebGPU, no `shader-f16`).
- Limitations observed: short no-reference input collapses; cloning from a synthetic reference
  leaks reference words.

## 6. Discussion / limitations
- fp16 needs `shader-f16` (absent on the test Linux/NVIDIA Dawn adapter).
- WASM path cannot run int4/fp16 (missing kernels) — WebGPU-only currently.
- Cloning needs a clean human reference and reference-exact encode alignment.
- SwiftShader WebGPU is too slow to iterate; headful real-GPU testing required.

## 7. Related work / prior art
- VocoLoco, Vernacula, transformers.js; see `docs/PRIOR_ART.md`.

## 8. Reproducibility
- Commands in `AGENTS.md` and `README.md`; fixtures in `tools/golden/` (git-ignored).
- `tools/evaluate.py --manifest eval/manifest.json`.

## 9. Conclusion & future work
- Ship int4 browser default; add WASM-safe embeddings; progressive sharding; cloning polish;
  UTMOS + larger code-switched test sets.

## Figures / tables (to produce)
1. Pipeline diagram (sub-models + loop).
2. Precision-ladder WER/size bar chart (from `eval/results.json`).
3. Waveforms/spectrograms: causal vs bidirectional vs reference.
4. RTF/TTFA vs device table.
