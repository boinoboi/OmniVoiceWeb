# PLAN — autonomous execution

This is the live plan. Update checkboxes as work lands. The todo list mirrors the phases.

## Locked decisions

1. **Algorithm:** implement **greedy unmask first** (parity + first audio), then add **CFG
   diffusion** (quality). Ship whichever wins the A/B on the 3090.
2. **Precision:** build the full ladder; use the onnx-community `int4` + an **fp16** backbone
   (`cuda/llm_decoder` + root fp16 embeddings/heads) to unblock. Our own quant variants come later.
   Default device tier is chosen by measured quality, not by size alone.
3. **Runtime:** our own port from the reference, informed by reading `vocoloco_tts/workers/tts-worker.js`
   and `vernacula/web-demo/src/inference/omnivoice.ts`.
4. **Cloning:** support both offline precomputed `voices.json` (fast, small) and in-browser encode.

## Loop protocol

- Work through phases in order; after each phase run its **gate**.
- At each gate: commit, comment on the todo, and **message the user** where marked.
- Keep `AGENTS.md`, this file, and the todo list current so a context compaction is lossless.
- Prefer adding an automated test over a manual check. Never mark a task complete without its gate.

---

## Phase 1 — Runtime foundation + parity harness (M2/M7)

- [ ] Create `tools/` Python venv (3.11/3.12 via `uv`, else system python + pip).
- [ ] `tools/requirements.txt`: torch/onnxruntime-gpu/transformers/omnivoice/soundfile/scipy.
- [ ] `tools/golden.py`: load OmniVoice ONNX (or PyTorch), emit golden vectors for each sub-model
      (embeddings, one backbone step, heads, higgs decode) for fixed inputs. Save to `tools/golden/`.
- [ ] `src/engine/ort.ts` + `probe.ts` (done) — extend to feed golden inputs and diff against TS.
- [ ] `scripts/parity.mjs`: run heads/embeddings/backbone in ORT-web (SwiftShader) and compare to
      golden vectors; print `max|Δ|` per tensor.
- **Gate:** parity within tolerance for heads + embeddings + backbone. **MESSAGE USER.**
- **Compaction note:** sizes/URLs in `manifest.ts`; I/O in `AGENTS.md`; recipe in `AGENTS.md`.

## Phase 2 — Higgs codec (M3)

- [ ] `engine/codec.ts`: `higgsEncode(wav24, wav16)` and `higgsDecode(codes)`; return typed arrays.
- [ ] `engine/audio.ts`: WAV decode (WebAudio `decodeAudioData`), mono-mix, resample to 24k/16k
      (offline `OfflineAudioContext`), WAV encode (16-bit PCM).
- [ ] Round-trip test: wav → codes → wav; compare to Python `higgs_inference.py` output.
- **Gate:** codec round-trip matches golden within tolerance.

## Phase 3 — Diffusion loop (M4)

- [ ] `engine/algorithm.ts`: greedy confidence unmask (port of `inference.py`), then CFG variant
      (port from prior art: shifted-timestep schedule, guidance log-prob mix, layer penalty, top-k).
- [ ] `engine/tokenizer.ts`: Qwen2 tokenizer. Use `@huggingface/transformers` `AutoTokenizer`, or a
      minimal BPE if the bundle hurts. Build the exact prompt the model expects.
- [ ] Unit tests for unmask scheduling + a golden test for one full generation (codes match Python).
- **Gate:** TS-generated codes match Python reference for a fixed sentence. **MESSAGE USER.**

## Phase 3b — REAL algorithm (CFG + special-token prompt)  [REQUIRED for quality]

The greedy path produces out-of-distribution audio. Port the true OmniVoice loop from the
reference (see `docs/ALGORITHM.md`): special-token prompt, classifier-free guidance
(cond+uncond), timestep schedule (`t_shift=0.1`), layer penalty (5.0), gumbel position
sampling (T=5.0), greedy class (T=0).

- [ ] `engine/prompt.ts`, `engine/duration.ts`; extend `algorithm.ts` + `backbone.ts`.
- [ ] Reference goldens from the `omnivoice` Python package (install in `tools/.venv`).
- **Gate:** browser audio has speech-like spectrum + intelligible; MESSAGE USER (clip).

## Phase 4 — End-to-end auto-voice (M5)  [pipeline done; quality via Phase 3b]

- [ ] `engine/pipeline.ts`: text → codes → higgs → WAV; progress events per stage.
- [ ] `worker/inference.worker.ts`: host the engine; post progress + audio.
- [ ] `ui/Synthesizer.tsx`: wire the real Generate button; play + download WAV; show stage progress.
- **Gate:** hear real audio in the browser (SwiftShader). Compare vs Python reference audio
  (spectrogram + MCD). **MESSAGE USER (send clip).**

## Phase 5 — Voice cloning (M6)

- [ ] `tools/precompute_voices.py`: encode reference clips → `voices.json`; ship in `public/`.
- [ ] In-browser encode path (acoustic + semantic + quantizer) behind a `full` profile.
- [ ] `ui/Synthesizer.tsx`: reference upload, transcript, consent checkbox.
- **Gate:** cloned voice matches reference timbre; cloned vs auto-voice A/B. **MESSAGE USER.**

## Phase 6 — Cache + memory manager (P5)

- [ ] LRU eviction over cached shards; storage accounting UI; delete/clear.
- [ ] `engine/memory.ts`: session release + idle timeout; best-effort GPU buffer release.
- [ ] Device-tier auto-selection (uses `capabilities.ts`): pick `lite`/`full`/`fp16` by RAM, GPU
      buffer limits and measured quality table.
- **Gate:** switching profiles releases memory; measured peak memory reported.

## Phase 7 — Streaming + progressive shards (P4g)

- [ ] Sentence chunking; generate + play sentence 1 while later sentences generate.
- [ ] Progressive external-data fetch with per-shard progress and resume.
- **Gate:** first audio starts before the whole model is resident.

## Phase 3c — Re-export the LLM with bidirectional attention  [BLOCKING for quality]

The onnx-community `llm_decoder` is causal (genai `GroupQueryAttention` + 2-D causal mask).
OmniVoice is a masked-diffusion LM requiring a **full bidirectional** mask. Without this the
model ignores the prompt (proven: reference exact ASR vs our generic output).

- [ ] `tools/export_llm.py`: load `OmniVoice` (torch, `tools/.venv-ref`), export the Qwen3 core
      with `attn_implementation="eager"`, `is_causal=False` (or the model's flex-attention mask),
      `exclude_embeds=True`/`exclude_lm_head=True`, inputs `inputs_embeds`(+`attention_mask`) →
      `hidden_states`, opset 20, fp32.
- [ ] Numerical parity vs the reference hidden states.
- [ ] Swap into `manifest.ts`; re-run the Python CFG reference (`tools/generate_cfg.py`) → ASR should
      match the text.
- **Gate:** reference-quality ASR on auto-voice; then MESSAGE USER (clip).

## Phase 8 — Quantization toolkit (P2/P3)

- [ ] `tools/export.py`: ONNX export with dynamic axes + external-data sharding; parity check.
- [ ] `tools/quantize.py`: FP16, dynamic INT8, static INT8 (calibrated), INT4 weight-only, mixed
      (e.g. int4 backbone + fp16 diffusion-sensitive parts).
- [ ] Calibration corpus: code-switched (Hinglish + Marathi-English) + monolingual sentences.
- [ ] Publish variants to a dedicated HF repo; add to `manifest.ts` as selectable profiles.
- **Gate:** size + parity + latency table for every variant. **MESSAGE USER.**

## Phase 9 — Evaluation (P6)

- [ ] Test sets: Hinglish + Marathi-English (with transcripts) under `eval/`.
- [ ] `tools/evaluate.py`: UTMOS, Whisper round-trip WER, RTF, TTFA, peak memory per variant/device.
- [ ] `ui/Dashboard.tsx`: in-app results table + charts (size/quality/speed Pareto).
- **Gate:** results tables + charts committed; Pareto variant chosen. **MESSAGE USER** before
      real-GPU/mobile matrix.

## Phase 10 — Report / paper / demo (P7)

- [ ] README polish, `docs/RESULTS.md`, reproducibility instructions.
- [ ] Demo video script + shot list. (Human records.)
- [ ] Paper-draft outline from the results.
- **Gate:** final report, presentation, working demo.

---

## Checkpoints that require the user

1. After Phase 1 parity. 2. After Phase 4 first audio. 3. Greedy-vs-CFG + int4-vs-fp16 A/B.
4. After Phase 8 variants. 5. Before real-GPU/mobile testing. Also: report/paper content decisions.
