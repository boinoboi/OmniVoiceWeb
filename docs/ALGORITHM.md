# OmniVoice inference algorithm (reference port spec)

Source: `k2-fsa/OmniVoice` `omnivoice/models/omnivoice.py`
(`OmniVoiceGenerationConfig`, `_prepare_inference_inputs`, `_generate_iterative`,
`_predict_tokens_with_scoring`, helpers). The onnx-community `inference.py` is a
**simplified/wrong** path (raw text + greedy, no CFG, no special tokens) and produces
out-of-distribution audio. This is the spec we must implement.

## Special tokens (tokenizer.json, single ids)

`<|denoise|>`=151669, `<|lang_start|>`=151670, `<|lang_end|>`=151671,
`<|instruct_start|>`=151672, `<|instruct_end|>`=151673, `<|text_start|>`=151674,
`<|text_end|>`=151675.

## Generation config (defaults)

```
num_step            = 32
guidance_scale      = 2.0      # CFG
t_shift             = 0.1      # schedule shaping
layer_penalty_factor= 5.0      # codebooks lower index unmask first
position_temperature= 5.0      # gumbel on confidence scores
class_temperature   = 0.0      # 0 => greedy argmax
audio_mask_id       = 1024
num_audio_codebook  = 8
```

## 1. Prompt construction (`_prepare_inference_inputs`)

```
style_text = (<|denoise|> if denoise and ref_audio else "")
           + "<|lang_start|>" + (lang or "None") + "<|lang_end|>"
           + "<|instruct_start|>" + (instruct or "None") + "<|instruct_end|>"

full_text  = _combine_text(ref_text, text)      # ref_text.strip()+" "+text, whitespace/chinese rules
text_text  = "<|text_start|>" + full_text + "<|text_end|>"

input_ids  = [ style_ids | text_ids | ref_audio_tokens? | MASK×target_len ]   # repeated over 8 codebooks
audio_mask = True on ref_audio_tokens and target positions, False on style/text
```

`_combine_text`: if ref_text, `ref_text.strip()+" "+text.strip()`; strip newlines; CJK parens→ASCII;
collapse spaces; remove spaces adjacent to CJK.

## 2. CFG batch

For each item, build **conditional** (full prompt above) and **unconditional**
(`input_ids[..., -u_len:]`, i.e. only the target MASK region; `audio_mask` likewise).
Reference batches them as `2B` with a 4-D block mask that is **full (bidirectional)**
within each sequence. Our ONNX `llm_decoder` takes a 2-D `attention_mask`; for a single
item that is all-ones. Simplest faithful port: run two forwards (cond, uncond).

## 3. Per-step decoding (`_generate_iterative`)

Schedule via `_get_time_steps(t_start=0,t_end=1,num_step,t_shift)`:
`t = t_shift*linspace(0,1,num_step+1) / (1+(t_shift-1)*linspace)`;
per step `k = ceil(total_mask*(t[step+1]-t[step]))`, last step takes the remainder,
`total_mask = target_len * 8`.

Each step:
1. `logits = model(cond)` and `logits_u = model(uncond)`, cast to float32.
2. Take the target region `[C, T, V]` from each.
3. `_predict_tokens_with_scoring`:
   ```
   c = log_softmax(c_logits); u = log_softmax(u_logits)
   lp = log_softmax(c + guidance_scale*(c-u))
   lp[..., audio_mask_id] = -inf
   pred = argmax(lp)                 # class_temperature = 0
   conf = lp.max(-1)
   ```
4. `scores = conf - layer_ids * layer_penalty_factor` where `layer_ids = arange(8)[:,None]`.
5. Position sampling: `scores += gumbel_noise` scaled by `position_temperature`
   (`_gumbel_sample` = `logits/temperature + gumbel`).
6. Mask already-filled positions (`sample_tokens != MASK`): `scores[...] = -inf`.
7. `topk(k)` over flattened `[C*T]` → write `pred` into those positions of `input_ids`
   for both cond and uncond target regions.

## 4. Duration / target length (`_estimate_target_tokens`)

Rule estimator (`utils/duration.py`): per-char weights by script; reference
`ref_text` weight vs `ref_audio_tokens` length gives a speed ratio. No-ref fallback uses
ref_text="Nice to meet you.", 25 frames. Port the ratio formula; exact length is not
quality-critical (`speed` divides it).

## 5. Voice cloning (`create_voice_clone_prompt`)

Encode reference audio with the Higgs encoders → `ref_audio_tokens [C, T_ref]`, and pass
`ref_text` (its transcript) into `_combine_text`. Precompute offline into `voices.json`
(prior-art win: drop the 654 MB encoder at runtime). Auto-voice without a reference is
out-of-distribution for short input and can emit noise — always-clone is the stable path.

## Port checklist

- [ ] `engine/prompt.ts`: special-token prompt builder + `_combine_text`
- [ ] `engine/algorithm.ts`: add CFG loop + schedule + layer penalty + gumbel position sampling
- [ ] `engine/backbone.ts`: allow batch of cond+uncond (or run twice)
- [ ] `engine/duration.ts`: target-token estimator
- [ ] golden test: Python reference (`omnivoice` package) codes vs TS for the same prompt
- [ ] re-run e2e and check speech-band/UTMOS
