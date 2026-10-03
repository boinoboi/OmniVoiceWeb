"""Faithful OmniVoice CFG generation using the ONNX backbone (reference for the TS port)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
GOLDEN = ROOT / "golden"

C = 8
MASK_ID = 1024
VOCAB = 1025


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return (x - m) - np.log(e.sum(axis=-1, keepdims=True))


def combine_text(text: str, ref_text: str | None) -> str:
    import re

    full = f"{ref_text.strip()} {text.strip()}" if ref_text else text.strip()
    full = re.sub(r"[\r\n]+", "", full)
    full = full.replace("\uff08", "(").replace("\uff09", ")")
    full = re.sub(r"[ \t]+", " ", full)
    cjk = r"[\u4e00-\u9fff]"
    full = re.sub(rf"(?<={cjk})\s+|\s+(?={cjk})", "", full)
    return full


def time_steps(num_step: int, t_shift: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_step + 1)
    return t_shift * t / (1 + (t_shift - 1) * t)


def schedule(total_mask: int, num_step: int, t_shift: float) -> list[int]:
    ts = time_steps(num_step, t_shift)
    rem = total_mask
    out = []
    for step in range(num_step):
        num = rem if step == num_step - 1 else min(math.ceil(total_mask * (ts[step + 1] - ts[step])), rem)
        out.append(int(num))
        rem -= int(num)
    return out


def build_prompt(tok, text, lang=None, instruct=None, ref_text=None, ref_codes=None, target=80):
    style = []
    if ref_codes is not None:
        style += tok.encode("<|denoise|>", add_special_tokens=False)
    style += tok.encode("<|lang_start|>", add_special_tokens=False)
    style += tok.encode(lang or "None", add_special_tokens=False)
    style += tok.encode("<|lang_end|>", add_special_tokens=False)
    style += tok.encode("<|instruct_start|>", add_special_tokens=False)
    style += tok.encode(instruct or "None", add_special_tokens=False)
    style += tok.encode("<|instruct_end|>", add_special_tokens=False)

    full = combine_text(text, ref_text)
    text_ids = tok.encode("<|text_start|>", add_special_tokens=False)
    text_ids += tok.encode(full, add_special_tokens=False)
    text_ids += tok.encode("<|text_end|>", add_special_tokens=False)

    prefix = style + text_ids
    cond = np.tile(np.array(prefix, dtype=np.int64), (C, 1))
    if ref_codes is not None:
        cond = np.concatenate([cond, ref_codes.astype(np.int64)], axis=1)
    target_ids = np.full((C, target), MASK_ID, dtype=np.int64)
    cond = np.concatenate([cond, target_ids], axis=1)
    gen_start = cond.shape[1] - target
    cond_mask = np.zeros(cond.shape[1], dtype=bool)
    cond_mask[gen_start:] = True

    uncond = target_ids.copy()
    uncond_mask = np.ones(target, dtype=bool)
    return cond[None], cond_mask[None], uncond[None], uncond_mask[None], gen_start


def load_sessions(provider: str = "CPUExecutionProvider", llm_path: str = "int4/llm_decoder.onnx"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def s(rel):
        return ort.InferenceSession(str(MODELS / rel), sess_options=opts, providers=[provider])

    return s("int4/audio_embeddings_encoder.onnx"), s(llm_path), s("int4/audio_heads_decoder.onnx")


NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64, "tensor(bool)": bool}


def cast_feed(sess, arrays):
    want = {i.name: i.type for i in sess.get_inputs()}
    return {k: (v.astype(NP[want[k]]) if k in want else v) for k, v in arrays.items()}


def model_forward(emb, llm, heads, input_ids, audio_mask):
    inputs_embeds = emb.run(["inputs_embeds"], cast_feed(emb, {"input_ids": input_ids, "audio_mask": audio_mask}))[0]
    b, s, _ = inputs_embeds.shape
    feed = {"inputs_embeds": inputs_embeds}
    for inp in llm.get_inputs():
        if inp.name == "attention_mask":
            feed[inp.name] = np.ones((b, s), dtype=np.int64)
        elif "past" in inp.name:
            feed[inp.name] = np.zeros((b, 8, 0, 128), dtype=np.float16)
    hidden = llm.run(["hidden_states"], cast_feed(llm, feed))[0]
    return heads.run(["logits"], cast_feed(heads, {"hidden_states": hidden}))[0].astype(np.float32)


def gumbel(rng, shape):
    u = rng.random(shape).astype(np.float32)
    return -np.log(-np.log(u + 1e-10) + 1e-10)


def generate(emb, llm, heads, cond, cond_mask, uncond, uncond_mask, gen_start, target, cfg):
    scale = cfg["scale"]
    penalty = cfg["layer_penalty"]
    pos_temp = cfg["position_temperature"]
    num_step = cfg["num_step"]
    sched = schedule(target * C, num_step, cfg["t_shift"])
    rng = np.random.default_rng(cfg["seed"])
    layer_ids = np.arange(C, dtype=np.float32)[:, None]
    target_ids = np.full((C, target), MASK_ID, dtype=np.int64)

    for step in range(num_step):
        k = sched[step]
        if k <= 0:
            continue
        c_logits = model_forward(emb, llm, heads, cond, cond_mask)[0, :, gen_start:, :]
        u_logits = model_forward(emb, llm, heads, uncond, uncond_mask)[0]
        clp = log_softmax(c_logits)
        ulp = log_softmax(u_logits)
        lp = log_softmax(clp + scale * (clp - ulp))
        lp[..., MASK_ID] = -np.inf
        pred = lp.argmax(-1)
        conf = lp.max(-1)
        scores = conf - layer_ids * penalty
        if pos_temp > 0:
            scores = scores / pos_temp + gumbel(rng, scores.shape)
        scores[target_ids != MASK_ID] = -np.inf
        flat = scores.reshape(-1)
        idx = np.argpartition(-flat, k - 1)[:k]
        tflat = target_ids.reshape(-1).copy()
        pflat = pred.reshape(-1)
        tflat[idx] = pflat[idx]
        target_ids = tflat.reshape(C, target)
        cond[0, :, gen_start:] = target_ids
        uncond[0, :] = target_ids
        if step % 8 == 0 or step == num_step - 1:
            print(f"  step {step+1}/{num_step} k={k} masked={(target_ids==MASK_ID).any(axis=0).sum()}")
    return target_ids


def metrics(w):
    n = min((len(w) // 480) * 480, 480 * 60)
    n = n if n >= 8192 else min(len(w), 8192)
    X = np.abs(np.fft.rfft(w[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / 24000)
    p = X**2
    p /= p.sum() + 1e-12
    band = (f >= 300) & (f <= 3400)
    return {
        "peak": float(np.abs(w).max()),
        "rms": float(np.sqrt((w**2).mean())),
        "centroid": float((f * p).sum()),
        "speech_band_pct": float(p[band].sum() * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello, this is a test of the system.")
    ap.add_argument("--target", type=int, default=80)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--layer-penalty", type=float, default=5.0)
    ap.add_argument("--pos-temp", type=float, default=5.0)
    ap.add_argument("--t-shift", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-wav", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--llm", default="int4/llm_decoder.onnx")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS / "int4")
    emb, llm, heads = load_sessions(args.provider, args.llm)

    ref_codes = None
    if args.ref_wav:
        import sys

        sys.path.insert(0, str(ROOT))
        from golden import GOLDEN as _g  # noqa: F401
        import soundfile as sf
        import onnxruntime as ort

        def hsess(name):
            return ort.InferenceSession(str(MODELS / "audio_tokenizer" / name), providers=["CUDAExecutionProvider"])

        from scipy.signal import resample_poly
        from math import gcd

        wav24, sr = sf.read(args.ref_wav, dtype="float32")
        if wav24.ndim > 1:
            wav24 = wav24.mean(1)
        if sr != 24000:
            g = gcd(24000, sr)
            wav24 = resample_poly(wav24, 24000 // g, sr // g).astype(np.float32)
        wav16 = resample_poly(wav24, 16000 // 24000 if False else 2, 3).astype(np.float32)
        ac = hsess("acoustic_encoder.onnx")
        se = hsess("semantic_encoder.onnx")
        qe = hsess("quantizer_encoder.onnx")
        a = ac.run(["acoustic_features"], {"waveform_24k": wav24[None, None, :]})[0]
        s = se.run(["semantic_features"], {"waveform_16k": wav16[None, :]})[0]
        t = min(a.shape[2], s.shape[2])
        ref_codes = qe.run(["codes"], {"acoustic_features": a[:, :, :t], "semantic_features": s[:, :, :t]})[0][:, 0, :]
        print("ref codes", ref_codes.shape)

    cond, cond_mask, uncond, uncond_mask, gen_start = build_prompt(
        tok, args.text, ref_text=args.ref_text, ref_codes=ref_codes, target=args.target
    )
    print(f"prompt: cond len {cond.shape[2]}, gen_start {gen_start}, target {args.target}")

    codes = generate(
        emb, llm, heads, cond, cond_mask, uncond, uncond_mask, gen_start, args.target,
        {
            "scale": args.scale,
            "layer_penalty": args.layer_penalty,
            "position_temperature": args.pos_temp,
            "num_step": args.steps,
            "t_shift": args.t_shift,
            "seed": args.seed,
        },
    )

    hd = __import__("onnxruntime").InferenceSession(
        str(MODELS / "audio_tokenizer" / "higgs_decoder.onnx"), providers=["CUDAExecutionProvider"]
    )
    import soundfile as sf

    w = hd.run(["waveform_24k"], {"codes": codes[:, None, :].astype(np.int64)})[0].squeeze()
    out = args.out or str(GOLDEN / "cfg_out.wav")
    sf.write(out, w, 24000)
    m = metrics(w)
    print(f"dur={len(w)/24000:.2f}s peak={m['peak']:.3f} rms={m['rms']:.3f} centroid={m['centroid']:.0f}Hz speech_band={m['speech_band_pct']:.1f}%  -> {out}")


if __name__ == "__main__":
    main()
