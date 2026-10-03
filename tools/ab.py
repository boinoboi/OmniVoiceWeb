"""Precision A/B for the greedy OmniVoice loop: int4 vs fp16 backbone, with audio metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
GOLDEN = ROOT / "golden"

CODEBOOKS = 8
MASK_ID = 1024
CB_WEIGHTS = np.array([8, 8, 6, 6, 4, 4, 2, 2], dtype=np.float32) / 40

VARIANTS = {
    "int4": {
        "emb": "int4/audio_embeddings_encoder.onnx",
        "llm": "int4/llm_decoder.onnx",
        "heads": "int4/audio_heads_decoder.onnx",
    },
    "fp16": {
        "emb": "audio_embeddings_encoder.onnx",
        "llm": "cuda/llm_decoder.onnx",
        "heads": "audio_heads_decoder.onnx",
    },
}

FP16_FILES = [
    "audio_embeddings_encoder.onnx",
    "audio_embeddings_encoder.onnx.data",
    "audio_heads_decoder.onnx",
    "cuda/llm_decoder.onnx",
    "cuda/llm_decoder.onnx.data",
]

NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(bool)": bool,
}


def download_fp16() -> None:
    from huggingface_hub import hf_hub_download

    for name in FP16_FILES:
        hf_hub_download("onnx-community/OmniVoice-Onnx", name, local_dir=MODELS)
        print("  downloaded", name)


def load(rel: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(MODELS / rel), sess_options=opts, providers=providers)


def cast_feed(sess, arrays: dict) -> dict:
    want = {i.name: i.type for i in sess.get_inputs()}
    feed = {}
    for name, value in arrays.items():
        feed[name] = value.astype(NP[want[name]]) if name in want else value
    return feed


def run_step(emb, llm, heads, input_ids, audio_mask):
    embeds = emb.run(["inputs_embeds"], cast_feed(emb, {"input_ids": input_ids, "audio_mask": audio_mask}))[0]
    b, s, _ = embeds.shape
    arrays = {"inputs_embeds": embeds}
    for inp in llm.get_inputs():
        if inp.name == "attention_mask":
            arrays[inp.name] = np.ones((b, s), dtype=np.int64)
        elif "past" in inp.name:
            arrays[inp.name] = np.zeros((b, 8, 0, 128), dtype=np.float32)
    hidden = llm.run(["hidden_states"], cast_feed(llm, arrays))[0]
    logits = heads.run(["logits"], cast_feed(heads, {"hidden_states": hidden}))[0]
    return logits.astype(np.float32)


def generate(emb, llm, heads, text_tokens, tokens, steps):
    seq = len(text_tokens) + tokens
    input_ids = np.zeros((1, CODEBOOKS, seq), dtype=np.int64)
    input_ids[0, :, : len(text_tokens)] = text_tokens
    input_ids[0, :, len(text_tokens) :] = MASK_ID
    audio_mask = np.zeros((1, seq), dtype=bool)
    audio_mask[0, len(text_tokens) :] = True
    gen_start = len(text_tokens)
    remaining = tokens
    for step in range(steps):
        if remaining == 0:
            break
        gen = run_step(emb, llm, heads, input_ids, audio_mask)[0, :, gen_start:, :]
        real = gen[:, :, :1024]
        prob = np.exp(real - real.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        conf = (prob.max(-1) * CB_WEIGHTS[:, None]).sum(0)
        masked = np.where(input_ids[0, 0, gen_start:] == MASK_ID)[0]
        if not len(masked):
            break
        n_this = max(1, int(np.ceil(len(masked) / max(1, steps - step))))
        for best in masked[np.argsort(-conf[masked])][:n_this]:
            for cb in range(CODEBOOKS):
                input_ids[0, cb, gen_start + best] = int(gen[cb, best, :1024].argmax())
        remaining -= n_this
    return input_ids[0, :, gen_start:]


def metrics(w: np.ndarray, sr: int = 24000) -> dict:
    n = min(8192, len(w))
    X = np.abs(np.fft.rfft(w[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / sr)
    p = X**2
    p /= p.sum() + 1e-12
    band = (f >= 300) & (f <= 3400)
    return {
        "peak": float(np.abs(w).max()),
        "rms": float(np.sqrt((w**2).mean())),
        "centroid": float((f * p).sum()),
        "speech_band_pct": float(p[band].sum() * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--text", default="Hello world.")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    if args.download:
        download_fp16()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS / "int4")
    text_tokens = tok.encode(args.text, add_special_tokens=True)

    spec = VARIANTS[args.variant]
    emb, llm, heads = load(spec["emb"]), load(spec["llm"]), load(spec["heads"])
    codes = generate(emb, llm, heads, text_tokens, args.tokens, args.steps)

    import soundfile as sf

    hd = load("audio_tokenizer/higgs_decoder.onnx")
    w = hd.run(["waveform_24k"], {"codes": codes[:, None, :].astype(np.int64)})[0].squeeze()
    sf.write(GOLDEN / f"ab_{args.variant}.wav", w, 24000)
    m = metrics(w)
    print(
        f"{args.variant}: tokens={args.tokens} steps={args.steps} dur={len(w)/24000:.2f}s "
        f"peak={m['peak']:.3f} rms={m['rms']:.3f} centroid={m['centroid']:.0f}Hz speech_band={m['speech_band_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
