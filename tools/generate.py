"""Reference greedy iterative-unmask generation for golden comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
GOLDEN = ROOT / "golden"

HIDDEN = 1024
CODEBOOKS = 8
MASK_ID = 1024
CB_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]


def sessions(provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def load(name):
        return ort.InferenceSession(str(MODELS / "int4" / name), sess_options=opts, providers=[provider])

    return {
        "embeddings": load("audio_embeddings_encoder.onnx"),
        "llm": load("llm_decoder.onnx"),
        "heads": load("audio_heads_decoder.onnx"),
    }


def tokenize(text: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS / "int4")
    return tok.encode(text, add_special_tokens=True)


def run_backbone_step(sess, input_ids: np.ndarray, audio_mask: np.ndarray) -> np.ndarray:
    embeds = sess["embeddings"].run(
        ["inputs_embeds"], {"input_ids": input_ids, "audio_mask": audio_mask}
    )[0]
    b, s, _ = embeds.shape
    feed = {"inputs_embeds": embeds.astype(np.float32)}
    for inp in sess["llm"].get_inputs():
        if inp.name == "attention_mask":
            feed[inp.name] = np.ones((b, s), dtype=np.int64)
        elif "past" in inp.name:
            feed[inp.name] = np.zeros((b, 8, 0, 128), dtype=np.float32)
    hidden = sess["llm"].run(["hidden_states"], feed)[0]
    logits = sess["heads"].run(["logits"], {"hidden_states": hidden.astype(np.float32)})[0]
    return logits


def iterative_unmask(sess, text_tokens, num_audio_tokens, num_steps, prefix_codes=None, capture=None):
    t_text = len(text_tokens)
    t_ref = prefix_codes.shape[2] if prefix_codes is not None else 0
    seq = t_text + t_ref + num_audio_tokens

    input_ids = np.zeros((1, CODEBOOKS, seq), dtype=np.int64)
    input_ids[0, :, :t_text] = text_tokens
    if t_ref:
        for cb in range(CODEBOOKS):
            input_ids[0, cb, t_text : t_text + t_ref] = prefix_codes[cb, 0, :]
    input_ids[0, :, t_text + t_ref :] = MASK_ID

    audio_mask = np.zeros((1, seq), dtype=bool)
    audio_mask[0, t_text + t_ref :] = True

    weights = np.array(CB_WEIGHTS, dtype=np.float32)
    weights /= weights.sum()

    gen_start = t_text + t_ref
    num_masked = num_audio_tokens

    for step in range(num_steps):
        if num_masked == 0:
            break
        logits = run_backbone_step(sess, input_ids, audio_mask)
        gen_logits = logits[0, :, gen_start:, :]
        if capture is not None and step == 0:
            capture.append(gen_logits.astype(np.float32))
        real = gen_logits[:, :, :1024]
        prob = np.exp(real - real.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        max_prob = prob.max(axis=-1)
        confidence = (max_prob * weights[:, None]).sum(axis=0)

        masked = np.where(input_ids[0, 0, gen_start:] == MASK_ID)[0]
        if len(masked) == 0:
            break
        remaining_steps = max(1, num_steps - step)
        n_this = max(1, int(np.ceil(len(masked) / remaining_steps)))
        order = masked[np.argsort(-confidence[masked])][:n_this]
        for best in order:
            for cb in range(CODEBOOKS):
                input_ids[0, cb, gen_start + best] = int(gen_logits[cb, best, :1024].argmax())
        num_masked -= len(order)

    leftover = np.where(input_ids[0, 0, gen_start:] == MASK_ID)[0]
    if len(leftover):
        gl = run_backbone_step(sess, input_ids, audio_mask)[0, :, gen_start:, :1024]
        for pos in leftover:
            for cb in range(CODEBOOKS):
                input_ids[0, cb, gen_start + pos] = int(gl[cb, pos].argmax())

    return input_ids[0, :, gen_start:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello, kaise ho tum? Let's test.")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    GOLDEN.mkdir(parents=True, exist_ok=True)
    sess = sessions(args.provider)
    text_tokens = tokenize(args.text)
    capture: list = []
    codes = iterative_unmask(sess, text_tokens, args.tokens, args.steps, capture=capture)
    print(f"text tokens ({len(text_tokens)}): {text_tokens}")
    print(f"codes: {codes.shape}")

    np.ascontiguousarray(capture[0].reshape(-1)).tofile(
        GOLDEN / f"generation{args.suffix}_step0.bin"
    )

    np.ascontiguousarray(np.array(text_tokens, dtype=np.int64)).tofile(
        GOLDEN / f"generation{args.suffix}_tokens.bin"
    )
    np.ascontiguousarray(codes.astype(np.int64)).tofile(
        GOLDEN / f"generation{args.suffix}_codes.bin"
    )
    (GOLDEN / f"generation{args.suffix}.json").write_text(
        json.dumps(
            {
                "text": args.text,
                "text_tokens": text_tokens,
                "num_audio_tokens": args.tokens,
                "num_steps": args.steps,
                "codes_shape": list(codes.shape),
                "step0_shape": list(capture[0].shape),
                "provider": args.provider,
            },
            indent=2,
        )
    )
    print(f"wrote generation fixtures to {GOLDEN}")


if __name__ == "__main__":
    main()
