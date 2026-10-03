"""Generate golden vectors from the OmniVoice ONNX sub-models for TS parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = "onnx-community/OmniVoice-Onnx"
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
GOLDEN = ROOT / "golden"

FILES = [
    "int4/audio_embeddings_encoder.onnx",
    "int4/audio_embeddings_encoder.onnx.data",
    "int4/audio_heads_decoder.onnx",
    "int4/llm_decoder.onnx",
    "int4/llm_decoder.onnx.data",
    "int4/tokenizer.json",
    "int4/tokenizer_config.json",
    "int4/config.json",
]

HIDDEN = 1024
CODEBOOKS = 8

DTYPES = {
    np.dtype("float32"): "float32",
    np.dtype("int64"): "int64",
    np.dtype("bool"): "bool",
}


def download() -> None:
    from huggingface_hub import hf_hub_download

    for name in FILES:
        path = hf_hub_download(REPO, name, local_dir=MODELS)
        print(f"  {name} -> {path}")


def session(rel: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        str(MODELS / rel), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def tensor_desc(name: str, array: np.ndarray | None, case: str, suffix: str, **special) -> dict:
    if array is not None:
        dtype = DTYPES[array.dtype]
        shape = list(array.shape)
        rel = f"{case}_{suffix}.bin"
        (GOLDEN / rel).write_bytes(np.ascontiguousarray(array).tobytes())
        return {"name": name, "dtype": dtype, "shape": shape, "file": rel}
    return {"name": name, **special}


def gen_heads(rng: np.random.Generator, seq: int = 16):
    sess = session("int4/audio_heads_decoder.onnx")
    hidden = (rng.standard_normal((1, seq, HIDDEN)) * 0.1).astype(np.float32)
    logits = sess.run(["logits"], {"hidden_states": hidden})[0]
    print(f"heads: {hidden.shape} -> {logits.shape}")
    return {
        "model": "int4/audio_heads_decoder.onnx",
        "inputs": [tensor_desc("hidden_states", hidden, "heads", "input")],
        "outputs": [tensor_desc("logits", logits, "heads", "output")],
    }


def gen_embeddings(rng: np.random.Generator, text: int = 8, audio: int = 12):
    sess = session("int4/audio_embeddings_encoder.onnx")
    seq = text + audio
    input_ids = np.zeros((1, CODEBOOKS, seq), dtype=np.int64)
    audio_mask = np.zeros((1, seq), dtype=bool)
    input_ids[0, :, :text] = rng.integers(100, 200, size=(CODEBOOKS, text))
    input_ids[0, :, text:] = rng.integers(0, 1024, size=(CODEBOOKS, audio))
    audio_mask[0, text:] = True
    embeds = sess.run(["inputs_embeds"], {"input_ids": input_ids, "audio_mask": audio_mask})[0]
    print(f"embeddings: {input_ids.shape} -> {embeds.shape}")
    return {
        "model": "int4/audio_embeddings_encoder.onnx",
        "inputs": [
            tensor_desc("input_ids", input_ids, "embeddings", "input_ids"),
            tensor_desc("audio_mask", audio_mask, "embeddings", "audio_mask"),
        ],
        "outputs": [tensor_desc("inputs_embeds", embeds, "embeddings", "output")],
    }


def backbone_inputs(sess, embeds: np.ndarray) -> dict:
    b, s, _ = embeds.shape
    feed = {"inputs_embeds": embeds}
    for inp in sess.get_inputs():
        if inp.name == "attention_mask":
            feed[inp.name] = np.ones((b, s), dtype=np.int64)
        elif "past" in inp.name:
            feed[inp.name] = np.zeros((b, 8, 0, 128), dtype=np.float32)
    return feed


def gen_backbone(rng: np.random.Generator, seq: int = 20):
    sess = session("int4/llm_decoder.onnx")
    embeds = (rng.standard_normal((1, seq, HIDDEN)) * 0.5).astype(np.float32)
    hidden = sess.run(["hidden_states"], backbone_inputs(sess, embeds))[0]
    print(f"backbone: {embeds.shape} -> {hidden.shape}")

    inputs = [tensor_desc("inputs_embeds", embeds, "backbone", "input")]
    inputs.append(tensor_desc("attention_mask", None, "backbone", "", ones=True, shape=[1, seq], dtype="int64"))
    for inp in sess.get_inputs():
        if "past" in inp.name:
            inputs.append(
                tensor_desc(inp.name, None, "backbone", "", zeros=True, shape=[1, 8, 0, 128], dtype="float32")
            )
    return {
        "model": "int4/llm_decoder.onnx",
        "inputs": inputs,
        "outputs": [tensor_desc("hidden_states", hidden, "backbone", "output")],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    if args.download:
        download()

    rng = np.random.default_rng(args.seed)
    cases = [gen_heads(rng), gen_embeddings(rng), gen_backbone(rng)]
    (GOLDEN / "parity.json").write_text(
        json.dumps({"seed": args.seed, "cases": cases}, indent=2)
    )
    print(f"wrote parity fixtures to {GOLDEN}")


if __name__ == "__main__":
    main()
