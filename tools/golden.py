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
    "audio_tokenizer/fp16/acoustic_encoder.onnx",
    "audio_tokenizer/fp16/acoustic_encoder.onnx.data",
    "audio_tokenizer/fp16/semantic_encoder.onnx",
    "audio_tokenizer/fp16/semantic_encoder.onnx.data",
    "audio_tokenizer/fp16/quantizer_encoder.onnx",
    "audio_tokenizer/fp16/higgs_decoder.onnx",
    "audio_tokenizer/fp16/higgs_decoder.onnx.data",
    "audio_tokenizer/fp16/model_config.json",
    "audio_tokenizer/semantic_encoder.onnx",
    "audio_tokenizer/acoustic_encoder.onnx",
    "audio_tokenizer/quantizer_encoder.onnx",
    "audio_tokenizer/higgs_decoder.onnx",
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


def session(rel: str, provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    available = ort.get_available_providers()
    if provider == "auto":
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    return ort.InferenceSession(str(MODELS / rel), sess_options=opts, providers=[provider])


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


def gen_codec(rng: np.random.Generator):
    from math import gcd

    from scipy.signal import resample_poly

    sr24 = 24000
    n = sr24
    t = np.arange(n) / sr24
    wav24 = (
        0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.02 * rng.standard_normal(n)
    ).astype(np.float32)
    wav24 = np.clip(wav24, -1.0, 1.0)
    g = gcd(16000, sr24)
    wav16 = resample_poly(wav24, 16000 // g, sr24 // g).astype(np.float32)

    ac = session("audio_tokenizer/acoustic_encoder.onnx", "auto")
    se = session("audio_tokenizer/semantic_encoder.onnx", "auto")
    qe = session("audio_tokenizer/quantizer_encoder.onnx", "auto")
    hd = session("audio_tokenizer/higgs_decoder.onnx", "auto")
    print("higgs acoustic inputs:", [i.name for i in ac.get_inputs()])
    print("higgs semantic inputs:", [i.name for i in se.get_inputs()])
    print("higgs quantizer inputs:", [i.name for i in qe.get_inputs()])
    print("higgs decoder inputs:", [i.name for i in hd.get_inputs()])

    wave24 = wav24[None, None, :]
    wave16 = wav16[None, :]
    acoustic = ac.run(["acoustic_features"], {"waveform_24k": wave24})[0]
    semantic = se.run(["semantic_features"], {"waveform_16k": wave16})[0]
    frames = min(acoustic.shape[2], semantic.shape[2])
    acoustic = acoustic[:, :, :frames]
    semantic = semantic[:, :, :frames]
    codes = qe.run(["codes"], {"acoustic_features": acoustic, "semantic_features": semantic})[0]
    decoded = hd.run(["waveform_24k"], {"codes": codes})[0]
    print(f"codec: wav24 {wave24.shape} -> acoustic {acoustic.shape} + semantic {semantic.shape}")
    print(f"codec: codes {codes.shape} -> waveform {decoded.shape}")

    return [
        {
            "model": "audio_tokenizer/acoustic_encoder.onnx",
            "inputs": [tensor_desc("waveform_24k", wave24, "codec", "waveform24")],
            "outputs": [tensor_desc("acoustic_features", acoustic, "codec", "acoustic")],
        },
        {
            "model": "audio_tokenizer/semantic_encoder.onnx",
            "inputs": [tensor_desc("waveform_16k", wave16, "codec", "waveform16")],
            "outputs": [tensor_desc("semantic_features", semantic, "codec", "semantic")],
        },
        {
            "model": "audio_tokenizer/quantizer_encoder.onnx",
            "inputs": [
                tensor_desc("acoustic_features", acoustic, "codec", "q_acoustic"),
                tensor_desc("semantic_features", semantic, "codec", "q_semantic"),
            ],
            "outputs": [tensor_desc("codes", codes, "codec", "codes")],
        },
        {
            "model": "audio_tokenizer/higgs_decoder.onnx",
            "inputs": [tensor_desc("codes", codes, "codec", "d_codes")],
            "outputs": [tensor_desc("waveform_24k", decoded, "codec", "decoded")],
        },
    ]


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
    cases.extend(gen_codec(rng))
    (GOLDEN / "parity.json").write_text(
        json.dumps({"seed": args.seed, "cases": cases}, indent=2)
    )
    print(f"wrote parity fixtures to {GOLDEN}")


if __name__ == "__main__":
    main()
