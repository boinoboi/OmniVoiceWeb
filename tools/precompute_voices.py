"""Precompute reference voice codes into `public/voices.json` (fast loading, no in-browser encode).

Encodes one or more reference clips with the Higgs encoders, applying the same preprocessing as the
runtime (RMS normalise + edge-silence + hop trim), and writes base64 int32 codes.

    tools/.venv/bin/python tools/precompute_voices.py \
        --voice demo=../refs/reference.wav --voice-text demo=../refs/reference.txt \
        --out ../public/voices.json
"""

from __future__ import annotations

import argparse
import base64
import json
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
HOP = 960
SAMPLE_RATE = 24000


def preprocess(wav: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        wav = resample_poly(wav, SAMPLE_RATE // g, sr // g).astype(np.float32)
    rms = float(np.sqrt(np.mean(wav**2)))
    if 0 < rms < 0.1:
        wav = wav * (0.1 / rms)
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak > 0:
        above = np.where(np.abs(wav) >= 0.01 * peak)[0]
        if above.size:
            wav = wav[above[0] : above[-1] + 1]
    usable = (wav.shape[0] // HOP) * HOP
    if usable > 0:
        wav = wav[:usable]
    return wav.astype(np.float32), rms


def encode(path: str, provider: str) -> tuple[np.ndarray, float]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav, _rms = preprocess(wav, sr)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def s(name: str) -> ort.InferenceSession:
        return ort.InferenceSession(str(MODELS / "audio_tokenizer" / name), sess_options=opts, providers=[provider])

    acoustic = s("acoustic_encoder.onnx").run(
        ["acoustic_features"], {"waveform_24k": wav[None, None, :]}
    )[0]
    wav16 = resample_poly(wav, 2, 3).astype(np.float32)
    semantic = s("semantic_encoder.onnx").run(
        ["semantic_features"], {"waveform_16k": wav16[None, :]}
    )[0]
    t = min(acoustic.shape[2], semantic.shape[2])
    codes = s("quantizer_encoder.onnx").run(
        ["codes"], {"acoustic_features": acoustic[:, :, :t], "semantic_features": semantic[:, :, :t]}
    )[0][:, 0, :]
    rms = float(np.sqrt(np.mean(wav**2)))
    return codes.astype(np.int32), rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", required=True, help="id=path (repeatable)")
    ap.add_argument("--voice-text", action="append", default=[], help="id=path to transcript (repeatable)")
    ap.add_argument("--out", default=str(ROOT.parent / "public" / "voices.json"))
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    args = ap.parse_args()

    texts = dict(v.split("=", 1) for v in args.voice_text)
    voices = []
    for item in args.voice:
        vid, path = item.split("=", 1)
        codes, rms = encode(path, args.provider)
        frames = int(codes.shape[1])
        text = ""
        if vid in texts:
            text = Path(texts[vid]).read_text().strip()
        voices.append(
            {
                "id": vid,
                "name": vid.upper(),
                "text": text,
                "frames": frames,
                "rms": round(rms, 5),
                "codes": base64.b64encode(codes.tobytes()).decode("ascii"),
            }
        )
        print(f"encoded {vid}: {frames} frames, rms {rms:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"voices": voices}))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
