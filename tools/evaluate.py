"""Evaluate generated TTS audio: round-trip WER + spectral metrics + RTF.

Runs in `tools/.venv` (numpy, scipy, soundfile, faster-whisper).

    tools/.venv/bin/python tools/evaluate.py --wav out.wav --text "hello world" [--gen-seconds 2.0]
    tools/.venv/bin/python tools/evaluate.py --manifest eval/manifest.json

Manifest rows: {"name": str, "wav": str, "text": str, "gen_seconds": float?}
"""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent


def load_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    return audio, sr


def resample_to(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    g = gcd(target, sr)
    return resample_poly(audio, target // g, sr // g).astype(np.float32)


def spectral(audio: np.ndarray, sr: int) -> dict[str, float]:
    windowed = audio * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    power = spectrum**2
    total = power[(freqs > 20) & (freqs < sr * 0.45)].sum()
    speech = power[(freqs > 300) & (freqs < 3400)].sum()
    return {
        "centroid_hz": float((freqs * spectrum).sum() / max(1e-9, spectrum.sum())),
        "speech_band": float(speech / total) if total > 0 else 0.0,
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[-1][-1] / max(1, len(ref))


_ASR = None


def transcribe(audio: np.ndarray, sr: int) -> str:
    global _ASR
    from faster_whisper import WhisperModel

    if _ASR is None:
        _ASR = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = _ASR.transcribe(resample_to(audio, sr, 16000), vad_filter=False)
    return " ".join(segment.text for segment in segments).strip()


def evaluate(name: str, wav: str, text: str | None, gen_seconds: float | None) -> dict:
    audio, sr = load_mono(wav)
    duration = len(audio) / sr
    row: dict[str, object] = {
        "name": name,
        "wav": wav,
        "duration_s": round(duration, 3),
        "rms": round(float(np.sqrt(np.mean(audio**2))), 4),
        "peak": round(float(np.abs(audio).max()), 4),
        **{k: round(v, 4) for k, v in spectral(audio, sr).items()},
    }
    if text:
        hypothesis = transcribe(audio, sr)
        row["asr"] = hypothesis
        row["wer"] = round(word_error_rate(text, hypothesis), 4)
    if gen_seconds:
        row["gen_seconds"] = round(gen_seconds, 3)
        row["rtf"] = round(gen_seconds / duration, 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--text")
    ap.add_argument("--name", default="sample")
    ap.add_argument("--gen-seconds", type=float)
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.manifest:
        for item in json.loads(Path(args.manifest).read_text()):
            rows.append(
                evaluate(
                    item.get("name", Path(item["wav"]).stem),
                    item["wav"],
                    item.get("text"),
                    item.get("gen_seconds"),
                )
            )
    elif args.wav:
        rows.append(evaluate(args.name, args.wav, args.text, args.gen_seconds))
    else:
        ap.error("provide --wav or --manifest")

    for row in rows:
        print(json.dumps(row))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
