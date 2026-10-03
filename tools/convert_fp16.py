from __future__ import annotations
import argparse
from pathlib import Path
import onnx
from onnxconverter_common import float16
ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "models/bidir/llm_decoder.onnx"))
    ap.add_argument("--output", default=str(ROOT / "models/bidir/llm_decoder_fp16.onnx"))
    args = ap.parse_args()
    m = onnx.load(args.input)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    out = Path(args.output)
    onnx.save_model(m16, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=out.name + ".data", size_threshold=1024)
    size = sum(f.stat().st_size for f in out.parent.glob(out.name + "*"))
    print(f"wrote {out} ({size/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
