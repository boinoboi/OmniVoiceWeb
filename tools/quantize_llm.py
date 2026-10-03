"""Weight-only int4 quantization of the bidirectional OmniVoice LLM (MatMul -> MatMulNBits)."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import DynamicQuantConfig, QuantType, quantize

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "models/bidir/llm_decoder.onnx"))
    ap.add_argument("--output", default=str(ROOT / "models/bidir/llm_decoder_int4.onnx"))
    ap.add_argument("--weight-type", default="int4", choices=["int4", "int8", "uint4"])
    args = ap.parse_args()

    weight_type = {"int4": QuantType.QInt4, "uint4": QuantType.QUInt4, "int8": QuantType.QInt8}[args.weight_type]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"quantizing {args.input} -> {out} ({args.weight_type}, MatMul only)")
    quantize(
        args.input,
        str(out),
        DynamicQuantConfig(weight_type=weight_type, op_types_to_quantize=["MatMul"]),
    )
    size = sum(f.stat().st_size for f in out.parent.glob(f"{out.name}*"))
    print(f"done: {size/1e6:.0f} MB")

    import onnx

    m = onnx.load(str(out), load_external_data=False)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("MatMulNBits:", ops.get("MatMulNBits", 0), "MatMul:", ops.get("MatMul", 0))


if __name__ == "__main__":
    main()
