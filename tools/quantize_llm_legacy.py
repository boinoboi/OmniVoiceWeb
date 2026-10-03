"""Weight-only int4 quantization of the bidirectional OmniVoice LLM (MatMul -> MatMulNBits).

Requires ORT 1.20 + onnx 1.16 in a throwaway venv (ORT 1.30's new int4 quantizer is broken and the
legacy MatMul4BitsQuantizer was removed from 1.30). Pass the raw ModelProto to MatMul4BitsQuantizer;
wrapping it in an ONNXModel first double-wraps and crashes with "'method' object is not iterable".

    uv venv --python 3.12 tools/.venv-quant
    uv pip install --python tools/.venv-quant onnxruntime==1.20.1 onnx==1.16.2 numpy
    tools/.venv-quant/bin/python tools/quantize_llm_legacy.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_4bits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMul4BitsQuantizer,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "models/bidir/llm_decoder.onnx"))
    ap.add_argument("--output", default=str(ROOT / "models/bidir/llm_decoder_int4.onnx"))
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    start = time.time()
    model = onnx.load(args.input)
    quantizer = MatMul4BitsQuantizer(
        model,
        algo_config=DefaultWeightOnlyQuantConfig(block_size=args.block_size, is_symmetric=True),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(args.output, use_external_data_format=True)

    out = Path(args.output)
    size = sum(f.stat().st_size for f in out.parent.glob(f"{out.name}*"))
    graph = onnx.load(args.output, load_external_data=False)
    ops: dict[str, int] = {}
    for node in graph.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"done: {size / 1e6:.0f} MB in {time.time() - start:.1f}s")
    print("MatMulNBits", ops.get("MatMulNBits", 0), "MatMul", ops.get("MatMul", 0))


if __name__ == "__main__":
    main()
