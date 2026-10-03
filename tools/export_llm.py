"""Re-export the OmniVoice Qwen3 core with FULL bidirectional attention (non-causal).

The onnx-community llm_decoder is causal, which breaks the masked-diffusion loop. This
exports the same weights with a 4-D all-zero additive mask (full attention), matching the
reference block mask used by k2-fsa/OmniVoice `_generate_iterative`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "bidir"


class BidirLLM(nn.Module):
    def __init__(self, llm: nn.Module):
        super().__init__()
        self.llm = llm

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        b, s, _ = inputs_embeds.shape
        mask = torch.zeros((b, 1, s, s), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=mask, return_dict=True)
        return out[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--name", default="llm_decoder.onnx")
    args = ap.parse_args()

    from omnivoice.models.omnivoice import OmniVoice

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    print(f"loading OmniVoice ({args.dtype}, cpu)…")
    model = OmniVoice.from_pretrained(args.model, device_map="cpu", dtype=torch_dtype)
    llm = model.llm
    llm.eval()
    llm.config._attn_implementation = "eager"
    llm.config.use_cache = False

    wrapper = BidirLLM(llm).eval()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = outdir / args.name

    dummy = torch.randn(1, 16, llm.config.hidden_size, dtype=torch_dtype)

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, inputs_embeds):
            return self.inner(inputs_embeds)

    with torch.no_grad():
        torch.onnx.export(
            Export(wrapper),
            (dummy,),
            str(onnx_path),
            input_names=["inputs_embeds"],
            output_names=["hidden_states"],
            dynamic_axes={"inputs_embeds": {0: "batch", 1: "sequence"}, "hidden_states": {0: "batch", 1: "sequence"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )
    size = sum(f.stat().st_size for f in outdir.glob("*.onnx*"))
    print(f"wrote {onnx_path} ({size/1e6:.0f} MB)")

    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    gqa = [n.op_type for n in m.graph.node if n.op_type == "GroupQueryAttention"]
    print("GroupQueryAttention nodes:", len(gqa), "(should be 0)")


if __name__ == "__main__":
    main()
