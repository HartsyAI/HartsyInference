"""Hy4 Flash FP8 fixtures, built from the tiny-config dump in data/.

Writes, under data/:
  fp8/            the tiny model with the routed experts and the three attention projections in e4m3 with 128x128 BF16 block scales
                  (names and scale keys as the released checkpoint writes them: <name>_scale_inv), config.json with quantization_config.
  fp8_twin/       the same model with those tensors dequantized to F32 (the values fp8/ decodes to). Exact comparison target.
  fp8/block_case.json  a block dequantization case with partial edge blocks, expected values computed by torch's own float8 decoder.

Run in hartsy-ref:cpu:  python make_fp8_fixtures.py /wt/tests/python-reference/hy4_flash/data
The released FP8 repo keeps the embedding, head, shared experts, router, projections not listed here and the Engram tables in BF16; this tiny
fixture keeps everything it does not quantize in F32, so only the FP8 read path is under test, not the mixed-dtype layout.
"""

import base64
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

BLOCK = 128
E4M3_MAX = 448.0
QUANTIZED = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$|\.self_attn\.(o_proj|q_b_proj|linear_gate)\.weight$")


def quantize(w):
    """w: float32 [..., out, in] -> (e4m3 weight [..., out, in], BF16 block scales [..., ceil(out/128), ceil(in/128)], dequantized F32)."""
    *lead, out, inn = w.shape
    gr, gc = -(-out // BLOCK), -(-inn // BLOCK)
    padded = torch.zeros(*lead, gr * BLOCK, gc * BLOCK, dtype=torch.float32)
    padded[..., :out, :inn] = w
    blocks = padded.reshape(*lead, gr, BLOCK, gc, BLOCK)
    amax = blocks.abs().amax(dim=(-3, -1))
    scale = (amax / E4M3_MAX).clamp(min=1e-12).to(torch.bfloat16)
    s = scale.float()[..., :, None, :, None]
    q = (blocks / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    deq = (q.float() * s).reshape(*lead, gr * BLOCK, gc * BLOCK)[..., :out, :inn].contiguous()
    qw = q.reshape(*lead, gr * BLOCK, gc * BLOCK)[..., :out, :inn].contiguous()
    return qw, scale.contiguous(), deq


def block_case(out_file):
    gen = torch.Generator().manual_seed(9)
    shape = (1, 130, 150)
    raw = torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x3C
    scale = (torch.rand((1, 2, 2), generator=gen) * 0.5 + 0.1).to(torch.bfloat16).float()
    values = raw.view(torch.float8_e4m3fn).float()
    rows = np.arange(130) // BLOCK
    cols = np.arange(150) // BLOCK
    expanded = scale[0][rows][:, cols]
    expected = (values[0] * expanded).numpy().astype("<f4")
    encode = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
    out_file.write_text(json.dumps({
        "shape": list(shape),
        "raw_b64": encode(raw.numpy()),
        "scale_b64": encode(scale.numpy().astype("<f4")),
        "expected_b64": encode(expected),
    }))


def main(data):
    data = Path(data)
    weights = load_file(str(data / "weights.safetensors"))
    fp8, twin, quantized = {}, {}, []
    for name, t in weights.items():
        if QUANTIZED.search(name):
            qw, scale, deq = quantize(t.float())
            fp8[name] = qw
            fp8[name + "_scale_inv"] = scale
            twin[name] = deq
            quantized.append(name)
        else:
            fp8[name] = t
            twin[name] = t

    (data / "fp8").mkdir(exist_ok=True)
    (data / "fp8_twin").mkdir(exist_ok=True)
    save_file(fp8, str(data / "fp8" / "weights.safetensors"), metadata={"format": "pt"})
    save_file(twin, str(data / "fp8_twin" / "weights.safetensors"), metadata={"format": "pt"})

    config = json.loads((data / "config.json").read_text())
    shutil.copy(data / "config.json", data / "fp8_twin" / "config.json")
    config["text_config"]["quantization_config"] = {
        "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [BLOCK, BLOCK],
    }
    (data / "fp8" / "config.json").write_text(json.dumps(config, indent=2))

    block_case(data / "fp8" / "block_case.json")
    print(f"quantized {len(quantized)} tensors; fp8 and twin written under {data}")


if __name__ == "__main__":
    main(sys.argv[1])
