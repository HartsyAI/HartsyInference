"""Dumps comfy-kitchen's eager INT8 / INT8-ConvRot reference for the C# parity tests.

ComfyUI ships LTX 2.5 and MiniMax-H3 as `int8_tensorwise` with `convrot: true, convrot_groupsize: 256`.
The eager backend is the authority (comfy_kitchen/backends/eager/quantization.py); its CUDA and Triton
backends are bit-checked against it upstream, so parity against eager is parity against ComfyUI.

Run with the ComfyUI venv's python (there is no system torch), pointing PYTHONPATH at the vendored
comfy_kitchen so the CUDA extension registry is bypassed entirely:

  PYTHONPATH=/home/hartsy/Desktop/HartsyInference/Models/bench-comfy/pylibs \
  "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/bin/python3" \
  int8_convrot_reference.py

Writes int8_convrot_reference.bin next to this file (gitignored; the parity test skips when absent).
"""
import struct
import torch

from comfy_kitchen.tensor.int8_utils import _build_hadamard
from comfy_kitchen.backends.eager.quantization import (
    int8_linear,
    quantize_int8_convrot_weight,
    quantize_int8_rowwise,
)

# (N, K, M, group) — group 0 means the layer is quantized unrotated, which is what ComfyUI does
# whenever in_features % 256 != 0. Everything runs in float32: the engine's F32 Linear path is the
# one that can match eager closely, and a bf16 dump would fold torch's rounding into the tolerance.
CASES = [
    (128, 256, 8, 256),
    (64, 512, 33, 256),      # M not a multiple of 32 — exercises the IMMA row padding
    (96, 320, 16, 0),        # K % 256 != 0 → unrotated, still per-row weight scale
    (256, 1024, 64, 256),
    (64, 256, 1, 256),       # M = 1
    (128, 256, 8, 16),       # group != 256 (16 = 4^2, still a power of four)
]

GROUPS = sorted({g for *_, g in CASES if g > 0})


def f32(t):
    return t.contiguous().to(torch.float32).numpy().astype("<f4").tobytes()


g = torch.Generator().manual_seed(20260812)
with open("int8_convrot_reference.bin", "wb") as f:
    f.write(struct.pack("<i", len(GROUPS)))
    for size in GROUPS:
        h = _build_hadamard(size, dtype=torch.float32)
        f.write(struct.pack("<i", size))
        f.write(f32(h))
        print(f"hadamard {size}: symmetric={torch.equal(h, h.T)} "
              f"HH-I absmax={(h @ h - torch.eye(size)).abs().max():.3e}")

    f.write(struct.pack("<i", len(CASES)))
    for (n, k, m, group) in CASES:
        w = torch.randn(n, k, generator=g, dtype=torch.float32)
        x = torch.randn(m, k, generator=g, dtype=torch.float32)
        bias = torch.randn(n, generator=g, dtype=torch.float32)

        if group > 0:
            q, scale = quantize_int8_convrot_weight(w.clone(), group, stochastic_rounding=0)
        else:
            q, scale = quantize_int8_rowwise(w.clone(), stochastic_rounding=0)

        out = int8_linear(x.clone(), q, scale, None, torch.float32, group > 0, max(group, 1))
        out_bias = int8_linear(x.clone(), q, scale, bias, torch.float32, group > 0, max(group, 1))

        f.write(struct.pack("<4i", n, k, m, group))
        f.write(f32(w))
        f.write(q.contiguous().numpy().astype("<i1").tobytes())
        f.write(f32(scale.reshape(-1)))
        f.write(f32(x))
        f.write(f32(bias))
        f.write(f32(out))
        f.write(f32(out_bias))
        print(f"case N{n} K{k} M{m} G{group}: scale[0]={scale.reshape(-1)[0]:.6e} "
              f"out absmax={out.abs().max():.5f}")

print("wrote int8_convrot_reference.bin")
