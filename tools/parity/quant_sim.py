"""Simulated low-precision steps for the offline per-layer references, shared by every model that uses them.

hadamard: the orthonormal Walsh-Hadamard rotation that sparse-attention indexers apply to queries and keys before quantizing
them (vLLM-ERNIE nvidia/attention.py hadamard_rows over vllm ops.hadacore_transform; DeepSeek-V4 inference/model.py
rotate_activation, hadamard_transform with scale dim ** -0.5).

fp8_roundtrip_ue8m0: quantize to FP8 E4M3 and back with one power-of-two (UE8M0) scale per group of `block` values along the
last dim, the indexer cache's format (vLLM-ERNIE: per_token_group_quant_fp8(q, 128, use_ue8m0=True) for queries and
SparseAttnIndexer(..., 128, "ue8m0", ...) for keys; DeepSeek-V4 inference/kernel.py act_quant with fast_round_scale).
"""

from __future__ import annotations

import torch

F32 = torch.float32
FP8_MAX = 448.0


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform (Sylvester order) over the last dim, a power of two."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f"the Hadamard width must be a power of two, got {n}")
    y = x.to(F32).reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        y = torch.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], dim=2)
        h *= 2
    return y.reshape(x.shape) * n ** -0.5


def fp8_roundtrip_ue8m0(x: torch.Tensor, block: int = 128, eps: float = 1e-10) -> torch.Tensor:
    """x through FP8 E4M3 with a power-of-two scale per `block` values: scale = 2^ceil(log2(amax / 448))."""
    n = x.shape[-1]
    if n % block:
        raise ValueError(f"the last dim ({n}) must be a multiple of the block ({block})")
    g = x.to(F32).reshape(-1, n // block, block)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (g / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(F32)
    return (q * scale).reshape(x.shape)
