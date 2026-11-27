"""Compressed-KV pooling for the offline per-layer references.

The DeepSeek-V4 family compressor: every `ratio` consecutive tokens are pooled into one KV latent by a learned softmax gate.
ERNIE-5.5 HCA layers reuse it unchanged (vLLM-ERNIE's Ernie5_5Compressor subclasses vLLM's DeepseekCompressor), so any
reference for a model with this compressor imports this module instead of writing its own.

Follows deepseek-ai/DeepSeek-V4-Flash at 60d8d70770c6776ff598c94bb586a859a38244f1, inference/model.py: Compressor.forward
and get_compress_topk_idxs. That file is the plain-torch form of vLLM's DeepseekCompressor (vllm v0.32.0
models/deepseek_v4/compressor.py), whose kernel comment gives the same RoPE position and order (compress, RMS norm, RoPE).
It is not derived from the engine's KvCompressorState, so a parity run checks the engine against it.

What a caller gets: the latents after the RMS norm and before RoPE. The caller rotates the trailing rope dims at
group_starts(), because the RoPE table (and its YaRN variant) is the model's.

Not covered:
- Overlapping windows (DeepSeek-V4's ratio-4 layers): refused. The engine refuses them too.
- The source casts the pooled latent to the activation dtype (bf16) before the norm, and simulates FP8 storage
  (act_quant) on the non-rope dims after RoPE. Both are skipped: the latents stay F32, as the engine's host path keeps them.
"""

from __future__ import annotations

import torch

F32 = torch.float32


def pool(kv: torch.Tensor, score: torch.Tensor, ratio: int, ape: torch.Tensor | None = None) -> torch.Tensor:
    """One latent per completed group, from position 0: [T // ratio, d].

    kv, score: [T, d], the wkv and wgate projections of the attention input. ape: [ratio, d], added to the score of the
    token in slot `position % ratio`. The softmax runs over the group's tokens, per channel. A trailing partial group is
    not pooled; in decode the source holds it in its state until the group's last token arrives.
    """
    if ratio < 2:
        raise ValueError("ratio must be at least 2; ratio 1 keeps one latent per token and has no pooling.")
    if kv.shape != score.shape or kv.dim() != 2:
        raise ValueError("kv and score must both be [T, d].")
    t, d = kv.shape
    full = t // ratio
    kv = kv[: full * ratio].to(F32).reshape(full, ratio, d)
    score = score[: full * ratio].to(F32).reshape(full, ratio, d)
    if ape is not None:
        if tuple(ape.shape) != (ratio, d):
            raise ValueError(f"ape must be [{ratio}, {d}], got {tuple(ape.shape)}.")
        score = score + ape.to(F32)
    return (kv * score.softmax(dim=1)).sum(dim=1)


def compress(x: torch.Tensor, wkv: torch.Tensor, wgate: torch.Tensor, ratio: int, norm_weight: torch.Tensor, eps: float,
             ape: torch.Tensor | None = None, overlap: bool = False) -> torch.Tensor:
    """Projects x [T, dim] with wkv and wgate ([d, dim] each), pools, and RMS-normalizes each latent: [T // ratio, d], pre-RoPE."""
    if overlap:
        raise NotImplementedError("Overlapping compression windows (DeepSeek-V4 ratio 4) are not implemented.")
    x = x.to(F32)
    latent = pool(x @ wkv.to(F32).T, x @ wgate.to(F32).T, ratio, ape)
    return latent * torch.rsqrt(latent.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.to(F32)


def group_starts(count: int, ratio: int) -> torch.Tensor:
    """The RoPE position of each latent: its group's first token (the source's freqs_cis[:cutoff:ratio])."""
    return torch.arange(count) * ratio


def visible(position: int, ratio: int) -> int:
    """How many latents the query at `position` attends: those whose group ends at or before it (get_compress_topk_idxs)."""
    return (position + 1) // ratio
