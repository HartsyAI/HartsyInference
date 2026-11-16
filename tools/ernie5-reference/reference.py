#!/usr/bin/env python3
"""Plain CPU torch reference for the ERNIE-5.5 decoder, one layer at a time.

Reads a checkpoint with safetensors. Each tensor is dequantized to F32 when it is read (FP8 blocks through their
128x128 F32 scale grid), used, and dropped, so peak host memory is one layer's attention weights plus the
experts that the current tokens select. The residual stream is written after every decoder layer.

Math follows baidu/vLLM-ERNIE src/vllm_ernie/models/ernie5_5 (commit pinned in README.md), with the shared
mHC post and head ops of vllm==0.32.0. Where the sources do not give a formula, the code raises
NotImplementedError or the gap is listed in README.md under "Not covered". Nothing is guessed silently.

Usage:
  python reference.py --ckpt DIR --tokens 1,2,3,4,5 --out DIR [--max-layers N] [--logits]

Outputs in --out:
  layer_LL_all.npy    float32 [T, hc, hidden]  residual stream after decoder layer LL, every token
  layer_LL_last.npy   float32 [hc, hidden]     the same, last token
  final_hidden.npy    float32 [T, hidden]      after the head read-out and final RMS norm
  logits_last.npy     float32 [vocab]          with --logits: LM head and multimax, last token
  meta.json           tokens, checkpoint, per-layer seconds
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

F32 = torch.float32
SCALE_SUFFIX = "_scale_inv"  # Ernie5WeightNames.ScaleSuffix
EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
FINAL_NORM = "model.norm.weight"
HEAD_FN = "model.mhc_contract.hc_head_fn"  # stored input-major [hc*hidden, hc]
HEAD_BASE = "model.mhc_contract.hc_head_base"
HEAD_SCALE = "model.mhc_contract.hc_head_scale"
MULTIMAX_RANGES = "model.lm_head.multimax_ranges"
MULTIMAX_GAINS = "model.lm_head.multimax_ts"


# --------------------------------------------------------------------------- checkpoint access


class Checkpoint:
    """The shards named by the index (or one model.safetensors), read as F32 one tensor at a time."""

    def __init__(self, directory: str) -> None:
        self.dir = Path(directory)
        self.cfg: dict = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            with safe_open(str(self.dir / "model.safetensors"), "pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        quant = self.cfg.get("quantization_config") or {}
        self.block = list(quant.get("weight_block_size") or [128, 128])
        self._handles: dict[str, object] = {}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def _raw(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"checkpoint has no tensor {key!r}")
        file = self.weight_map[key]
        if file not in self._handles:
            self._handles[file] = safe_open(str(self.dir / file), "pt")
        return self._handles[file].get_tensor(key)  # type: ignore[attr-defined]

    def read(self, key: str) -> torch.Tensor:
        """The tensor as F32. A float8_e4m3fn weight is dequantized through its block-scale grid."""
        t = self._raw(key)
        if t.dtype == torch.float8_e4m3fn:
            return dequant_block_fp8(t, self._raw(key + SCALE_SUFFIX), self.block)
        return t.to(F32)


def dequant_block_fp8(w: torch.Tensor, scale_inv: torch.Tensor, block: list[int]) -> torch.Tensor:
    """W = fp8 * scale_inv, each scale repeated over its bn x bk block (vllm-ernie quant_config.dequantize_block_fp8)."""
    n, k = w.shape
    bn, bk = block
    scale = scale_inv.to(F32).repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return w.to(F32) * scale


# --------------------------------------------------------------------------- primitives


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rms_unweighted(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rope_angles(positions: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv = theta ** (-torch.arange(0, dim, 2, dtype=F32) / dim)
    ang = positions.to(F32)[:, None] * inv[None, :]
    return ang.cos(), ang.sin()  # [T, dim/2]


def yarn_inv_freq(dim: int, base: float, factor: float, beta_fast: float, beta_slow: float, orig_max: int) -> torch.Tensor:
    """DeepSeek YaRN inverse frequencies: vllm-ernie config + vllm v0.32.0 rotary_embedding/deepseek_scaling_rope.py
    (_compute_inv_freq), with yarn_find_correction_range and yarn_linear_ramp_mask from rotary_embedding/common.py.
    The mscale ratio is 1 when mscale and mscale_all_dim are 0, as build_deepseek_v4_rope sets them.
    """
    pos = base ** (torch.arange(0, dim, 2, dtype=F32) / dim)
    extra, interp = 1.0 / pos, 1.0 / (factor * pos)

    def find_dim(rot: float) -> float:
        return (dim * math.log(orig_max / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(find_dim(beta_fast)), 0)
    high = min(math.ceil(find_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=F32) - low) / (high - low), 0, 1)
    return interp * ramp + extra * (1 - ramp)  # = interp * (1 - mask) + extra * mask, mask = 1 - ramp


def hca_rope_angles(cfg: dict, positions: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The HCA layers' RoPE (build_deepseek_v4_rope with compress_ratio > 1): theta = compress_rope_theta.

    A "default" rope type means factor 1, and YaRN with factor 1 equals plain RoPE, so the angle is plain.
    A YaRN type uses the DeepSeek YaRN frequencies above with original_max_position_embeddings.
    attention_factor == 1.0 selects deepseek_llama_scaling upstream; that formula is not read, so it refuses.
    """
    rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    theta = float(cfg["compress_rope_theta"])
    rope_type = str(rs.get("rope_type") or rs.get("type") or "default")
    if rope_type == "default":
        return rope_angles(positions, dim, theta)
    if rs.get("attention_factor") == 1.0:
        raise NotImplementedError("HCA rope with attention_factor 1.0 selects deepseek_llama_scaling; its formula is not implemented.")
    if rope_type not in ("yarn", "deepseek_yarn"):
        raise NotImplementedError(f"HCA rope type {rope_type!r} is not implemented.")
    inv = yarn_inv_freq(dim, theta, float(rs["factor"]), float(rs.get("beta_fast", 32)), float(rs.get("beta_slow", 1)),
                        int(rs["original_max_position_embeddings"]))
    ang = positions.to(F32)[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def _bcast(cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    return cos.reshape(shape), (-sin if inverse else sin).reshape(shape)


def rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """GPT-J layout: the pairs (2i, 2i+1) rotate together. x is [T, ..., dim]."""
    c, s = _bcast(cos, sin, x, inverse)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1).flatten(-2)


def rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half layout: the first and second halves of the dim rotate together."""
    c, s = _bcast(cos, sin, x, False)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def situ_mlp(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor, cfg: dict) -> torch.Tensor:
    """SiTU-GLU (vllm layers/activation.py SituAndMul): beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)."""
    beta, lbeta = float(cfg["activation_situ_beta"]), float(cfg["activation_situ_linear_beta"])
    g, u = x @ gate_w.T, x @ up_w.T
    return ((beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (lbeta * torch.tanh(u / lbeta))) @ down_w.T


# --------------------------------------------------------------------------- mHC


def mhc_pre(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
            cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ERNIE mHC pre (vllm-ernie mhc_kernels.py): mixes = (x @ fn^T) / (sqrt(mean(x^2)) + eps).

    That RMS factor 1/(sqrt(mean)+eps) is ERNIE's own; upstream vLLM uses rsqrt(mean+eps).
    stream [T, hc, H] -> post [T, hc], comb [T, hc, hc] (row = input stream), layer_input [T, H].
    """
    T, hc, H = stream.shape
    eps, sk_eps = float(cfg["rms_norm_eps"]), float(cfg["hc_eps"])
    x = stream.reshape(T, hc * H)
    mixes = (x @ fn.T) / (torch.sqrt(x.pow(2).mean(-1, keepdim=True)) + eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + sk_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) * 2.0
    comb = mixes[:, 2 * hc:].reshape(T, hc, hc) * scale[2] + base[2 * hc:].reshape(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + sk_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    for _ in range(int(cfg["hc_sinkhorn_iters"]) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sk_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    layer_input = (pre.unsqueeze(-1) * stream).sum(dim=1)
    return post, comb, layer_input


def mhc_post(x: torch.Tensor, stream: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """vllm kernels/mhc/torch.py mhc_post_torch: out[j] = sum_i comb[i, j] * stream[i] + post[j] * x."""
    return torch.einsum("tij,tih->tjh", comb, stream) + post.unsqueeze(-1) * x.unsqueeze(1)


def hc_head(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Read-out (vllm kernels/mhc/tilelang_kernels.py hc_head): pre = sigmoid(mix * rsqrt(mean+eps) * s0 + b) + eps, out = sum_i pre_i stream_i."""
    T, hc, H = stream.shape
    eps, sk_eps = float(cfg["rms_norm_eps"]), float(cfg["hc_eps"])
    x = stream.reshape(T, hc * H)
    mixes = x @ fn.T
    pre = torch.sigmoid(mixes * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * scale[0] + base) + sk_eps
    return (pre.unsqueeze(-1) * stream).sum(dim=1)


def read_mhc(ck: Checkpoint, key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fn (transposed to [mixes, hc*H] as the GEMM uses it), scale [3], base [mixes] of one hyper-connection module."""
    fn = ck.read(f"{key}.mapping_proj.weight").T
    base = ck.read(f"{key}.bias")
    scale = torch.cat([ck.read(f"{key}.alpha_pre"), ck.read(f"{key}.alpha_post"), ck.read(f"{key}.alpha_res")]).reshape(3)
    return fn, scale, base


# --------------------------------------------------------------------------- attention


def sink_weights(scores: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """Softmax over the keys, with the per-head sink logit in the denominator only.

    scores [heads, keys] (scaled); sink [heads]. Denominator-only is the rule stated in vllm-ernie
    attention.py (Ernie5_5FlashMLASparseImpl docstring). The kernel bodies are not in the sources.
    """
    m = torch.maximum(scores.max(dim=-1).values, sink)
    e = torch.exp(scores - m[:, None])
    return e / (e.sum(dim=-1) + torch.exp(sink - m))[:, None]


def vha(o: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """VHA post-mix (vllm-ernie attention.py VHAPostMix): o[t, h] <- sum_g m[h, g] o[t, g], m = I + V U^T. o [T, heads, dim].

    vLLM folds m with torch.addmm(eye, V, U.t()) in the parameter dtype (fold() in attention.py), which is bf16 for a bf16
    model, so m is rounded to BF16 here. The engine's FoldVha does the same (roundToBf16: true). Without the rounding the
    synthetic layer-0 gap is about 0.5 percent relative; with it the gap is about 1e-5 (see README).
    """
    m = (torch.eye(u.shape[0], dtype=F32) + v @ u.T).to(torch.bfloat16).to(F32)
    return torch.einsum("hg,tgd->thd", m, o)


def hca_attention(ck: Checkpoint, attn: str, qr: torch.Tensor, x: torch.Tensor, positions: torch.Tensor,
                  ratio: int) -> torch.Tensor:
    """HCA layer (vllm-ernie nvidia/attention.py Ernie5_5HCAAttentionMixin on vllm DeepseekV4Attention).

    Covered: q/kv projections and norms, per-head query RMS (inferred, see README), interleaved RoPE at
    compress_rope_theta (YaRN when the config sets it), window keys (inclusive, last sliding_window tokens), the sink, VHA, inverse RoPE on the
    rope slice, the grouped wo_a, the sigmoid gate from qr, and wo_b.
    Not covered: the compressor (vllm DeepseekCompressor, not ported yet; README: Not covered). It refuses to run once a block of
    `ratio` tokens completes, because those keys would be missing.
    """
    cfg = ck.cfg
    T = x.shape[0]
    if T >= ratio:
        raise NotImplementedError(
            f"HCA compressor (compress_ratio {ratio}) is not implemented: a block completes at token {ratio - 1}. "
            "Use fewer tokens than the ratio, or implement the compressor first (README: Not covered)."
        )
    eps = float(cfg["rms_norm_eps"])
    heads, head_dim, rope = int(cfg["num_attention_heads"]), int(cfg["head_dim"]), int(cfg["qk_rope_head_dim"])
    nope, groups, o_lora = head_dim - rope, int(cfg["o_groups"]), int(cfg["o_lora_rank"])
    window = int(cfg["sliding_window"])

    q = (qr @ ck.read(f"{attn}.linear_q_up_proj.weight").T).reshape(T, heads, head_dim)
    if str(cfg.get("qk_norm_type") or "") == "per_head":
        q = rms_unweighted(q, eps)
    kv = rms_norm(x @ ck.read(f"{attn}.linear_kv_proj.weight").T, ck.read(f"{attn}.kv_layernorm.weight"), eps)

    cos, sin = hca_rope_angles(cfg, positions, rope)
    q = torch.cat([q[..., :nope], rope_interleaved(q[..., nope:], cos, sin)], dim=-1)
    kv = torch.cat([kv[:, :nope], rope_interleaved(kv[:, nope:], cos, sin)], dim=-1)  # one KV head: K = V (inferred)

    sink = ck.read(f"{attn}.core_attention.attn_sink")
    scale = head_dim ** -0.5
    o = torch.zeros(T, heads, head_dim, dtype=F32)
    for t in range(T):
        seq = t + 1
        keys = torch.arange(seq - min(seq, window), seq)
        scores = torch.einsum("hd,kd->hk", q[t], kv[keys]) * scale
        o[t] = torch.einsum("hk,kd->hd", sink_weights(scores, sink), kv[keys])

    if cfg.get("use_vha_attention"):
        o = vha(o, ck.read(f"{attn}.vha_postmix_U"), ck.read(f"{attn}.vha_postmix_V"))
    o_rope = rope_interleaved(o[..., nope:], cos, sin, inverse=True)
    o = torch.cat([o[..., :nope], o_rope], dim=-1).reshape(T, heads * head_dim)

    # grouped wo_a: group g takes heads [g*heads/groups, (g+1)*heads/groups), contiguous (inferred, README)
    wo_a = ck.read(f"{attn}.linear_o_group_proj")  # [groups*o_lora, heads*head_dim/groups]
    in_g = heads * head_dim // groups
    z = torch.cat([o[:, g * in_g:(g + 1) * in_g] @ wo_a[g * o_lora:(g + 1) * o_lora].T for g in range(groups)], dim=-1)
    if cfg.get("use_gated_attn"):
        z = z * torch.sigmoid(qr @ ck.read(f"{attn}.gate_proj.weight").T)
    return z @ ck.read(f"{attn}.o_proj.weight").T


def dsa_attention(ck: Checkpoint, attn: str, qr: torch.Tensor, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """DSA layer (vllm-ernie nvidia/attention.py Ernie5_5DSAAttention): non-absorbed MLA, a lightning indexer, a sink.

    The indexer scores the keys before the sliding window (positions < seq - window) and keeps index_topk of them.
    Its ReLU is the standard form sum_h w_h * relu(q_h . k) (README: Not covered, the kernel is not in the sources).
    Query and key Hadamard transforms are skipped: an orthonormal transform on both sides leaves q.k unchanged.
    The indexer's FP8 quantization is skipped, so the scores are F32.
    """
    cfg = ck.cfg
    T = x.shape[0]
    eps = float(cfg["rms_norm_eps"])
    H = int(cfg["hybrid_mla_num_attention_heads"])
    kv_lora = int(cfg["hybrid_mla_kv_lora_rank"])
    nope, rope = int(cfg["hybrid_mla_qk_nope_head_dim"]), int(cfg["hybrid_mla_qk_rope_head_dim"])
    vdim = int(cfg["hybrid_mla_v_head_dim"])
    qk = nope + rope
    window, topk = int(cfg["sliding_window"]), int(cfg["index_topk"])
    cos, sin = rope_angles(positions, rope, float(cfg["rope_theta"]))  # plain RoPE at rope_theta (config.py mla_rope_parameters)

    q = (qr @ ck.read(f"{attn}.q_b_proj.weight").T).reshape(T, H, qk)
    q = torch.cat([q[..., :nope], rope_interleaved(q[..., nope:], cos, sin)], dim=-1)
    kva = x @ ck.read(f"{attn}.kv_a_proj_with_mqa.weight").T
    c = rms_norm(kva[:, :kv_lora], ck.read(f"{attn}.kv_a_layernorm.weight"), eps)
    k_pe = rope_interleaved(kva[:, kv_lora:], cos, sin)
    kvb = (c @ ck.read(f"{attn}.kv_b_proj.weight").T).reshape(T, H, nope + vdim)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]

    idx = f"{attn}.core_attention.indexer"
    n_idx, idx_dim = int(cfg["index_n_heads"]), int(cfg["index_head_dim"])
    iq = (qr @ ck.read(f"{idx}.wq_b.weight").T).reshape(T, n_idx, idx_dim)
    iq = torch.cat([rope_neox(iq[..., :rope], cos, sin), iq[..., rope:]], dim=-1)
    ik = torch.nn.functional.layer_norm(
        x @ ck.read(f"{idx}.wk.weight").T, (idx_dim,),
        weight=ck.read(f"{idx}.k_norm.weight"), bias=ck.read(f"{idx}.k_norm.bias"), eps=1e-6)
    ik = torch.cat([rope_neox(ik[:, :rope], cos, sin), ik[:, rope:]], dim=-1)
    iw = (x @ ck.read(f"{idx}.weights_proj.weight").T) * (idx_dim ** -0.5 * n_idx ** -0.5)
    logits = (torch.relu(torch.einsum("thd,jd->thj", iq, ik)) * iw[:, :, None]).sum(dim=1)  # [T, T]

    sink = ck.read(f"{attn}.core_attention.softmax_offset")
    scale = qk ** -0.5
    o = torch.zeros(T, H, vdim, dtype=F32)
    for t in range(T):
        seq = t + 1
        n_c = max(seq - window, 0)
        if n_c == 0:
            sel = torch.zeros(0, dtype=torch.long)
        elif n_c <= topk:
            sel = torch.arange(n_c)
        else:
            sel = torch.topk(logits[t, :n_c], topk).indices
        keys = torch.cat([sel, torch.arange(seq - min(seq, window), seq)])
        scores = (torch.einsum("hd,khd->hk", q[t, :, :nope], k_nope[keys])
                  + torch.einsum("hr,kr->hk", q[t, :, nope:], k_pe[keys])) * scale
        o[t] = torch.einsum("hk,khd->hd", sink_weights(scores, sink), v[keys])

    if cfg.get("use_vha_attention"):
        o = vha(o, ck.read(f"{attn}.vha_postmix_U"), ck.read(f"{attn}.vha_postmix_V"))
    o = o.reshape(T, H * vdim)
    if cfg.get("use_gated_attn"):
        o = o * torch.sigmoid(qr @ ck.read(f"{attn}.gate_proj.weight").T)
    return o @ ck.read(f"{attn}.o_proj.weight").T


# --------------------------------------------------------------------------- FFN


def moe(ck: Checkpoint, blk: str, h: torch.Tensor) -> torch.Tensor:
    """Latent MoE with split-feature routing and a shared expert (vllm-ernie nvidia/moe.py).

    scores = sigmoid(h W0^T) + sigmoid(h W1^T); top-k of scores + bias, weights = scores (renormalized when
    norm_topk_prob) * routed_scaling_factor_param[expert]. Experts run in the latent space; the latent norm and
    fc2 follow the combine. The shared expert reads h and is added after the routed branch (order from the
    FusedMoE transform naming, not from a line of the source; README: Not covered).
    """
    cfg = ck.cfg
    k = int(cfg["num_experts_per_tok"])
    eps = float(cfg["rms_norm_eps"])
    scores = torch.sigmoid(h @ ck.read(f"{blk}.gate.weight").T)
    if cfg.get("moe_split_feature_routing"):
        scores = scores + torch.sigmoid(h @ ck.read(f"{blk}.gate.weight_1").T)
    bias = ck.read(f"{blk}.e_score_correction_bias")
    sel = torch.topk(scores + bias, k, dim=-1, sorted=False).indices
    w = scores.gather(1, sel)
    if cfg.get("norm_topk_prob", True):
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    w = w * ck.read(f"{blk}.gate.routed_scaling_factor_param")[sel]

    z = h @ ck.read(f"{blk}.fc1_latent_proj.weight").T
    routed = torch.zeros_like(z)
    for e in torch.unique(sel).tolist():  # only the experts the tokens chose are read and dequantized
        tok, slot = (sel == e).nonzero(as_tuple=True)
        ye = situ_mlp(z[tok], ck.read(f"{blk}.experts.{e}.w1.weight"), ck.read(f"{blk}.experts.{e}.w3.weight"),
                      ck.read(f"{blk}.experts.{e}.w2.weight"), cfg)
        routed.index_add_(0, tok, w[tok, slot].unsqueeze(-1) * ye)
    if cfg.get("latent_moe_use_norm", True):
        routed = rms_norm(routed, ck.read(f"{blk}.latent_norm.weight"), eps)
    out = routed @ ck.read(f"{blk}.fc2_latent_proj.weight").T
    if int(cfg.get("n_shared_experts", 0)) > 0:
        out = out + situ_mlp(h, ck.read(f"{blk}.shared_experts.w1.weight"), ck.read(f"{blk}.shared_experts.w3.weight"),
                             ck.read(f"{blk}.shared_experts.w2.weight"), cfg)
    return out


def dense_ffn(ck: Checkpoint, prefix: str, h: torch.Tensor) -> torch.Tensor:
    return situ_mlp(h, ck.read(f"{prefix}.mlp.gate_proj.weight"), ck.read(f"{prefix}.mlp.up_proj.weight"),
                    ck.read(f"{prefix}.mlp.down_proj.weight"), ck.cfg)


# --------------------------------------------------------------------------- decoder


def decoder_layer(ck: Checkpoint, i: int, stream: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    cfg = ck.cfg
    eps = float(cfg["rms_norm_eps"])
    prefix = f"model.layers.{i}"
    ratio = int(cfg["compress_ratios"][i])

    fn, scale, base = read_mhc(ck, f"{prefix}.self_attention_hyper_connection")
    post, comb, x = mhc_pre(stream, fn, scale, base, cfg)
    x = rms_norm(x, ck.read(f"{prefix}.input_layernorm.weight"), eps)
    attn = f"{prefix}.self_attn"
    if ratio <= 0:  # is_dsa_layer: compress_ratios[i] <= 0 (config.py)
        qr = rms_norm(x @ ck.read(f"{attn}.q_a_proj.weight").T, ck.read(f"{attn}.q_a_layernorm.weight"), eps)
        out = dsa_attention(ck, attn, qr, x, positions)
    else:
        qr = rms_norm(x @ ck.read(f"{attn}.linear_q_down_proj.weight").T, ck.read(f"{attn}.q_layernorm.weight"), eps)
        out = hca_attention(ck, attn, qr, x, positions, ratio)
    stream = mhc_post(out, stream, post, comb)

    fn, scale, base = read_mhc(ck, f"{prefix}.mlp_hyper_connection")
    post, comb, x = mhc_pre(stream, fn, scale, base, cfg)
    x = rms_norm(x, ck.read(f"{prefix}.post_attention_layernorm.weight"), eps)
    if i < int(cfg["first_k_dense_replace"]):
        out = dense_ffn(ck, prefix, x)
    else:
        out = moe(ck, f"{prefix}.block_sparse_moe", x)
    return mhc_post(out, stream, post, comb)


def forward(ck: Checkpoint, tokens: list[int], out: Path, max_layers: int | None, want_logits: bool) -> dict:
    cfg = ck.cfg
    hc, H, eps = int(cfg["hc_mult"]), int(cfg["hidden_size"]), float(cfg["rms_norm_eps"])
    T = len(tokens)
    positions = torch.arange(T)
    emb = ck.read(EMBED)
    stream = emb[torch.tensor(tokens)][:, None, :].repeat(1, hc, 1).contiguous()
    del emb

    n_layers = int(cfg["num_hidden_layers"]) if max_layers is None else min(max_layers, int(cfg["num_hidden_layers"]))
    timings = {}
    for i in range(n_layers):
        t0 = time.time()
        stream = decoder_layer(ck, i, stream, positions)
        np.save(out / f"layer_{i:02d}_all.npy", stream.numpy().astype(np.float32))
        np.save(out / f"layer_{i:02d}_last.npy", stream[-1].numpy().astype(np.float32))
        timings[f"layer_{i:02d}_s"] = round(time.time() - t0, 3)
        print(f"layer {i:2d} done in {timings[f'layer_{i:02d}_s']:.2f} s", flush=True)

    head_fn = ck.read(HEAD_FN).T
    hidden = rms_norm(hc_head(stream, head_fn, ck.read(HEAD_SCALE).reshape(1), ck.read(HEAD_BASE), cfg),
                      ck.read(FINAL_NORM), eps)
    np.save(out / "final_hidden.npy", hidden.numpy().astype(np.float32))
    if want_logits and n_layers == int(cfg["num_hidden_layers"]):
        logits = hidden[-1] @ ck.read(LM_HEAD).T
        if ck.has(MULTIMAX_RANGES):
            r, g = ck.read(MULTIMAX_RANGES), ck.read(MULTIMAX_GAINS)
            logits = (logits + g[0] * torch.relu(r[0] - logits) + g[1] * torch.relu(logits - r[1])
                      + g[2] * torch.relu(r[2] - logits).pow(2) + g[3] * torch.relu(logits - r[3]).pow(2))
        np.save(out / "logits_last.npy", logits.numpy().astype(np.float32))
    return timings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint directory (config.json, index, shards)")
    p.add_argument("--tokens", required=True, help="comma-separated token ids")
    p.add_argument("--out", required=True, help="output directory (created)")
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--logits", action="store_true", help="also write the last token's logits (full LM head, about 3.3 GB F32)")
    args = p.parse_args()

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    ck = Checkpoint(args.ckpt)
    if any(t < 0 or t >= int(ck.cfg["vocab_size"]) for t in tokens):
        raise SystemExit("a token id is outside the vocabulary")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings = forward(ck, tokens, out, args.max_layers, args.logits)
    meta = {"tokens": tokens, "checkpoint": str(args.ckpt), "total_s": round(time.time() - t0, 3), **timings}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} in {meta['total_s']} s")


if __name__ == "__main__":
    main()
