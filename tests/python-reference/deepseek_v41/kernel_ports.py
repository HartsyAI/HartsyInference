"""Pure-torch ports of the tilelang kernels in DeepSeek-V4.1-Flash `inference/kernel.py`.

These exist so the unmodified upstream `model.py` can run on CPU for small-config reference dumps.
Agreement between these ports and the tilelang originals is only checkable on Hopper/Blackwell
hardware and is recorded in the parity ledger from the rented campaign, not from here.
CPU only: helpers allocate on the default device.
"""
from typing import Optional, Tuple

import torch

FP8_MAX: float = 448.0
FP4_MAX: float = 6.0
E2M1_MAGNITUDES: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_TABLE: Tuple[float, ...] = E2M1_MAGNITUDES + tuple(-m for m in E2M1_MAGNITUDES)

_F32_FP8_MAX_INV: torch.Tensor = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
_F32_FP4_MAX_INV: torch.Tensor = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32)


def round_scale_pow2(amax: torch.Tensor, max_inv: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(amax * max_inv)) via the IEEE-754 exponent, as kernel.fast_round_scale does."""
    scaled: torch.Tensor = (amax.to(torch.float32) * max_inv).contiguous()
    bits: torch.Tensor = scaled.view(torch.int32)
    exp: torch.Tensor = (bits >> 23) & 0xFF
    man: torch.Tensor = bits & ((1 << 23) - 1)
    log2_ceil: torch.Tensor = exp - 127 + (man != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """fp32 values (already clamped to +-6) to E2M1 codes 0..15, round-to-nearest-even on the code."""
    mag: torch.Tensor = x.abs().to(torch.float32)
    table: torch.Tensor = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=x.device)
    # midpoints between adjacent magnitudes; ties go to the even code
    mids: torch.Tensor = (table[1:] + table[:-1]) * 0.5
    code: torch.Tensor = (mag.unsqueeze(-1) > mids).sum(-1)
    tie: torch.Tensor = (mag.unsqueeze(-1) == mids)
    tie_idx: torch.Tensor = tie.to(torch.int64).argmax(-1)
    is_tie: torch.Tensor = tie.any(-1)
    # a tie at mids[i] sits between codes i and i+1; pick the even one
    even_code: torch.Tensor = torch.where(tie_idx % 2 == 0, tie_idx, tie_idx + 1)
    code = torch.where(is_tie, even_code, code)
    sign: torch.Tensor = ((x < 0) | ((x == 0) & torch.signbit(x))).to(torch.int64) << 3
    return (code | sign).to(torch.uint8)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    table: torch.Tensor = torch.tensor(FP4_TABLE, dtype=torch.float32, device=codes.device)
    return table[codes.to(torch.int64)]


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Two codes per byte along the last dim, even element in the low nibble."""
    lo: torch.Tensor = codes[..., 0::2]
    hi: torch.Tensor = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Packed bytes [.., K/2] to fp32 values [.., K]."""
    b: torch.Tensor = packed.view(torch.uint8) if packed.dtype != torch.uint8 else packed
    lo: torch.Tensor = e2m1_decode(b & 0x0F)
    hi: torch.Tensor = e2m1_decode((b >> 4) & 0x0F)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8 e4m3 quantization along the last dim; inplace writes the dequantized values back."""
    n: int = x.size(-1)
    assert n % block_size == 0
    xf: torch.Tensor = x.to(torch.float32).reshape(-1, n // block_size, block_size)
    amax: torch.Tensor = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    if scale_fmt is not None:
        s: torch.Tensor = round_scale_pow2(amax, _F32_FP8_MAX_INV)
    else:
        s = amax * _F32_FP8_MAX_INV
    q: torch.Tensor = (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.to(torch.float32) * s).reshape(x.shape).to(x.dtype))
        return x
    scales: torch.Tensor = s.squeeze(-1).reshape(*x.shape[:-1], n // block_size).to(scale_dtype)
    return q.reshape(x.shape), scales


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    """E2M1 quantization with E8M0 (indexer) or E4M3 (compressed KV) scales along the last dim."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n: int = x.size(-1)
    assert n % block_size == 0
    xf: torch.Tensor = x.to(torch.float32).reshape(-1, n // block_size, block_size)
    amax: torch.Tensor = xf.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s: torch.Tensor = (amax / FP4_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = round_scale_pow2(amax, _F32_FP4_MAX_INV)
    codes: torch.Tensor = e2m1_encode((xf / s).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((e2m1_decode(codes) * s).reshape(x.shape).to(x.dtype))
        return x
    packed: torch.Tensor = pack_e2m1(codes.reshape(*x.shape[:-1], n))
    scales: torch.Tensor = s.squeeze(-1).reshape(*x.shape[:-1], n // block_size).to(scale_dtype)
    return packed, scales


def _expand_block_scales(scales: torch.Tensor, rows: int, cols: int, block_rows: int, block_cols: int) -> torch.Tensor:
    s: torch.Tensor = scales.to(torch.float32)
    s = s.repeat_interleave(block_rows, dim=0)[:rows]
    return s.repeat_interleave(block_cols, dim=1)[:, :cols]


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block_rows: int, block_cols: int) -> torch.Tensor:
    """FP8 weight [N,K] with a [ceil(N/br), ceil(K/bc)] scale grid to fp32 (scale is a multiplier)."""
    n, k = weight.shape
    return weight.to(torch.float32) * _expand_block_scales(scale, n, k, block_rows, block_cols)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 weight [N,K/2] with an e8m0 scale [N,K/32] to fp32 [N,K]."""
    w: torch.Tensor = unpack_e2m1(packed)
    return w * scale.to(torch.float32).repeat_interleave(32, dim=-1)[..., : w.size(-1)]


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """C = A[M,K] @ B[N,K]^T with per-row/K-block activation scales and (block,block) weight scales.

    One block_size serves both operands, as in upstream (`fp8_block_size` = 32 for activations and weights).
    """
    k: int = a.size(-1)
    m: int = a.numel() // k
    n: int = b.size(0)
    assert block_size in (32, 128) and k % block_size == 0
    assert a_s.numel() == m * (k // block_size)
    assert b_s.shape == ((n + block_size - 1) // block_size, k // block_size)
    a_f: torch.Tensor = a.reshape(m, k).to(torch.float32) * a_s.reshape(m, -1).to(torch.float32).repeat_interleave(
        block_size, dim=1
    )
    b_f: torch.Tensor = dequant_fp8_block(b, b_s, block_size, block_size)
    c: torch.Tensor = a_f @ b_f.T
    return c.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """C = A_fp8[M,K] @ B_fp4[N,K]^T with B packed [N,K/2] and an e8m0 scale [N,K/32]."""
    k: int = a.size(-1)
    m: int = a.numel() // k
    n: int = b.size(0)
    assert act_block_size in (32, 128) and k % act_block_size == 0
    assert a_s.numel() == m * (k // act_block_size)
    assert b_s.shape == (n, k // 32)
    a_f: torch.Tensor = a.reshape(m, k).to(torch.float32) * a_s.reshape(m, -1).to(torch.float32).repeat_interleave(
        act_block_size, dim=1
    )
    c: torch.Tensor = a_f @ dequant_mxfp4(b, b_s).T
    return c.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Blockwise online-softmax port: 64-index blocks, running max seeded at -1e30, bf16 probabilities
    for the value GEMM, sink added to the denominator, rows with no valid index give zeros."""
    b, m, h, d = q.shape
    topk: int = topk_idxs.size(-1)
    block: int = 64
    qf: torch.Tensor = q.to(torch.float32)
    kvf: torch.Tensor = kv.to(torch.float32)
    sink: torch.Tensor = attn_sink.to(torch.float32).view(1, 1, h)
    acc_o: torch.Tensor = torch.zeros(b, m, h, d, dtype=torch.float32)
    sum_exp: torch.Tensor = torch.zeros(b, m, h, dtype=torch.float32)
    scores_max: torch.Tensor = torch.full((b, m, h), -1e30, dtype=torch.float32)
    batch: torch.Tensor = torch.arange(b).view(b, 1, 1)
    for start in range(0, topk, block):
        idx: torch.Tensor = topk_idxs[..., start : start + block].to(torch.int64)
        valid: torch.Tensor = idx != -1
        gathered: torch.Tensor = kvf[batch, idx.clamp_min(0)] * valid.unsqueeze(-1)
        s: torch.Tensor = torch.einsum("bmhd,bmkd->bmhk", qf, gathered) * softmax_scale
        s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
        prev: torch.Tensor = scores_max
        scores_max = torch.maximum(prev, s.amax(-1))
        scale: torch.Tensor = torch.exp(prev - scores_max)
        p: torch.Tensor = torch.exp(s - scores_max.unsqueeze(-1))
        sum_exp = sum_exp * scale + p.sum(-1)
        acc_o = acc_o * scale.unsqueeze(-1) + torch.einsum(
            "bmhk,bmkd->bmhd", p.to(torch.bfloat16).to(torch.float32), gathered
        )
    sum_exp = sum_exp + torch.exp(sink - scores_max)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


def sparse_attn_exact(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """float64 one-shot softmax over the valid indices plus the sink; the accuracy reference for the port."""
    b, m, h, d = q.shape
    idx: torch.Tensor = topk_idxs.to(torch.int64)
    valid: torch.Tensor = idx != -1
    batch: torch.Tensor = torch.arange(b).view(b, 1, 1)
    gathered: torch.Tensor = kv.to(torch.float64)[batch, idx.clamp_min(0)] * valid.unsqueeze(-1)
    s: torch.Tensor = torch.einsum("bmhd,bmkd->bmhk", q.to(torch.float64), gathered) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink: torch.Tensor = attn_sink.to(torch.float64).view(1, 1, h, 1).expand(b, m, h, 1)
    logits: torch.Tensor = torch.cat([s, sink], dim=-1)
    weights: torch.Tensor = torch.softmax(logits, dim=-1)[..., :-1]
    weights = torch.nan_to_num(weights)
    return torch.einsum("bmhk,bmkd->bmhd", weights, gathered).to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split [.., (2+hc)*hc] mixes into pre[hc], post[hc] and a Sinkhorn-normalised comb[hc,hc]."""
    lead: Tuple[int, ...] = tuple(mixes.shape[:-1])
    hc: int = hc_mult
    x: torch.Tensor = mixes.to(torch.float32).reshape(-1, (2 + hc) * hc)
    scale: torch.Tensor = hc_scale.to(torch.float32)
    base: torch.Tensor = hc_base.to(torch.float32)
    pre: torch.Tensor = torch.sigmoid(x[:, :hc] * scale[0] + base[:hc]) + eps
    post: torch.Tensor = 2 * torch.sigmoid(x[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb: torch.Tensor = (x[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre.reshape(*lead, hc), post.reshape(*lead, hc), comb.reshape(*lead, hc, hc)
