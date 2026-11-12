"""CPU stand-in for FlexGEMM's submanifold sparse conv: the neighbour map by sorted-key search, then one gather GEMM per kernel offset.

The kernel index is row-major (kd, kh, kw) over the weight layout (Co, Kd, Kh, Kw, Ci), with kd moving along the first spatial coordinate
column (x), kh along y and kw along z. That ordering is the one the C# engine assumes too (`SparseOps`), so this stand-in checks the
arithmetic of the decoders, not that axis convention: FlexGEMM's own source was not available verbatim when this was written.
"""
import torch


def set_algorithm(*args, **kwargs):
    pass


def set_hashmap_ratio(*args, **kwargs):
    pass


def _keys(coords, stride):
    c = coords.to(torch.int64)
    return ((c[:, 0] * stride[0] + c[:, 1]) * stride[1] + c[:, 2]) * stride[2] + c[:, 3]


def _neighbor_map(coords, ksize, dilation):
    n = coords.shape[0]
    lo = coords[:, 1:].min().item() - 8 * max(dilation)
    hi = coords[:, 1:].max().item() + 8 * max(dilation)
    shifted = coords.to(torch.int64).clone()
    shifted[:, 1:] -= lo
    stride = (hi - lo + 1,) * 3
    keys = _keys(shifted, stride)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    kd, kh, kw = ksize
    nm = torch.full((n, kd * kh * kw), -1, dtype=torch.int64)
    v = 0
    for a in range(kd):
        for b in range(kh):
            for c in range(kw):
                q = shifted.clone()
                q[:, 1] += (a - kd // 2) * dilation[0]
                q[:, 2] += (b - kh // 2) * dilation[1]
                q[:, 3] += (c - kw // 2) * dilation[2]
                qk = _keys(q, stride)
                pos = torch.searchsorted(sorted_keys, qk).clamp(max=n - 1)
                hit = sorted_keys[pos] == qk
                nm[:, v] = torch.where(hit, order[pos], torch.full_like(pos, -1))
                v += 1
    return nm


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias=None, neighbor_cache=None, dilation=(1, 1, 1)):
    co, kd, kh, kw, ci = weight.shape
    if neighbor_cache is None:
        neighbor_cache = _neighbor_map(coords, (kd, kh, kw), tuple(dilation))
    w = weight.reshape(co, kd * kh * kw, ci)
    out = torch.zeros(feats.shape[0], co, dtype=feats.dtype)
    for v in range(kd * kh * kw):
        idx = neighbor_cache[:, v]
        ok = idx >= 0
        if ok.any():
            out[ok] += feats[idx[ok]] @ w[:, v, :].t()
    if bias is not None:
        out = out + bias
    return out, neighbor_cache
