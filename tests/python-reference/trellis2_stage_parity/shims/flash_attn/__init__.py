"""Naive CPU stand-in for flash_attn's varlen entry points: per sequence, softmax(q k^T * scale) v in float32."""
import torch


def _attend(q, k, v):
    qh, kh, vh = (t.permute(1, 0, 2).float() for t in (q, k, v))
    scores = (qh @ kh.transpose(1, 2)) * (q.shape[-1] ** -0.5)
    return (torch.softmax(scores, dim=-1) @ vh).permute(1, 0, 2)


def _run(q, k, v, cu_q, cu_kv):
    outs = [_attend(q[cu_q[i]:cu_q[i + 1]], k[cu_kv[i]:cu_kv[i + 1]], v[cu_kv[i]:cu_kv[i + 1]]) for i in range(len(cu_q) - 1)]
    return torch.cat(outs, 0).to(q.dtype)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen):
    c = cu_seqlens.tolist()
    return _run(qkv[:, 0], qkv[:, 1], qkv[:, 2], c, c)


def flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_kv, max_q, max_kv):
    return _run(q, kv[:, 0], kv[:, 1], cu_q.tolist(), cu_kv.tolist())


def flash_attn_varlen_func(q, k, v, cu_q, cu_kv, max_q, max_kv):
    return _run(q, k, v, cu_q.tolist(), cu_kv.tolist())
