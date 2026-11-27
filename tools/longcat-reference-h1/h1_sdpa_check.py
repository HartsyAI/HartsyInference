"""CPU check of the H1 varlen cross-attention substitute (`_h1_varlen_sdpa` in the patched base attention.py).

What it covers, and what it does not:
- Covers: the value of `_h1_varlen_sdpa` against a naive per-segment softmax attention (float64), for uneven key lengths
  per sample, including a key length of 1; and that a query segment's output does not depend on any other sample's keys
  or values (no leakage across the varlen segments).
- Does not cover: flash-attn itself. flash-attn is not installed and is not used; this checks the semantics the reference
  asks of flash_attn_varlen_func (no causal mask, default scale 1/sqrt(D), segment b of queries attends only to segment b of keys).

Run: python h1_sdpa_check.py <path to the patched longcat_video/modules/attention.py>
"""

import ast
import math
import sys

import torch
import torch.nn.functional as F


def load_helper(path):
    """Compile only `_h1_varlen_sdpa` from the patched file, so the check does not need the package's other imports."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_h1_varlen_sdpa")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, path, "exec"), namespace)
    return namespace["_h1_varlen_sdpa"]


def naive(q, k, v, cu_q, cu_k):
    """Per-segment softmax(q k^T / sqrt(D)) v, in float64, written out by hand."""
    out = torch.zeros(q.shape, dtype=torch.float64)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for b in range(len(cu_q) - 1):
        qs, qe, ks, ke = cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1]
        for h in range(q.shape[1]):
            scores = (q[qs:qe, h].double() @ k[ks:ke, h].double().T) * scale
            weights = torch.softmax(scores, dim=-1)
            out[qs:qe, h] = weights @ v[ks:ke, h].double()
    return out


def main(path):
    helper = load_helper(path)
    torch.manual_seed(0)
    H, D = 4, 8
    n_queries = 5
    kv_lengths = [7, 3, 12]            # uneven per sample; the reference's text cross-attention has per-sample lengths too
    B = len(kv_lengths)
    cu_q = [0] + [n_queries] * B
    cu_q = [sum(cu_q[: i + 1]) for i in range(B + 1)]
    cu_k = [0]
    for length in kv_lengths:
        cu_k.append(cu_k[-1] + length)

    q = torch.randn(cu_q[-1], H, D)
    k = torch.randn(cu_k[-1], H, D)
    v = torch.randn(cu_k[-1], H, D)

    got = helper(q, k, v, cu_q, cu_k).double()
    want = naive(q, k, v, cu_q, cu_k)
    err = (got - want).abs().max().item()
    print(f"[1] uneven key lengths {kv_lengths}: max abs error vs naive float64 = {err:.3e}")
    assert err < 1e-5, err

    # key length 1 in the middle sample
    kv1 = [4, 1, 6]
    cu_k1 = [0]
    for length in kv1:
        cu_k1.append(cu_k1[-1] + length)
    k1 = torch.randn(cu_k1[-1], H, D)
    v1 = torch.randn(cu_k1[-1], H, D)
    got1 = helper(q, k1, v1, cu_q, cu_k1).double()
    want1 = naive(q, k1, v1, cu_q, cu_k1)
    err1 = (got1 - want1).abs().max().item()
    print(f"[2] key lengths {kv1} (one of length 1): max abs error = {err1:.3e}")
    assert err1 < 1e-5, err1

    # leakage: change sample 1's keys and values only; sample 0 and sample 2 query outputs must not move
    k2 = k.clone()
    v2 = v.clone()
    seg1 = slice(cu_k[1], cu_k[2])
    k2[seg1] = torch.randn_like(k2[seg1]) * 10
    v2[seg1] = torch.randn_like(v2[seg1]) * 10
    base = helper(q, k, v, cu_q, cu_k)
    moved = helper(q, k2, v2, cu_q, cu_k)
    q0 = slice(cu_q[0], cu_q[1])
    q2 = slice(cu_q[2], cu_q[3])
    q1 = slice(cu_q[1], cu_q[2])
    leak = max((base[q0] - moved[q0]).abs().max().item(), (base[q2] - moved[q2]).abs().max().item())
    changed = (base[q1] - moved[q1]).abs().max().item()
    print(f"[3] changing sample 1's keys/values: max change in samples 0 and 2 = {leak:.3e} (must be 0); in sample 1 = {changed:.3e}")
    assert leak == 0.0, leak
    assert changed > 0.0

    print("PASS: the substitute matches the naive varlen semantics and does not leak across samples.")


if __name__ == "__main__":
    main(sys.argv[1])
