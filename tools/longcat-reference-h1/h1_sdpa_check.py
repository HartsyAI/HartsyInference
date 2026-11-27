"""CPU check of the H1 SDPA substitutes, through the patched call sites.

Run: python h1_sdpa_check.py <root of a reference checkout with h1-sdpa.patch and h1-sdpa-cross.patch applied>

What it covers:
- The avatar DiT's text cross-attention, `MultiHeadCrossAttention._process_cross_attn` (base attention.py), run twice on the same
  weights and inputs: once with H1_ATTN unset, where the flash branch calls a fake `flash_attn_varlen_func` that implements the
  flash contract from the cu_seqlens arguments that the ORIGINAL call builds (cumsum of [0] + lengths), and once with
  H1_ATTN=sdpa, where the patched call site runs. The outputs must match. Batch sizes 1 and 3, uneven key lengths, and a
  length-1 key segment are tested.
- `_h1_varlen_sdpa` against a naive float64 attention, and a no-leakage check across samples.
- The helper's assertions: mismatched lengths fail.

What it does not cover: flash-attn itself (not installed, not used). The fake is the flash contract as written here, not the
flash library. The triton import is stubbed so the package imports on a CPU machine; the stub is never used by the substitutes.
"""

import math
import os
import sys
import types

import torch
import torch.nn.functional as F


class _Dummy:
    """Stands in for any triton name at import time: a decorator returns the function it wraps; other uses return another dummy."""

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _Dummy()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()

    def __getitem__(self, key):
        return _Dummy()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()


def install_stubs(root):
    sys.path.insert(0, root)
    for name in ["triton", "triton.language"]:
        sys.modules[name] = _StubModule(name)
    # The block-sparse path is off in the avatar config (enable_bsa: false) and is not used here. Stub its package and module so
    # attention.py imports without the Triton kernels; the real files are used on the server.
    pkg = _StubModule("longcat_video.block_sparse_attention")
    pkg.__path__ = []
    sys.modules["longcat_video.block_sparse_attention"] = pkg
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = _StubModule("longcat_video.block_sparse_attention.bsa_interface")


def naive_segments(q, k, v, cu_q, cu_k):
    """Per-segment softmax(q k^T / sqrt(D)) v in float64, written out by hand."""
    out = torch.zeros(q.shape, dtype=torch.float64)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for b in range(len(cu_q) - 1):
        qs, qe, ks, ke = cu_q[b], cu_q[b + 1], cu_k[b], cu_k[b + 1]
        for h in range(q.shape[1]):
            scores = (q[qs:qe, h].double() @ k[ks:ke, h].double().T) * scale
            out[qs:qe, h] = torch.softmax(scores, dim=-1) @ v[ks:ke, h].double()
    return out


def install_fake_flash():
    """The flash contract: query segment b attends only to key segment b; cu_seqlens are cumulative with a leading 0."""
    fake = types.ModuleType("flash_attn")

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        cu_q = [int(x) for x in cu_seqlens_q.tolist()]
        cu_k = [int(x) for x in cu_seqlens_k.tolist()]
        return naive_segments(q, k, v, cu_q, cu_k).to(q.dtype)

    fake.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = fake


def check_cross_attention(attention_module):
    MultiHeadCrossAttention = attention_module.MultiHeadCrossAttention
    torch.manual_seed(0)
    dim, heads = 32, 4
    layer = MultiHeadCrossAttention(dim, heads, enable_flashattn2=True).double()
    cases = [
        ("B=1, N=5, kv [7]", 5, [7]),
        ("B=3, N=5, kv [7, 3, 12]", 5, [7, 3, 12]),
        ("B=3, N=4, kv [1, 1, 1]", 4, [1, 1, 1]),
    ]
    for label, n, kv_seqlen in cases:
        B = len(kv_seqlen)
        x = torch.randn(B, n, dim, dtype=torch.float64)
        cond = torch.randn(1, sum(kv_seqlen), dim, dtype=torch.float64)
        os.environ.pop("H1_ATTN", None)
        with torch.no_grad():
            reference = layer._process_cross_attn(x, cond, kv_seqlen)
        os.environ["H1_ATTN"] = "sdpa"
        try:
            with torch.no_grad():
                patched = layer._process_cross_attn(x, cond, kv_seqlen)
        finally:
            os.environ.pop("H1_ATTN", None)
        err = (patched - reference).abs().max().item()
        print(f"[cross-attention] {label}: max abs difference, patched call site vs flash contract = {err:.3e}")
        assert err < 1e-9, (label, err)


def check_helper(helper):
    torch.manual_seed(1)
    H, D = 4, 8
    q = torch.randn(15, H, D)
    k = torch.randn(22, H, D)
    v = torch.randn(22, H, D)
    q_lens, k_lens = [5, 5, 5], [7, 3, 12]
    out = helper(q, k, v, q_lens, k_lens)
    cu_q = [0, 5, 10, 15]
    cu_k = [0, 7, 10, 22]
    err = (out.double() - naive_segments(q, k, v, cu_q, cu_k)).abs().max().item()
    print(f"[helper] per-sample lengths {q_lens} / {k_lens}: max abs error vs naive float64 = {err:.3e}")
    assert err < 1e-5, err

    # no leakage: changing sample 1's keys and values leaves samples 0 and 2 unchanged
    k2, v2 = k.clone(), v.clone()
    k2[7:10] = torch.randn_like(k2[7:10]) * 10
    v2[7:10] = torch.randn_like(v2[7:10]) * 10
    moved = helper(q, k2, v2, q_lens, k_lens)
    leak = max((out[0:5] - moved[0:5]).abs().max().item(), (out[10:15] - moved[10:15]).abs().max().item())
    print(f"[helper] changing sample 1's keys: change in samples 0 and 2 = {leak:.3e} (must be 0)")
    assert leak == 0.0, leak

    # mismatched lengths must fail loudly, not silently cut the tensors
    for bad_q, bad_k, label in [([5, 5, 4], k_lens, "query lengths short by one token"), (q_lens, [7, 3, 11], "key lengths short by one token")]:
        try:
            helper(q, k, v, bad_q, bad_k)
        except AssertionError as exc:
            print(f"[helper] {label}: refused as expected ({exc})")
        else:
            raise AssertionError(f"{label}: no assertion")


def main(root):
    install_stubs(root)
    install_fake_flash()
    import ast
    helper_src = open(os.path.join(root, "longcat_video", "modules", "attention.py"), encoding="utf-8").read()
    tree = ast.parse(helper_src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_h1_varlen_sdpa")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "_h1_varlen_sdpa", "exec"), namespace)
    helper = namespace["_h1_varlen_sdpa"]

    from longcat_video.modules import attention as attention_module
    check_helper(helper)
    check_cross_attention(attention_module)
    print("PASS: the patched call sites match the flash contract for batch sizes 1 and 3; the helper refuses mismatched lengths.")


if __name__ == "__main__":
    main(sys.argv[1])
