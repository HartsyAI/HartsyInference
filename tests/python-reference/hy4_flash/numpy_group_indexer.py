"""NumPy transcription of the Hy4 Flash compressed indexer (HYV4FlashCompressedIndexer, transformers PR 50903 at eb1d5db), float32, no torch.

This is a TRANSCRIPTION of the reference maths, not the Hugging Face class: it shares any misreading the author made. It writes
tests/HartsyInference.LLM.Tests/Fixtures/Hy4Flash/group_indexer_case.json, which Hy4FlashGroupIndexerTests compares the C# indexer against.
Replace it with the HF dump (dump_tiny_reference.py, indexer_*_keep) once that has run.

    python -I numpy_group_indexer.py <output.json>
"""

import json
import sys

import numpy as np

DIM, QR, NH, HD, ROPE, KPOOL, TOPK, THETA, EPS = 24, 16, 2, 8, 4, 4, 16, 1.0e7, 1e-5
LENGTH = 37


def rms(x, w):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(EPS)) * w).astype(np.float32)


def rotary(x, positions):
    """Interleaved-pair rotation of the trailing ROPE channels; x is [n, ..., HD], positions [n]."""
    half = ROPE // 2
    inverse = (THETA ** (-np.arange(half, dtype=np.float64) * 2.0 / ROPE))
    angle = positions[:, None].astype(np.float64) * inverse[None, :]
    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    c, s = c.reshape(shape), s.reshape(shape)
    out = x.copy()
    tail = out[..., HD - ROPE:]
    even, odd = tail[..., 0::2].copy(), tail[..., 1::2].copy()
    tail[..., 0::2] = even * c - odd * s
    tail[..., 1::2] = odd * c + even * s
    return out


def main():
    rng = np.random.default_rng(7)

    def f(*shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w = {
        "query_projection": f(NH * HD, QR), "head_weight_projection": f(NH, DIM), "token_key_projection": f(HD, DIM),
        "token_key_norm": (1 + f(HD, scale=0.3)), "key_value_projection": f(HD, HD), "key_gate_projection": f(HD, HD),
        "intra_group_ape": f(KPOOL, HD), "key_norm": (1 + f(HD, scale=0.3)),
    }
    x = f(LENGTH, DIM, scale=1.0)
    q_latent = f(LENGTH, QR, scale=1.0)

    token_keys = rms(x @ w["token_key_projection"].T, w["token_key_norm"])

    def pool(keys):
        groups = keys.shape[0] // KPOOL
        grouped = keys[: groups * KPOOL].reshape(groups, KPOOL, HD)
        values = grouped @ w["key_value_projection"].T
        gates = grouped @ w["key_gate_projection"].T + w["intra_group_ape"]
        gates = gates - gates.max(axis=1, keepdims=True)
        soft = np.exp(gates)
        soft = soft / soft.sum(axis=1, keepdims=True)
        return rms((values * soft).sum(axis=1).astype(np.float32), w["key_norm"])

    def select(length):
        if length <= TOPK:
            return [list(range(t + 1)) for t in range(length)], None
        groups = pool(token_keys[:length])
        n_groups = groups.shape[0]
        group_keys = rotary(groups, np.arange(n_groups) * KPOOL)
        query = rotary((q_latent[:length] @ w["query_projection"].T).reshape(length, NH, HD), np.arange(length))
        head_weights = (x[:length] @ w["head_weight_projection"].T) * np.float32(NH ** -0.5 * HD ** -0.5)
        scores = np.maximum(np.einsum("thd,gd->thg", query, group_keys), 0)
        logits = np.einsum("th,thg->tg", head_weights, scores)
        keep = []
        for t in range(length):
            own = t // KPOOL
            row = logits[t].copy()
            row[np.arange(n_groups) >= own] = -np.inf
            complete = min(TOPK // KPOOL - 1, n_groups)
            order = sorted(range(n_groups), key=lambda g: (-row[g], g))[:complete]
            chosen = set(order) | {own}
            tokens = sorted(g * KPOOL + i for g in chosen for i in range(KPOOL) if g * KPOOL + i <= t and g * KPOOL + i < length)
            keep.append(tokens)
        return keep, group_keys

    keep_long, group_keys = select(LENGTH)
    keep_dense, _ = select(12)
    case = {
        "dim": DIM, "q_lora_rank": QR, "heads": NH, "head_dim": HD, "rope_dim": ROPE, "kpool": KPOOL, "topk": TOPK, "theta": THETA,
        "norm_eps": EPS, "length": LENGTH,
        "weights": {k: v.reshape(-1).tolist() for k, v in w.items()},
        "x": x.reshape(-1).tolist(), "q_latent": q_latent.reshape(-1).tolist(),
        "token_keys": token_keys.reshape(-1).tolist(), "group_keys": group_keys.reshape(-1).tolist(),
        "keep_long": keep_long, "keep_dense": keep_dense,
    }
    with open(sys.argv[1], "w") as handle:
        json.dump(case, handle, separators=(",", ":"))
    selected = sum(len(r) for r in keep_long)
    dense = sum(t + 1 for t in range(LENGTH))
    print(f"wrote {sys.argv[1]}; the indexer keeps {selected} of {dense} causal keys over {LENGTH} queries")


if __name__ == "__main__":
    main()
