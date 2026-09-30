"""Dump Engram hash-id cases from the UNMODIFIED upstream engram.py NgramHashState.forward.

The compressed token map is the pinned constants fixture (token_map.i32.bin, produced by build_compressed_token_map in
dump_engram_constants.py and cross-checked there against the GGUF and MLX copies); it is substituted for the tokenizer
walk so the case dump does not need the tokenizer. Everything else (multipliers, primes, offsets, DEAD/blocked handling,
history cache across start_pos) runs upstream code. Cases are batch 1: each step is one forward call, so a case with
several steps exercises the prefill/decode cache carry.

Usage: python dump_engram_hash_fixtures.py [--upstream ~/dsv41-ref/upstream] [--constants <dir>] [--out fixtures/engram_hash_ids.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONSTANTS = os.path.join(HERE, "..", "..", "..", "src", "HartsyInference.LLM", "DeepSeekV41", "Engram", "Constants")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--constants", default=DEFAULT_CONSTANTS)
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "engram_hash_ids.json"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import engram

    with open(os.path.join(a.upstream, "inference", "config.json")) as f:
        cfg = json.load(f)
    token_map = np.fromfile(os.path.join(a.constants, "token_map.i32.bin"), dtype="<i4").tolist()
    manifest = json.load(open(os.path.join(a.constants, "manifest.json")))
    vocab = manifest["compressed_vocab"]
    engram.build_compressed_token_map = lambda tokenizer: (token_map, vocab)

    class Args:
        engram_layer_ids = tuple(cfg["engram_layer_ids"])
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]
        engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
        max_batch_size = 1
        max_seq_len = 256

    layout = engram.EngramLayout.from_args(Args)

    rng = np.random.default_rng(20260930)
    by_compressed: dict = {}
    for tid, cid in enumerate(token_map):
        by_compressed.setdefault(cid, []).append(tid)
    collapsing = [v[:3] for v in by_compressed.values() if len(v) >= 3][:6]
    assert collapsing, "no compressed ids with >=3 raw tokens"
    pad_token = Args.engram_pad_id
    tokenizer_vocab = manifest["tokenizer_vocab"]

    def rand_ids(n: int) -> list:
        return [int(x) for x in rng.integers(0, tokenizer_vocab, size=n)]

    cases = []

    def add(name: str, steps: list) -> None:
        state = engram.NgramHashState(Args, layout, None)
        out_steps = []
        for start, ids, mask in steps:
            t = torch.tensor([ids], dtype=torch.int64)
            m = None if mask is None else torch.tensor([mask], dtype=torch.bool)
            h = state(t, start, m)  # [1, L, layers, 24]
            assert h.shape == (1, len(ids), len(Args.engram_layer_ids), 24), h.shape
            out_steps.append({"startPos": start, "ids": ids, "mask": None if mask is None else [int(x) for x in mask],
                              "hash": [int(v) for v in h.reshape(-1)]})
        cases.append({"name": name, "steps": out_steps})

    add("prefill_plain", [(0, rand_ids(12), None)])
    add("single_token_prefill", [(0, rand_ids(1), None)])
    add("two_token_prefill", [(0, rand_ids(2), None)])
    add("vocab_extremes", [(0, [0, tokenizer_vocab - 1, 0, tokenizer_vocab - 1, 1, 2, 3], None)])
    flat_collapse = [t for group in collapsing[:3] for t in group]
    add("case_variants_collapse", [(0, flat_collapse, None)])
    add("pad_token_inline", [(0, [pad_token, 500, pad_token, pad_token, 77, pad_token], None)])
    add("dead_middle_span", [(0, rand_ids(10), [1, 1, 1, 0, 0, 0, 1, 1, 1, 1])])
    add("dead_first_and_last", [(0, rand_ids(8), [0, 1, 1, 1, 1, 1, 1, 0])])
    add("all_dead", [(0, rand_ids(5), [0, 0, 0, 0, 0])])
    ids = rand_ids(8)
    add("prefill_then_decode_1", [(0, ids[:6], None)] + [(6 + i, [ids[6 + i]], None) for i in range(2)])
    ids = rand_ids(14)
    add("prefill_then_decode_dead", [(0, ids[:7], None), (7, [ids[7]], [0]), (8, [ids[8]], None), (9, [ids[9]], None),
                                     (10, [ids[10]], None), (11, ids[11:14], [1, 0, 1])])
    ids = rand_ids(9)
    add("decode_from_single_token_prefill", [(0, ids[:1], None)] + [(1 + i, [ids[1 + i]], None) for i in range(8)])
    add("prefill_dead_tail_then_decode", [(0, rand_ids(6), [1, 1, 1, 1, 0, 0]), (6, rand_ids(1), None),
                                          (7, rand_ids(2), None)])
    long_ids = rand_ids(96)
    long_mask = [int(x) for x in (rng.random(96) > 0.12)]
    add("long_random_masked", [(0, long_ids, long_mask)])
    ids = rand_ids(40)
    mask = [int(x) for x in (rng.random(40) > 0.2)]
    add("chunked_prefill_random", [(0, ids[:16], mask[:16]), (16, ids[16:29], mask[16:29]), (29, ids[29:], mask[29:])])

    payload = {
        "source": "upstream inference/engram.py NgramHashState.forward",
        "checkpointRevision": manifest["checkpoint_revision"],
        "layerIds": list(Args.engram_layer_ids), "columns": 24, "layers": len(Args.engram_layer_ids),
        "padId": Args.engram_pad_id, "cases": cases,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    print(len(cases), "cases", os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
