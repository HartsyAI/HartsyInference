"""Dump the Engram constants the C# port must load as pinned fixtures rather than re-derive:
compressed token map, hash multipliers, primes, cumulative offsets. Uses the unmodified upstream engram.py.

Usage: python dump_engram_constants.py [--upstream ~/dsv41-ref/upstream] [--out deepseek_v41_ref/engram]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_v41_ref", "engram"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    from engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers
    from transformers import PreTrainedTokenizerFast

    cfg = json.load(open(os.path.join(a.upstream, "inference", "config.json")))
    root_cfg = json.load(open(os.path.join(a.upstream, "config.json")))["text_config"]
    for k in ("engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size", "engram_vocab_size",
              "engram_n_heads", "engram_head_dim", "engram_compressed_vocab_size"):
        assert list(np.atleast_1d(cfg[k])) == list(np.atleast_1d(root_cfg[k])), k

    class Args:
        engram_layer_ids = tuple(cfg["engram_layer_ids"])
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]
        engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.upstream, "tokenizer.json"))
    token_map, vocab = build_compressed_token_map(tok)
    assert vocab == Args.engram_compressed_vocab_size, vocab
    layout = EngramLayout.from_args(Args)
    mult = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab).numpy()
    flat = [[p for per in layer for p in per] for layer in layout.primes]
    offsets = np.array([np.cumsum([0, *s[:-1]]) for s in flat], dtype=np.int64)
    totals = [int(sum(s)) for s in flat]
    assert tuple(totals) == Args.engram_num_embeddings, (totals, Args.engram_num_embeddings)

    os.makedirs(a.out, exist_ok=True)
    files = {
        "token_map.i32.bin": np.asarray(token_map, dtype="<i4"),
        "multipliers.i64.bin": mult.astype("<i8"),
        "offsets.i64.bin": offsets.astype("<i8"),
        "primes.i64.bin": np.asarray(layout.primes, dtype="<i8"),
    }
    manifest = {"upstream_config": "inference/config.json", "tokenizer_vocab": len(tok), "compressed_vocab": vocab,
                "pad_id": Args.engram_pad_id, "pad_compressed_id": token_map[Args.engram_pad_id],
                "layer_ids": list(layout.layer_ids), "max_ngram_size": layout.max_ngram_size, "n_heads": layout.n_heads,
                "table_rows": totals, "files": {}}
    for name, arr in files.items():
        path = os.path.join(a.out, name)
        arr.tofile(path)
        manifest["files"][name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha256(path)}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))
    print("multipliers", mult.tolist())
    print("first primes", np.asarray(layout.primes)[0, 0].tolist())


if __name__ == "__main__":
    main()
