"""Dump the Engram constants the C# port must load as pinned fixtures rather than re-derive:
compressed token map, hash multipliers, primes, cumulative offsets. Uses the unmodified upstream engram.py.

Usage: python dump_engram_constants.py [--upstream ~/dsv41-ref/upstream] [--out deepseek_v41_ref/engram] [--cross-check]

--cross-check range-fetches two small public sources (no token, no weights) and records whether they match the dump:
the DwarfStar GGUF metadata head (deepseek41.engram.token_map/primes/multipliers) and the MLX engram_token_map.json.
The committed copy lives in src/HartsyInference.LLM/DeepSeekV41/Engram/Constants/ and is embedded into the LLM assembly.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

import numpy as np

CHECKPOINT_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
GGUF_URL = ("https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/"
            "dd8a266f7145edc19e2334b46e19b6821f221dc7/DeepSeek-V4.1-Flash-Q2.gguf")
MLX_URL = ("https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-MLX-4bit/resolve/"
           "100694c1c65d34e331f4f0c2841212f0e849a0ea/engram_token_map.json")
GGUF_HEAD_BYTES = 24 * 1024 * 1024


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, byte_range: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={byte_range}"} if byte_range else {})
    with urllib.request.urlopen(request) as response:
        return response.read()


def read_gguf_engram_arrays(head: bytes) -> dict[str, np.ndarray]:
    """Walks the GGUF key/value block and returns the deepseek41.engram.* numeric arrays (the head must cover it)."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    pos = 0

    def unpack(fmt: str):
        nonlocal pos
        value = struct.unpack_from(fmt, head, pos)
        pos += struct.calcsize(fmt)
        return value

    def string() -> None:
        nonlocal pos
        (n,) = unpack("<Q")
        pos += n

    def value(kind: int):
        nonlocal pos
        if kind == 8:
            string()
            return None
        if kind != 9:
            return unpack("<" + scalar[kind])[0]
        (item,) = unpack("<I")
        (n,) = unpack("<Q")
        if item in scalar and item != 7:
            fmt = scalar[item]
            arr = np.frombuffer(head, dtype=np.dtype("<" + fmt), count=n, offset=pos)
            pos += n * struct.calcsize(fmt)
            return arr
        for _ in range(n):
            value(item)
        return None

    _, _, _, kv_count = unpack("<IIQQ")
    found = {}
    for _ in range(kv_count):
        (n,) = unpack("<Q")
        key = head[pos:pos + n].decode()
        pos += n
        (kind,) = unpack("<I")
        v = value(kind)
        if key.startswith("deepseek41.engram.") and isinstance(v, np.ndarray):
            found[key] = v
    return found


def cross_check(token_map: np.ndarray, mult: np.ndarray, primes: np.ndarray) -> dict:
    gguf = read_gguf_engram_arrays(fetch(GGUF_URL, f"0-{GGUF_HEAD_BYTES - 1}"))
    mlx = np.asarray(json.loads(fetch(MLX_URL)), dtype="<i4")
    return {
        "gguf_dwarfstar": {
            "repo": "antirez/deepseek-v4.1-flash-gguf", "revision": "dd8a266f7145edc19e2334b46e19b6821f221dc7",
            "token_map_equal": bool(np.array_equal(token_map, gguf["deepseek41.engram.token_map"].astype("<i4"))),
            "primes_equal": bool(np.array_equal(primes.ravel(), gguf["deepseek41.engram.primes"].astype("<i8"))),
            "multipliers_equal": bool(np.array_equal(mult.ravel(), gguf["deepseek41.engram.multipliers"].astype("<i8"))),
        },
        "mlx_4bit": {
            "repo": "mlx-community/DeepSeek-V4.1-Flash-MLX-4bit", "revision": "100694c1c65d34e331f4f0c2841212f0e849a0ea",
            "token_map_equal": bool(np.array_equal(token_map, mlx)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_v41_ref", "engram"))
    ap.add_argument("--cross-check", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    from engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers
    from transformers import PreTrainedTokenizerFast

    with open(os.path.join(a.upstream, "inference", "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(a.upstream, "config.json")) as f:
        root_cfg = json.load(f)["text_config"]
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
    manifest = {"checkpoint": "deepseek-ai/DeepSeek-V4.1-Flash", "checkpoint_revision": CHECKPOINT_REVISION,
                "upstream_config": "inference/config.json", "tokenizer_vocab": len(tok), "compressed_vocab": vocab,
                "pad_id": Args.engram_pad_id, "pad_compressed_id": token_map[Args.engram_pad_id],
                "layer_ids": list(layout.layer_ids), "max_ngram_size": layout.max_ngram_size, "n_heads": layout.n_heads,
                "table_rows": totals, "files": {}}
    for name, arr in files.items():
        path = os.path.join(a.out, name)
        arr.tofile(path)
        manifest["files"][name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha256(path)}
    if a.cross_check:
        manifest["cross_checks"] = cross_check(files["token_map.i32.bin"], files["multipliers.i64.bin"],
                                               files["primes.i64.bin"])
        assert all(v for src in manifest["cross_checks"].values() for k, v in src.items() if k.endswith("_equal"))
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))
    print("multipliers", mult.tolist())
    print("first primes", np.asarray(layout.primes)[0, 0].tolist())


if __name__ == "__main__":
    main()
