"""Dump the upstream Engram module (gate + value injection) into fixtures/engram_module.json.

The FP8 table lookup is replaced by a bf16-rounded float table, so the fixture records the module's own maths: wkv
projection, per-copy normalization, signed-sqrt sigmoid gate, token mask. Table decoding has its own fixtures.
Usage: python dump_engram_module_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
"""
import argparse
import json
import os
import sys

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports


def flat(t: torch.Tensor):
    return [float(v) for v in t.detach().float().reshape(-1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod
    from engram import EngramLayout

    mod.default_dtype = torch.float32
    torch.set_default_dtype(torch.float32)
    cases = []
    for name, dim, hc, tokens, masked in [("plain", 8, 3, 6, False), ("masked", 8, 3, 6, True), ("hc4", 6, 4, 5, True)]:
        torch.manual_seed(500 + dim + hc)
        rows, head_dim, n_heads, ngram = 40, 6, 2, 3
        args = mod.ModelArgs(dim=dim, hc_mult=hc, norm_eps=1e-6, engram_layer_ids=(1,), engram_num_embeddings=(rows,),
                             engram_max_ngram_size=ngram, engram_vocab_size=10, engram_n_heads=n_heads,
                             engram_head_dim=head_dim, engram_compressed_vocab_size=50, vocab_size=16)
        layout = EngramLayout.from_args(args)
        eng = mod.Engram(args, 1, layout)
        table = torch.randn(rows, head_dim).to(torch.bfloat16)
        eng.embed = torch.nn.Embedding.from_pretrained(table.float())
        with torch.no_grad():
            eng.wkv.weight.normal_(0, 0.5)
            eng.q_weight.normal_(1, 0.3)
            eng.k_weight.normal_(1, 0.3)
        n_cols = (ngram - 1) * n_heads
        hash_ids = torch.randint(0, rows, (1, tokens, n_cols))
        x = torch.randn(1, tokens, hc, dim)
        mask = torch.tensor([[(i % 3) != 1 for i in range(tokens)]]) if masked else None
        with torch.no_grad():
            y = eng(x.clone(), hash_ids, mask)
        cases.append({
            "name": name, "dim": dim, "hc": hc, "tokens": tokens, "rows": rows, "headDim": head_dim, "cols": n_cols,
            "eps": 1e-6, "table": flat(table), "hashIds": [int(v) for v in hash_ids.reshape(-1)],
            "wkv": flat(eng.wkv.weight), "qWeight": flat(eng.q_weight), "kWeight": flat(eng.k_weight),
            "x": flat(x), "mask": None if mask is None else [bool(v) for v in mask.reshape(-1)], "y": flat(y),
        })
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "engram_module.json"), "w") as fh:
        json.dump({"cases": cases}, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
