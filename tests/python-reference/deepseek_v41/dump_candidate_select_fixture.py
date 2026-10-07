"""Dump upstream select_candidate_blocks and the Indexer's final top-k selection into fixtures/candidate_select.json.

Scores have no ties among finite values, so the expected picks do not depend on torch's tie order.
Usage: python dump_candidate_select_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
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
    return [None if float(v) == float("-inf") else float(v) for v in t.detach().float().reshape(-1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    g = torch.Generator().manual_seed(11)
    blocks = []
    for width, lens, topk_blocks, block_size in [(20, 20, 3, 4), (20, 13, 3, 4), (17, 9, 2, 8), (10, 3, 5, 4), (12, 12, 100, 4)]:
        logits = torch.randn(1, width, generator=g)
        logits[:, lens:] = float("-inf")
        mask = mod.select_candidate_blocks(logits, lens, topk_blocks, block_size)
        blocks.append({"width": width, "compressLen": lens, "topkBlocks": topk_blocks, "blockSize": block_size,
                       "logits": flat(logits), "mask": [bool(v) for v in mask.reshape(-1)]})

    picks = []
    # (positions, compress_len, topk, offset): mirrors Indexer.forward's tail for a decode row (compress_lens == positions)
    # and a prefill row where compress_lens < positions
    for positions, lens, topk_cfg, offset in [(16, 16, 4, 128), (16, 9, 4, 128), (16, 3, 6, 128), (8, 8, 20, 0)]:
        score = torch.randn(1, positions, generator=g)
        score[:, lens:] = float("-inf")
        topk = min(topk_cfg, positions)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        out = torch.where(idxs < lens, idxs + offset, -1).int()
        picks.append({"positions": positions, "compressLen": lens, "indexTopk": topk_cfg, "offset": offset,
                      "scores": flat(score), "indices": [int(v) for v in out.reshape(-1)]})
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "candidate_select.json"), "w") as fh:
        json.dump({"blocks": blocks, "picks": picks}, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
