"""Dump dequantized 64-row windows of DeepSeek-V4.1-Flash shard 3 (layers.0) using the official convert.py/kernel.py semantics.

Dense F8_E4M3 weights: convert.py wo_a formula (float weight * per-block E8M0 scale), wo_a rounded to bf16 as convert.py does.
Routed experts (I8, two E2M1 per byte): convert.py FP4_TABLE nibbles (low nibble = even element) * per-32 E8M0 scale.
Reads only the rows it needs, so a sparse replica holding real bytes for those rows is enough. Torch CPU only.

Usage: python dump_deepseek_v41_shard3_ref.py --shard model-00003-of-00048.safetensors --out DIR [--upstream ~/dsv41-ref/upstream]
       python dump_deepseek_v41_shard3_ref.py --shard ... --ranges   (prints the [offset, length] byte ranges the dump reads)
Output: DIR/manifest.tsv plus <name>.w.f32 (window), <name>.x.f32 ([4,in] seeded input), <name>.y.f32 (F.linear(x, window)).
"""
import argparse
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F

ROWS: int = 64
BATCH: int = 4
DENSE = [
    ("attn.wq_a", 0), ("attn.wq_b", 0), ("attn.wq_b", 16384), ("attn.wkv", 0), ("attn.wo_a", 0), ("attn.wo_a", 4096),
    ("attn.wo_b", 0), ("ffn.shared_experts.w1", 0), ("ffn.shared_experts.w2", 0), ("ffn.shared_experts.w3", 0),
]
EXPERT_IDS = (0, 191, 383)
EXPERT_WEIGHTS = ("w1", "w2", "w3")
EXPERT_EXTRA = [("ffn.experts.191.w1", 1024)]
ITEM_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}


def windows() -> list:
    out = [(f"layers.0.{n}", r, "dense") for n, r in DENSE]
    for e in EXPERT_IDS:
        out += [(f"layers.0.ffn.experts.{e}.{w}", 0, "fp4") for w in EXPERT_WEIGHTS]
    out += [(f"layers.0.{n}", r, "fp4") for n, r in EXPERT_EXTRA]
    return out


class Shard:
    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.base = 8 + n

    def rows_span(self, name: str, row0: int, rows: int) -> tuple:
        meta = self.header[name]
        shape = meta["shape"]
        row_bytes = ITEM_BYTES[meta["dtype"]] * (shape[1] if len(shape) > 1 else 1)
        return self.base + meta["data_offsets"][0] + row0 * row_bytes, rows * row_bytes

    def read_rows(self, name: str, row0: int, rows: int) -> torch.Tensor:
        offset, length = self.rows_span(name, row0, rows)
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = bytearray(f.read(length))
        return torch.frombuffer(data, dtype=torch.uint8).view(rows, -1)


def scale_rows(weight_rows: int, weight_meta: dict, scale_meta: dict) -> tuple:
    """Scale-row block size (convert.py infers it from the shapes) and the scale row window for the weight rows."""
    out_dim, scale_out = weight_meta["shape"][0], scale_meta["shape"][0]
    assert out_dim % scale_out == 0
    return out_dim // scale_out, scale_meta["shape"][1]


def ranges(shard: Shard) -> list:
    out = []
    for key, row0, kind in windows():
        wm, sm = shard.header[key + ".weight"], shard.header[key + ".scale"]
        bo, _ = scale_rows(ROWS, wm, sm)
        out.append(shard.rows_span(key + ".weight", row0, ROWS))
        out.append(shard.rows_span(key + ".scale", row0 // bo, ROWS // bo))
    return out


def dequant_dense(shard: Shard, key: str, row0: int) -> torch.Tensor:
    wm, sm = shard.header[key + ".weight"], shard.header[key + ".scale"]
    bo, _ = scale_rows(ROWS, wm, sm)
    bi = wm["shape"][1] // sm["shape"][1]
    assert (bo, bi) in ((32, 32), (128, 128)), (key, wm["shape"], sm["shape"])
    w = shard.read_rows(key + ".weight", row0, ROWS).view(torch.float8_e4m3fn)
    s = shard.read_rows(key + ".scale", row0 // bo, ROWS // bo).view(torch.float8_e8m0fnu)
    out = w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float() * s[:, None, :, None].float()
    out = out.flatten(2, 3).flatten(0, 1)
    return out.bfloat16().float() if key.endswith("wo_a") else out


def dequant_fp4(shard: Shard, key: str, row0: int, fp4_table: torch.Tensor) -> torch.Tensor:
    x = shard.read_rows(key + ".weight", row0, ROWS)
    s = shard.read_rows(key + ".scale", row0, ROWS).view(torch.float8_e8m0fnu).float()
    low, high = x & 0x0F, (x >> 4) & 0x0F
    vals = torch.stack([fp4_table[low.long()], fp4_table[high.long()]], dim=-1).flatten(1)
    assert vals.shape[1] == s.shape[1] * 32, (key, vals.shape, s.shape)
    return (vals.view(ROWS, -1, 32) * s[:, :, None]).flatten(1)


def write_f32(path: str, t: torch.Tensor) -> None:
    with open(path, "wb") as f:
        f.write(t.contiguous().float().numpy().astype("<f4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out")
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--ranges", action="store_true")
    a = ap.parse_args()
    shard = Shard(a.shard)
    if a.ranges:
        print(json.dumps(ranges(shard)))
        return
    if not a.out:
        ap.error("--out is required unless --ranges")
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import convert  # noqa: E402  (official FP4_TABLE; importing does not run the conversion)

    os.makedirs(a.out, exist_ok=True)
    manifest = ["name\tkind\trowOffset\trows\tcols\tbatch"]
    with torch.no_grad():
        for i, (key, row0, kind) in enumerate(windows()):
            w = dequant_dense(shard, key, row0) if kind == "dense" else dequant_fp4(shard, key, row0, convert.FP4_TABLE)
            g = torch.Generator().manual_seed(0xD5741 + i)
            x = torch.randn(BATCH, w.shape[1], generator=g)
            y = F.linear(x, w)
            name = f"{key}@{row0}"
            write_f32(os.path.join(a.out, name + ".w.f32"), w)
            write_f32(os.path.join(a.out, name + ".x.f32"), x)
            write_f32(os.path.join(a.out, name + ".y.f32"), y)
            manifest.append(f"{name}\t{kind}\t{row0}\t{w.shape[0]}\t{w.shape[1]}\t{BATCH}")
    with open(os.path.join(a.out, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest) - 1} windows to {a.out}")


if __name__ == "__main__":
    main()
