"""Run the UNMODIFIED upstream vision tower and aligner (inference/vision.py) on the REAL checkpoint's vision weights (CPU, float32).

Reads the 266 tensors (vision.*, aligner.*, image_*) by byte range from their shards (no safe_open and no mmap), widens the BF16 values to float32, loads
them into upstream's ViT and Aligner and runs fixed seeded inputs: patches uniform in [-1, 1], one grid per --grids entry. Writes each input and every
stage (patch embedding, each block's output stream, the final norm, the aligner's folded rows, its hidden activation and its output) plus the three image
embeddings as raw little-endian float32, and meta.json. The C# side (DeepSeekV41VisionRealWeightsTests) loads the same tensors through
DeepSeekV41VisionLoader and compares stage by stage; the dump holds the inputs, so the C# side replicates no random generator.
Usage: python dump_real_vision.py <checkpoint dir> <out dir> [--grids 28x28,17x23] [--seed 20260930] [--threads 6]
Then set DSV41_VISION_ORACLE=<out dir> for the test.
"""
import argparse
import json
import os
import struct
import sys
import time
import types

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS: tuple = ("image_start", "image_end", "image_newline")
VISION_FIELDS: tuple = ("vision_patch_size", "vision_dim", "vision_n_heads", "vision_inter_dim", "vision_n_layers", "vision_rope_theta",
                        "vision_downsample_ratio", "dim")


class ShardReader:
    """Reads tensors by byte range: the safetensors header is parsed once per shard and each tensor is one pread."""

    def __init__(self, directory: str, weight_map: dict):
        self.directory = directory
        self.weight_map = weight_map
        self.headers: dict = {}

    def header(self, shard: str):
        if shard not in self.headers:
            with open(os.path.join(self.directory, shard), "rb") as fh:
                (length,) = struct.unpack("<Q", fh.read(8))
                self.headers[shard] = (json.loads(fh.read(length)), 8 + length)
        return self.headers[shard]

    def read(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        header, base = self.header(shard)
        entry = header[key]
        if entry["dtype"] != "BF16":
            sys.exit(f"{key} is {entry['dtype']}; this dump handles the official BF16 vision tensors")
        start, end = entry["data_offsets"]
        fd = os.open(os.path.join(self.directory, shard), os.O_RDONLY)
        try:
            buf = bytearray(os.pread(fd, end - start, base + start))
        finally:
            os.close(fd)
        if len(buf) != end - start:
            sys.exit(f"{key}: short read ({len(buf)} of {end - start} bytes)")
        return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(entry["shape"]).to(torch.float32)


def write_f32(path: str, t: torch.Tensor) -> list:
    t = t.detach().to(torch.float32).contiguous().cpu()
    with open(path, "wb") as fh:
        fh.write(t.numpy().tobytes())
    return list(t.shape)


def parse_grids(text: str) -> list:
    grids = []
    for part in text.split(","):
        h, w = part.lower().split("x")
        grids.append((int(h), int(w)))
    return grids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--grids", default="28x28,17x23")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import vision

    with open(os.path.join(a.checkpoint, "inference", "config.json")) as fh:
        raw = json.load(fh)
    cfg = {k: raw[k] for k in VISION_FIELDS}
    args = types.SimpleNamespace(**cfg)
    with open(os.path.join(a.checkpoint, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]

    torch.set_default_dtype(torch.float32)
    t0 = time.time()
    vit = vision.ViT(args)
    aligner = vision.Aligner(args)
    wanted = {f"vision.{n}": prm for n, prm in vit.named_parameters()}
    wanted.update({f"aligner.{n}": prm for n, prm in aligner.named_parameters()})
    in_checkpoint = {k for k in weight_map if k.startswith(("vision.", "aligner.")) or k in EMBEDDINGS}
    if set(wanted) | set(EMBEDDINGS) != in_checkpoint:
        sys.exit(f"checkpoint and model disagree: only in model {sorted(set(wanted) - in_checkpoint)[:5]}, "
                 f"only in checkpoint {sorted(in_checkpoint - set(wanted) - set(EMBEDDINGS))[:5]}")
    reader = ShardReader(a.checkpoint, weight_map)
    for key, prm in wanted.items():
        value = reader.read(key)
        if tuple(value.shape) != tuple(prm.shape):
            sys.exit(f"{key}: checkpoint shape {tuple(value.shape)} != model shape {tuple(prm.shape)}")
        prm.data = value
    vit.eval()
    aligner.eval()
    print(f"loaded {len(wanted)} tensors in {time.time() - t0:.1f}s")

    os.makedirs(a.out, exist_ok=True)
    embedding_files = {}
    for name in EMBEDDINGS:
        embedding_files[name] = write_f32(os.path.join(a.out, f"{name}.f32"), reader.read(name))

    taps: dict = {}
    vit.patch_embed.register_forward_hook(lambda _m, _i, out: taps.__setitem__("patch_embed", out))
    for index, block in enumerate(vit.blocks):
        block.register_forward_hook(lambda _m, _i, out, index=index: taps.__setitem__(f"block.{index}", out))
    vit.norm.register_forward_hook(lambda _m, _i, out: taps.__setitem__("norm", out))
    aligner.w1.register_forward_pre_hook(lambda _m, inp: taps.__setitem__("unfold", inp[0]))
    aligner.w2.register_forward_pre_hook(lambda _m, inp: taps.__setitem__("hidden", inp[0]))

    patch = cfg["vision_patch_size"]
    ratio = cfg["vision_downsample_ratio"]
    grids_meta = []
    for n_h, n_w in parse_grids(a.grids):
        tag = f"g{n_h}x{n_w}"
        generator = torch.Generator().manual_seed(a.seed + 1000 * n_h + n_w)
        patches = torch.rand(n_h * n_w, 3, patch, patch, generator=generator) * 2 - 1
        taps.clear()
        t1 = time.time()
        with torch.inference_mode():
            out = aligner(vit(patches, n_h, n_w), n_h, n_w)
        seconds = time.time() - t1
        stages = dict(taps)
        stages["patches"] = patches.flatten(1)
        stages["out"] = out
        shapes = {}
        for stage, value in stages.items():
            if not torch.isfinite(value).all():
                sys.exit(f"{tag} {stage} holds non-finite values")
            shapes[stage] = write_f32(os.path.join(a.out, f"{tag}.{stage}.f32"), value)
        last = f"block.{cfg['vision_n_layers'] - 1}"
        print(f"{tag}: {seconds:.1f}s, output {tuple(out.shape)}, |out| max {float(out.abs().max()):.3f}, "
              f"|norm| max {float(stages['norm'].abs().max()):.3f}, |{last}| max {float(stages[last].abs().max()):.3f}")
        grids_meta.append({"tag": tag, "height": n_h, "width": n_w, "patches": n_h * n_w, "tokenHeight": -(-n_h // ratio),
                           "tokenWidth": -(-n_w // ratio), "seconds": round(seconds, 2), "shapes": shapes})

    meta = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "dba1be0a40aa45a94ad051997016db3960a90277", "dtype": "float32",
            "weights": "BF16 checkpoint values widened to float32", "tensors": len(wanted) + len(EMBEDDINGS), "seed": a.seed, "torch": torch.__version__,
            "threads": a.threads, "config": cfg, "layers": cfg["vision_n_layers"], "embeddings": embedding_files, "grids": grids_meta}
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
