"""Reference forward of upstream Pixal3D's projection-mode structure flow on the CPU in float32 (real checkpoint), for comparing with the C# engine.

Usage: python -I dump_pixal3d_structure_flow.py <upstream src dir> <ckpt dir> <out dir>
Writes x.bin, t.bin, global.bin, proj.bin (inputs) and out.bin (output), all float32 little-endian, plus meta.json with timings.
"""
import json
import os
import struct
import sys
import time

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
src, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel

cfg = json.load(open(os.path.join(ckpt, "ss_flow_img_dit_1_3B_64_bf16.json")))["args"]
cfg = dict(cfg)
cfg["dtype"] = "float32"
model = SparseStructureFlowModel(**cfg)


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dt = meta["dtype"]
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if dt == "F32":
                t = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
            elif dt == "BF16":
                t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
            elif dt == "F16":
                t = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).float()
            else:
                print("skipping", name, dt)
                continue
            tensors[name] = t.reshape(meta["shape"])
    return tensors


weights = read_safetensors(os.path.join(ckpt, "ss_flow_img_dit_1_3B_64_bf16.safetensors"))
missing, unexpected = model.load_state_dict(weights, strict=False)
print("missing", missing, "unexpected", unexpected)
model.eval()

g = torch.Generator().manual_seed(1234)
x = torch.randn(1, 8, 16, 16, 16, generator=g)
t = torch.tensor([500.0])
glob = torch.randn(1, 5, 1024, generator=g)
proj = torch.randn(1, 16 * 16 * 16, cfg.get("proj_in_channels") or 1024, generator=g)
os.makedirs(out, exist_ok=True)
for name, tensor in (("x", x), ("t", t), ("global", glob), ("proj", proj)):
    tensor.numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))
with torch.no_grad():
    t0 = time.time()
    y = model(x, t, {"global": glob, "proj": proj})
    elapsed = time.time() - t0
y.numpy().astype("<f4").tofile(os.path.join(out, "out.bin"))
json.dump({"seconds": elapsed, "threads": torch.get_num_threads(), "out_shape": list(y.shape), "out_absmax": float(y.abs().max()), "out_mean": float(y.mean()), "torch": torch.__version__}, open(os.path.join(out, "meta.json"), "w"))
print("forward seconds", elapsed, "shape", tuple(y.shape))
