"""Reference forwards of upstream Pixal3D's sparse projection flows (the 1024 shape flow and the 1024 texture flow) on the CPU in float32 with the real
checkpoints, over N random voxels, for comparing with the C# engine.

Usage: python -I dump_pixal3d_slat_flows.py <upstream src dir> <ckpt dir> <out dir> <N>
Writes coords.bin (int32 [N,4]), t.bin, global.bin, proj.bin and, per flow, <name>_x.bin, <name>_concat.bin (texture only), <name>_out.bin; meta.json.
"""
import json
import os
import struct
import sys
import time

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
src, ckpt, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, src)
import numpy as np
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
import pixal3d.modules.sparse as sp
from pixal3d.models.structured_latent_flow import SLatFlowModel


def read_safetensors(path):
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hl))
        base = 8 + hl
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
            else:
                print("skipping", name, dt)
                continue
            tensors[name] = t.reshape(meta["shape"])
    return tensors


g = torch.Generator().manual_seed(4321)
rng = np.random.default_rng(4321)
flat = rng.choice(64 ** 3, size=n, replace=False)
coords = np.stack([np.zeros(n, dtype=np.int32), flat // 4096, (flat // 64) % 64, flat % 64], axis=1).astype(np.int32)
order = np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))
coords = coords[order]
os.makedirs(out, exist_ok=True)
coords.astype("<i4").tofile(os.path.join(out, "coords.bin"))
t = torch.tensor([500.0])
glob = torch.randn(1, 5, 1024, generator=g)
proj = torch.randn(n, 2048, generator=g)
for name, tensor in (("t", t), ("global", glob), ("proj", proj)):
    tensor.numpy().astype("<f4").tofile(os.path.join(out, name + ".bin"))
meta = {"n": n, "torch": torch.__version__, "threads": torch.get_num_threads()}

for name, fname, in_ch, concat in (
    ("shape", "slat_flow_img2shape_dit_1_3B_1024_bf16", 32, False),
    ("texture", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16", 32, True),
):
    cfg = dict(json.load(open(os.path.join(ckpt, fname + ".json")))["args"])
    cfg["dtype"] = "float32"
    model = SLatFlowModel(**cfg)
    weights = read_safetensors(os.path.join(ckpt, fname + ".safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(name, "missing", missing, "unexpected", unexpected)
    model.eval()
    feats = torch.randn(n, in_ch, generator=g)
    feats.numpy().astype("<f4").tofile(os.path.join(out, name + "_x.bin"))
    cc = None
    if concat:
        cfeats = torch.randn(n, 32, generator=g)
        cfeats.numpy().astype("<f4").tofile(os.path.join(out, name + "_concat.bin"))
        cc = sp.SparseTensor(feats=cfeats, coords=torch.from_numpy(coords))
    x = sp.SparseTensor(feats=feats, coords=torch.from_numpy(coords))
    with torch.no_grad():
        t0 = time.time()
        y = model(x, t, {"global": glob, "proj": proj}, concat_cond=cc)
        elapsed = time.time() - t0
    y.feats.numpy().astype("<f4").tofile(os.path.join(out, name + "_out.bin"))
    meta[name] = {"seconds": elapsed, "out_shape": list(y.feats.shape), "out_absmax": float(y.feats.abs().max())}
    print(name, "forward seconds", elapsed, tuple(y.feats.shape))
json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
