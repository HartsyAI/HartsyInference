"""Shared setup for the TRELLIS.2 stage-parity scripts: CPU fp32 upstream code with stand-ins for the CUDA-only pieces.

Nothing from upstream is copied into this repository: `setup` registers the upstream `trellis2` package from the clone you point it at
(without running its `__init__`, which imports the renderers and pipelines) and puts the stand-ins in `shims/` in front of the CUDA
libraries (`flex_gemm`, `flash_attn`, `o_voxel`, `cumesh`).
"""
import json
import os
import struct
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
state = {}


def setup(upstream, models, shape_vae):
    """upstream: a clone of microsoft/TRELLIS.2; models: folder with the ckpts (ss_dec, slat flows, tex decoder); shape_vae: path of the
    shape decoder without extension (shape_dec_next_dc_f16c32_fp16)."""
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRELLIS2_UPSTREAM"] = upstream
    sys.path[:0] = [os.path.join(HERE, "shims")]
    package = types.ModuleType("trellis2")
    package.__path__ = [os.path.join(upstream, "trellis2")]
    sys.modules["trellis2"] = package
    state.update(models=models, shape_vae=shape_vae)
    import torch
    torch.set_grad_enabled(False)
    return torch


def load_model(path):
    """An upstream model from `<path>.json` and `<path>.safetensors`, in float32 (no fp16 or bf16 anywhere)."""
    import torch
    from safetensors.torch import load_file
    from trellis2 import models
    cfg = json.load(open(path + ".json"))
    args = dict(cfg["args"])
    if "use_fp16" in args:
        args["use_fp16"] = False
    if "dtype" in args:
        args["dtype"] = "float32"
    model = getattr(models, cfg["name"])(**args)
    weights = {k: v.float() for k, v in load_file(path + ".safetensors").items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{path}: {len(missing)} missing, {len(unexpected)} unexpected keys, e.g. {missing[:2]} {unexpected[:2]}")
    return model.float().eval()


def read_tns(path):
    """The tensor container the C# capture harness writes: repeated [int32 nameLen][name][int32 dtype 0=f32 1=i32][int32 ndim][int64 dims][data]."""
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        pos += 4
        name = data[pos:pos + n].decode()
        pos += n
        dtype, nd = struct.unpack_from("<ii", data, pos)
        pos += 8
        dims = struct.unpack_from("<" + "q" * nd, data, pos)
        pos += 8 * nd
        count = int(np.prod(dims)) if nd else 1
        out[name] = np.frombuffer(data, dtype=np.float32 if dtype == 0 else np.int32, count=count, offset=pos).reshape(dims).copy()
        pos += count * 4
    return out


def stats(label, ref, got):
    """Prints max |diff|, mean |diff| and the correlation of two arrays of the same size (ref is the C# side, got the upstream side)."""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    got = np.asarray(got, dtype=np.float64).ravel()
    d = np.abs(ref - got)
    corr = float(np.corrcoef(ref, got)[0, 1]) if ref.std() > 0 and got.std() > 0 else float("nan")
    print(f"PARITY {label}: n={ref.size} max|diff|={d.max():.3e} mean|diff|={d.mean():.3e} corr={corr:.7f} ref_absmax={np.abs(ref).max():.3e}", flush=True)


def keys(c, base=4096):
    c = c.astype(np.int64)
    if c.shape[1] == 4:
        return ((c[:, 0] * base + c[:, 1]) * base + c[:, 2]) * base + c[:, 3]
    return (c[:, 0] * base + c[:, 1]) * base + c[:, 2]


def align(ref_coords, got_coords):
    """Row indices (into ref, into got) of the coordinates present in both, and the two row counts."""
    rk, gk = keys(ref_coords), keys(got_coords)
    ro, go = np.argsort(rk), np.argsort(gk)
    _, ri, gi = np.intersect1d(rk[ro], gk[go], return_indices=True)
    return ro[ri], go[gi], len(rk), len(gk)


def timed(label, fn):
    t = time.time()
    result = fn()
    print(f"TIME {label} {time.time() - t:.1f}s", flush=True)
    return result
