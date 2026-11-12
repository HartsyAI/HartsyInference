"""Reference values for the Hunyuan3D-Omni control encoder, produced by upstream's own code on the CPU.

Runs `FourierEmbedder` of `hy3dshape/models/modules/embedder.py` of a checkout of Tencent-Hunyuan/Hunyuan3D-Omni (pinned in docs/Research/HUNYUAN3D_OMNI_ARCHITECTURE.md) and
the layers `OmniEncoder` builds (`Linear(102, 1024)`, `RMSNorm(1024)`, GELU, `Embedding(4, 8)`, `Linear(8, 1024)`) with the released `cond_encoder/pytorch_model.bin`, and transcribes
the control branches of `OmniEncoder.forward` and its `generate_voxel` (the class itself imports transformers, torchvision and the whole `hy3dshape` package for the DINO encoder,
which this fixture does not need). The `.bin` file is read by the restricted unpickler of `dump_paint21_unet.py`, never by `torch.load`.

    python dump_omni_control.py --upstream <Omni checkout>/hy3dshape/models/modules/embedder.py --weights <cond_encoder/pytorch_model.bin> --out <fixture.safetensors>

Fixture keys, for `pose`, `bbox`, `voxel` and `point`: `<kind>.in` (the raw control input) and `<kind>.out` ([1, M + 10, 1024]: the control tokens, then the 10 signal tokens).
"""
import argparse
import importlib.util
import os
import types

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH = 1024


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_voxel(pc, resolution=16):
    """OmniEncoder.generate_voxel for one batch item (pc [1, N, 3] in -1..1), transcribed."""
    points_norm = (pc + 1) / 2
    voxels = (points_norm * resolution).floor().long()
    voxels = torch.clamp(voxels, 0, resolution - 1)
    vox_b = voxels[0]
    linear_idx = vox_b[:, 0] + vox_b[:, 1] * resolution + vox_b[:, 2] * resolution * resolution
    unique_idx = torch.unique(linear_idx)
    z = unique_idx // (resolution * resolution)
    y = (unique_idx % (resolution * resolution)) // resolution
    x = unique_idx % resolution
    unique_voxels = torch.stack([x, y, z], dim=1).float()
    voxel_size = 2.0 / resolution
    return (unique_voxels * voxel_size + voxel_size / 2 - 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="hy3dshape/models/modules/embedder.py of the Omni checkout")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    embedder = load_module(args.upstream, "omni_embedder")
    reader = load_module(os.path.join(HERE, "dump_paint21_unet.py"), "dump_paint21_unet")
    state = reader.read_state_dict(args.weights)

    pe = embedder.FourierEmbedder(num_freqs=8, include_pi=False)
    linear = nn.Sequential(nn.Linear(pe.get_dims(6), WIDTH), nn.RMSNorm(WIDTH), nn.GELU())
    linear[0].weight.data, linear[0].bias.data, linear[1].weight.data = state["linear.0.weight"].clone(), state["linear.0.bias"].clone(), state["linear.1.weight"].clone()
    signal_embedding = nn.Embedding(4, 8)
    signal_embedding.weight.data = state["cond_signal_embedding.weight"].clone()
    signal_linear = nn.Linear(8, WIDTH)
    signal_linear.weight.data, signal_linear.bias.data = state["cond_signal_linear.weight"].clone(), state["cond_signal_linear.bias"].clone()

    def encode(rows, kind):
        cond = linear(pe(rows))
        signal = signal_linear(signal_embedding(torch.tensor([kind]))).unsqueeze(0).repeat(1, 10, 1)
        return torch.cat([cond, signal], dim=1)

    out = {}
    with torch.no_grad():
        pose = torch.rand(1, 22, 6) * 2 - 1
        out["pose.in"], out["pose.out"] = pose[0], encode(pose, 0)[0]
        bbox = torch.tensor([[[0.5, 0.8, 0.3]]])
        out["bbox.in"], out["bbox.out"] = bbox[0], encode(bbox.repeat(1, 1, 2), 1)[0]
        cloud = torch.randn(1, 3000, 3)
        cloud = cloud / cloud.norm(dim=-1, keepdim=True) * (0.3 + 0.6 * torch.rand(1, 3000, 1))
        voxel = generate_voxel(cloud)
        out["voxel.in"], out["voxel.out"] = cloud[0], encode(voxel.repeat(1, 1, 2), 2)[0]
        points = torch.rand(1, 1024, 3) * 2 - 1
        out["point.in"], out["point.out"] = points[0], encode(points.repeat(1, 1, 2), 3)[0]
    save_file({k: v.contiguous().float() for k, v in out.items()}, args.out)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items() if k.endswith(".out")})


if __name__ == "__main__":
    main()
