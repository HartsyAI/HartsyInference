"""Reference values for the X-Part conditioner, produced by upstream's own modules on the CPU.

Composes `Conditioner.forward` of `XPart/partgen/models/conditioner/condioner_release.py` from the pieces the other two oracles already run against the released
`conditioner/conditioner.safetensors`: the object encoder and the part encoder (`dump_xpart_shapevae_encoder.py`), the Sonata feature extractor (`SonataFeatureExtractor`:
the Sonata backbone of `partgen/models/sonata`, the unpooled stage features concatenated to 1232 channels, the 1232-512-512-512 MLP, copied back to every point;
`dump_p3sam_features.py` runs the same modules), the nearest-point gather of the Sonata features at each encoder's query points (`torch.cdist` argmin over the object's points) and
the two output projections `obj_out_proj` and `geo_out_proj` (1536 to 1024). The differences from a normal run are those of the two scripts it reuses: stand-ins for `spconv`,
`torch_scatter`, `timm` and `addict`, `einops` replaced, the Sonata pooling shuffle pinned off, the voxel picked as the first point of each voxel (upstream picks at random), and given
query indices instead of `torch_cluster.fps` from a random start.

    python dump_xpart_conditioner.py --upstream <Hunyuan3D-Part checkout> --part <hunyuan3d-part folder> --out <fixture.safetensors>

Fixture keys: `points` [N, 3], `feats` [N, 4] (the object surface: normal and sharp-edge label), `part_points.<k>` [N, 3] and `part_feats.<k>` for two parts, `query_idx` [Q] and
`part_query_idx.<k>`, `sonata` [N, 512] (the Sonata features of the object), `obj_cond` [Q, 1024] and `geo_cond.<k>` [Q, 1024]. Times are printed.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear(tensors, prefix):
    w, b = tensors[prefix + ".weight"], tensors[prefix + ".bias"]
    layer = nn.Linear(w.shape[1], w.shape[0])
    layer.weight.data, layer.bias.data = w.clone(), b.clone()
    return layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="a Hunyuan3D-Part checkout (the folder holding XPart/ and P3-SAM/)")
    parser.add_argument("--part", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--queries", type=int, default=256)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    p3 = load_module("dump_p3sam_features")
    enc = load_module("dump_xpart_shapevae_encoder")

    path = os.path.join(args.part, "conditioner", "conditioner.safetensors")
    with safe_open(path, framework="pt") as handle:
        everything = {k: handle.get_tensor(k).float() for k in handle.keys()}

    # Sonata features of the object (the seg_feat_encoder).
    p3.install_stubs()
    sonata = p3.load_sonata(args.upstream)
    with open(os.path.join(args.upstream, "XPart", "partgen", "config", "sonata.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(enable_flash=False, shuffle_orders=False, drop_path=0.0)
    model = sonata.PointTransformerV3(**config)
    state = {k[len("seg_feat_encoder."):]: v for k, v in everything.items() if k.startswith("seg_feat_encoder.")}
    missing, unexpected = model.load_state_dict({k[len("sonata."):]: v for k, v in state.items() if k.startswith("sonata.")}, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Sonata weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.eval()
    for module in model.modules():
        if isinstance(module, sonata.GridPooling):
            module.shuffle_orders = False
    mlp = p3.mlp(state, "mlp")

    rng = np.random.default_rng(3)
    count = args.points
    theta, phi = np.arccos(1 - 2 * rng.random(count)), 2 * np.pi * rng.random(count)
    x, y, z = np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)
    points = np.stack([0.6 * x, 1.0 * y, 0.8 * z], axis=1) * 0.8
    normals = np.stack([x / 0.6, y / 1.0, z / 0.8], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    coord, grid, feat, selected, inverse = p3.transform(points, normals)
    data = {"coord": torch.tensor(coord, dtype=torch.float32), "grid_coord": torch.tensor(grid, dtype=torch.int32),
            "feat": torch.tensor(feat, dtype=torch.float32), "offset": torch.tensor([coord.shape[0]])}
    start = time.time()
    point = model(data)
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inv = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
        point = parent
    sonata_feats = mlp(point.feat)[torch.tensor(inverse)]  # [N, 512]
    print(f"sonata features for {count} points: {time.time() - start:.1f} s")

    # The two shape encoders (object, and the part in its box) as in the encoder oracle.
    pts = torch.tensor(points, dtype=torch.float32)
    # [normal, sharp-edge label]: upstream's label is 0 or 1; a continuous value here only exercises the 4th input channel.
    obj_feats = torch.cat([torch.tensor(normals, dtype=torch.float32), torch.rand(count, 1)], dim=1)
    blocks = enc.load_upstream(os.path.join(args.upstream, "XPart", "partgen", "models", "autoencoders"))
    obj_query = enc.farthest_points(pts, args.queries)
    obj_tokens = enc.encode(blocks, {k: v for k, v in everything.items() if k.startswith("obj_encoder.encoder.")}, "obj_encoder.", pts, obj_feats, obj_query)[1]

    out = {"points": pts, "feats": obj_feats, "query_idx": obj_query.float(), "sonata": sonata_feats}
    obj_gather = torch.argmin(torch.cdist(pts[obj_query], pts), dim=-1)
    obj_cond = linear(everything, "obj_out_proj")(torch.cat([obj_tokens, sonata_feats[obj_gather]], dim=-1))
    out["obj_cond"] = obj_cond

    geo_proj = linear(everything, "geo_out_proj")
    local_state = {k: v for k, v in everything.items() if k.startswith("geo_encoder.local_encoder.encoder.")}
    for k in range(2):
        # A part (a stand-in for sample_bbox_points_from_trimesh): points of the object surface inside a box, re-sampled with a different seed and nudged so that the nearest object point is not the point itself.
        keep = torch.nonzero(pts[:, 0] * (1 if k == 0 else -1) > -0.1).squeeze(1)
        take = keep[torch.randperm(len(keep))[:count]] if len(keep) >= count else keep[torch.randint(len(keep), (count,))]
        part_pts = pts[take] + 0.002 * torch.randn(count, 3)
        part_feats = obj_feats[take]
        part_query = enc.farthest_points(part_pts, args.queries)
        tokens = enc.encode(blocks, local_state, "geo_encoder.local_encoder.", part_pts, part_feats, part_query)[1]
        gather = torch.argmin(torch.cdist(part_pts[part_query], pts), dim=-1)
        out[f"part_points.{k}"], out[f"part_feats.{k}"], out[f"part_query_idx.{k}"] = part_pts, part_feats, part_query.float()
        out[f"geo_cond.{k}"] = geo_proj(torch.cat([tokens, sonata_feats[gather]], dim=-1))
    save_file({k: v.detach().float().contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items() if k in ("obj_cond", "geo_cond.0", "sonata")})


if __name__ == "__main__":
    main()
