"""Reference values for Pixal3D's view-aligned projection, produced by upstream's own code.

Runs `project_points_to_image_batch`, `sample_features` and `ProjGrid` from
`pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`, and `distance_from_fov` from `inference.py`, of a checkout of
https://github.com/TencentARC/Pixal3D (pinned in docs/Research/PIXAL3D_ARCHITECTURE.md). Only those pure-torch definitions are executed; the modules'
other imports (transformers, torchvision) are not needed. Output is the small JSON fixture the C# unit tests read.

    python dump_pixal3d_projection.py --upstream <Pixal3D checkout> --out <fixture.json>
"""
import argparse
import ast
import json
import math
import types
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_definitions(path, names):
    """Executes only the named top-level functions/classes of a source file in a torch namespace."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in keep}
    if missing:
        raise SystemExit(f"{path} does not define {sorted(missing)}")
    # PIL only appears in annotations of visualisation helpers that are never called here.
    namespace = {"torch": torch, "nn": nn, "F": F, "Image": types.SimpleNamespace(Image=object), "ImageDraw": types.SimpleNamespace()}
    namespace.update({name: getattr(typing, name) for name in ("Optional", "Tuple", "Union", "List", "Dict")})
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), namespace)
    return namespace


def projected_coordinates(ns, proj, fov, distance, mesh_scale, image_size):
    """The normalized coordinates ProjGrid.forward hands to sample_features, captured from its own code path."""
    captured = {}
    real = ns["sample_features"]

    def capture(fmap, queries_ndc):
        captured["uv"] = queries_ndc.detach().clone()
        return real(fmap, queries_ndc)

    ns["sample_features"] = capture
    try:
        fmap = torch.zeros(1, 1, 4, 4)
        proj(fmap, torch.tensor([fov]), torch.tensor([distance]), torch.tensor([mesh_scale]), BHWC=False)
    finally:
        ns["sample_features"] = real
    return captured["uv"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mixin = f"{args.upstream}/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py"
    ns = load_definitions(mixin, {"project_points_to_image_batch", "sample_features", "ProjGrid"})
    infer = load_definitions(f"{args.upstream}/inference.py", {"compute_f_pixels", "distance_from_fov"})
    ProjGrid, sample_features = ns["ProjGrid"], ns["sample_features"]

    cases = []
    configs = [
        (16, 512, 0.8575560450553894, 1.0),
        (8, 512, 0.5, 1.2),
        (8, 1024, 1.3, 0.9),
        (32, 512, 0.2, 1.0),
    ]
    for resolution, image_size, fov, mesh_scale in configs:
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = infer["distance_from_fov"](
            fov, grid_point, torch.tensor([0, image_size - 1]), mesh_scale, image_size)["distance_from_x"]
        proj = ProjGrid(grid_resolution=resolution, image_resolution=image_size)
        uv = projected_coordinates(ns, proj, fov, distance, mesh_scale, image_size)
        count = resolution ** 3
        stride = max(1, count // 300)
        indices = list(range(0, count, stride))
        cases.append({
            "grid_resolution": resolution,
            "image_size": image_size,
            "fov": fov,
            "mesh_scale": mesh_scale,
            "distance": distance,
            "cells": indices,
            "uv": [[float(uv[i, 0]), float(uv[i, 1])] for i in indices],
        })

    generator = torch.Generator().manual_seed(7)
    sample_cases = []
    for height, width, channels in [(6, 5, 3), (32, 32, 4)]:
        fmap = torch.randn(1, channels, height, width, generator=generator)
        grid = torch.rand(1, 40, 2, generator=generator) * 2.6 - 1.3
        out = sample_features(fmap, grid)[0]
        sample_cases.append({
            "height": height,
            "width": width,
            "channels": channels,
            # Row-major pixels with channels fastest, the layout of DINOv3 patch tokens.
            "rows": fmap[0].permute(1, 2, 0).reshape(-1).tolist(),
            "uv": grid[0].reshape(-1).tolist(),
            "expected": out.permute(1, 0).reshape(-1).tolist(),
        })

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"torch": torch.__version__, "projection": cases, "sampling": sample_cases}, handle, separators=(",", ":"))
    print(f"wrote {args.out}: {len(cases)} projection cases, {len(sample_cases)} sampling cases")


if __name__ == "__main__":
    main()
