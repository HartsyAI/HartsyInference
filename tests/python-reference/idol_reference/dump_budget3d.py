"""Budget3D on the CPU in float32: the octree tables, layouts from fixed scores, and one sparse forward per budget.

Usage: python -I dump_budget3d.py <denoiser dump dir> <out dir> [--budgets 0.5,0.25,0.1]

Reads denoiser_inputs.safetensors from dump_idol_denoiser.py. Writes:
  octree.safetensors    perm_to_octree, inv_perm [8192] (int64), cube_to_cells [512, 8], cell_ctr [4096, 3], cube_ctr [512, 3]
  layouts.safetensors   for the salience of the saved latent and for seeded random scores: score4 [1, 512], score2 [1, 4096],
                        and per budget b: b<b>_idx4, b<b>_idx2, b<b>_idx1 (int64) plus b<b>_k4k2 = [k4, k2]
  budget3d-idol.safetensors / budget3d-idol-turbo.safetensors
                        per budget: b<b>_v_latent [1, 8192, 16], b<b>_v_camera [1, 1, 5]; t = 731 (base) or 800 / 200 (turbo),
                        on the layout built from the saved latent's salience
"""
import argparse
import os
import sys

import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # common.py; python -I drops the script folder
import common
from budget3d.octree import N_SOBOL, build_layout, detail_scores, snap_realizable_tokens

ap = argparse.ArgumentParser()
ap.add_argument("den")
ap.add_argument("out")
ap.add_argument("--budgets", default="1.0,0.5,0.25,0.1,0.0625")
a = ap.parse_args()
budgets = [float(b) for b in a.budgets.split(",")]
key = lambda b: f"b{b:g}".replace(".", "p")

inp = common.load_file(os.path.join(a.den, "denoiser_inputs.safetensors"))
x = TensorDict({"latent": inp["latent"], "camera": inp["camera"]}, batch_size=1)
cond = TensorDict({"feature1": inp["feature1"], "feature2": inp["feature2"]}, batch_size=1)

base = common.budget3d_model(os.path.join(common.MODELS, "budget3d-idol", "budget3d_idol.safetensors"))
common.save(a.out, "octree", {"perm_to_octree": base.perm_to_octree, "inv_perm": base.inv_perm, "cube_to_cells": base.cube_to_cells,
                              "cell_ctr": base.cell_ctr, "cube_ctr": base.cube_ctr})

z_oct = inp["latent"][:, base.perm_to_octree].reshape(1, 16, 16, 16, 2, -1)
s4, s2 = detail_scores(z_oct)
g = torch.Generator().manual_seed(77)
r4, r2 = torch.rand(1, 512, generator=g), torch.rand(1, 4096, generator=g)
lay = {"score4": s4, "score2": s2, "rand_score4": r4, "rand_score2": r2}
layouts = {}
for b in budgets:
    n = snap_realizable_tokens(b * N_SOBOL)
    for tag, (p4, p2) in (("", (s4, s2)), ("rand_", (r4, r2))):
        L = build_layout(p4, p2, n)
        lay[f"{tag}{key(b)}_idx4"], lay[f"{tag}{key(b)}_idx2"], lay[f"{tag}{key(b)}_idx1"] = L.idx4, L.idx2, L.idx1
        lay[f"{tag}{key(b)}_k4k2"] = torch.tensor([L.k4, L.k2])
        if not tag:
            layouts[b] = L
        print(tag or "salience", b, L, flush=True)
common.save(a.out, "layouts", lay, {"budgets": a.budgets})


def forward(model, t, layout):
    with torch.no_grad(), common.Timer(f"forward t={t} N={layout.n_tokens}"):
        out = model(x, torch.full((1,), float(t)), cond, layout=layout)
    return out["latent"], out["camera"]


res = {}
for b in budgets:
    res[f"{key(b)}_v_latent"], res[f"{key(b)}_v_camera"] = forward(base, 731, layouts[b])
common.save(a.out, "budget3d-idol", res, {"t": 731})
del base

turbo = common.budget3d_model(os.path.join(common.MODELS, "budget3d-idol-turbo", "budget3d_idol_turbo.safetensors"), turbo=True)
res = {}
for b in budgets:
    for phase, t in (("slow", 800), ("fast", 200)):
        res[f"{key(b)}_{phase}_v_latent"], res[f"{key(b)}_{phase}_v_camera"] = forward(turbo, t, layouts[b])
common.save(a.out, "budget3d-idol-turbo", res, {"t_slow": 800, "t_fast": 200})
