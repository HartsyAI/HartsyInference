"""Shape decode, mesher and texture decode of upstream TRELLIS.2 on the CPU (fp32, naive attention, CPU sparse-conv stand-in) against the
C# captures of one real run (see README.md).

Usage: python -I run_stage_parity.py <TRELLIS.2 clone> <models dir> <shape decoder path without extension> <capture dir>
Reads shape_decode.tns, mesh.tns and texture_decode.tns from <capture dir>; prints max |diff|, mean |diff|, correlation and the CPU time of each stage.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import setup, load_model, read_tns, stats, keys, align, timed  # noqa: E402

upstream, models, shape_vae, cap = sys.argv[1:5]
torch = setup(upstream, models, shape_vae)
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
from o_voxel.convert import flexible_dual_grid_to_mesh  # noqa: E402

print("threads", torch.get_num_threads(), flush=True)

# ---- shape decode (the guided-by-prediction decoder, recording the per-level child logits) ----
sd = read_tns(f"{cap}/shape_decode.tns")
dec = load_model(shape_vae)
latent = sp.SparseTensor(torch.from_numpy(sd["latent_feats"]), torch.from_numpy(sd["latent_coords"]).int())
resolution = int(sd["latent_coords"][:, 1:].max() + 1) * 16
dec.set_resolution(resolution)
h, subs = timed("shape decode", lambda: SparseUnetVaeDecoder.forward(dec, latent, return_subs=True))
print(f"{h.feats.shape[0]} voxels, {len(subs)} subdivision levels", flush=True)
margin = dec.voxel_margin
verts = (1 + 2 * margin) * torch.sigmoid(h.feats[:, 0:3]) - margin
inter = h.feats[:, 3:6] > 0
split = F.softplus(h.feats[:, 6:7])
ri, gi, nr, ng = align(sd["coords"], h.coords[:, 1:].numpy())
print(f"voxels: C# {nr}, upstream {ng}, in both {len(ri)}", flush=True)
cs_offsets = sd["dual"] - sd["coords"].astype(np.float32)  # the C# field stores coordinate + offset
stats("shape decode dual vertex offsets", cs_offsets[ri], verts.numpy()[gi])
worst = np.abs(cs_offsets[ri] - verts.numpy()[gi]).max(axis=1)
print(f"  voxels whose offset differs by more than 1e-2: {int((worst > 1e-2).sum())}, by more than 1e-1: {int((worst > 1e-1).sum())} of {len(worst)}", flush=True)
differ = sd["inter"][ri].astype(bool) != inter.numpy()[gi]
print(f"PARITY intersected flags: {int(differ.sum())} of {differ.size} differ", flush=True)
stats("shape decode split weights", sd["split"][ri], split.numpy()[gi][:, 0])


def kids(parents, logits, perm):
    n, k = np.nonzero(logits > 0)
    off = np.stack([(k >> (2 - perm.index(a))) & 1 for a in range(3)], 1)
    c = parents[n].copy()
    c[:, 1:] = c[:, 1:] * 2 + off
    return c


# the order of a parent's eight children: the axis permutation under which every kept child exists on the next level
perm = next((list(p) for p in itertools.permutations(range(3))
             if np.isin(keys(kids(subs[0].coords.numpy(), subs[0].feats.numpy(), list(p))), keys(subs[1].coords.numpy())).all()), None)
print("child bit -> axis order", perm, flush=True)
parents = sd["latent_coords"].copy()
for i, s in enumerate(subs):
    logits = sd[f"subdiv{i}"].reshape(-1, 8)
    ri2, gi2, nr2, ng2 = align(parents, s.coords.numpy())
    print(f"subdivision level {i}: C# {nr2} parents, upstream {ng2}, in both {len(ri2)}", flush=True)
    stats(f"subdivision level {i} logits (matched parents)", logits[ri2], s.feats.numpy()[gi2])
    agree = (logits[ri2] > 0) == (s.feats.numpy()[gi2] > 0)
    print(f"  child keep decisions that differ: {int((~agree).sum())} of {agree.size}", flush=True)
    if perm is not None:
        parents = kids(parents, logits, perm)

# ---- mesher: upstream's flexible_dual_grid_to_mesh on the C# field (isolates the mesher) and on the upstream field ----
meshc = read_tns(f"{cap}/mesh.tns")
field = (torch.from_numpy(sd["coords"]).int(), torch.from_numpy(cs_offsets), torch.from_numpy(sd["inter"]).bool(), torch.from_numpy(sd["split"])[:, None])
v1, f1 = timed("upstream mesher", lambda: flexible_dual_grid_to_mesh(*field, aabb=[[-0.5] * 3, [0.5] * 3], grid_size=resolution, train=False))
print(f"mesh on the C# field: upstream {v1.shape[0]} vertices {f1.shape[0]} triangles; C# {meshc['vertices'].shape[0]} vertices {meshc['indices'].shape[0]} triangles", flush=True)
if v1.shape == meshc["vertices"].shape:
    stats("mesh vertices (same field)", meshc["vertices"], v1.numpy())
print(f"PARITY mesh triangles identical on the same field: {f1.shape == meshc['indices'].shape and bool((f1.numpy() == meshc['indices']).all())}", flush=True)
v2, f2 = flexible_dual_grid_to_mesh(h.coords[:, 1:].int(), verts, inter, split, aabb=[[-0.5] * 3, [0.5] * 3], grid_size=resolution, train=False)
print(f"upstream decode + upstream mesher: {v2.shape[0]} vertices {f2.shape[0]} triangles", flush=True)

# ---- texture decode, guided by the upstream subdivisions ----
td = read_tns(f"{cap}/texture_decode.tns")
tex = load_model(os.path.join(models, "tex_dec_next_dc_f16c32_fp16"))
tex_latent = sp.SparseTensor(torch.from_numpy(td["latent_feats"]), torch.from_numpy(td["latent_coords"]).int())
out = timed("texture decode", lambda: tex(tex_latent, guide_subs=subs) * 0.5 + 0.5)
ri, gi, nr, ng = align(td["out_coords"], out.coords.numpy())
print(f"texture voxels: C# {nr}, upstream {ng}, in both {len(ri)}", flush=True)
stats("texture decode attributes (6 channels)", td["out_feats"][ri], out.feats.numpy()[gi])
for c in range(6):
    stats(f"  channel {c}", td["out_feats"][ri][:, c], out.feats.numpy()[gi][:, c])
