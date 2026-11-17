"""Idol's two decoders on the CPU in float32 (Comfy-Org idol_decoder_fp16, cast to float32), on a fixed latent.

Usage: python -I dump_idol_decoders.py <sampler dump dir> <out dir> [--num-gaussians 262144]

The latent is the final latent of sampler-idol-turbo.safetensors. The octree anchor sampler draws from torch's RNG (systematic
resampling), so the engine cannot reproduce its anchors; it is compared on the recorded per-level inputs instead:
  octree.safetensors     per level l = 1..8: l<l>_x [1, P, 3] parent centres, l<l>_level [1] (the resolution 2^l the
                         decoder is given), l<l>_logits [1, P, 8]
                         and the final anchors [1, num_gaussians / 32, 3]
  gaussians.safetensors  decoder_gs(anchors, latent) features [1, A, 480]; the activated Gaussians (xyz, features_dc,
                         scaling, rotation, opacity) and the PLY values idol/gaussian.py writes (y-up xyz, log scale,
                         logit opacity, turned quaternion)
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # common.py; python -I drops the script folder
import common

ap = argparse.ArgumentParser()
ap.add_argument("samp")
ap.add_argument("out")
ap.add_argument("--num-gaussians", type=int, default=262144)
a = ap.parse_args()

from idol.gaussian import inverse_sigmoid, quaternion_multiply
from idol.models import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder

cfg = common.idol_config()
state = common.f32(common.load_file(os.path.join(common.IDOL_DIR, "vae", "idol_decoder_fp16.safetensors")))
dec = common.set_fp32(OctreeProbabilityFixedlenDecoder(**cfg["decoder"]["args"]))
dec.load_state_dict(common.strip(state, "octree."), strict=True)
dec_gs = common.set_fp32(GaussianFixedlenDecoder(**cfg["decoder_gs"]["args"]))
dec_gs.load_state_dict(common.strip(state, "gs."), strict=True)
dec.eval(), dec_gs.eval()

latent = common.load_file(os.path.join(a.samp, "sampler-idol-turbo.safetensors"))["latent"].float()
rec = {}
calls = []
dec.register_forward_hook(lambda _m, args, out: calls.append((args[0].clone(), args[1].clone(), out.detach().clone())))
torch.manual_seed(5)
per = dec_gs.rep_config["num_gaussians"]
with torch.no_grad(), common.Timer("octree descent"):
    anchors = OctreeProbabilityFixedlenDecoder.sample(dec, latent, num_points=a.num_gaussians // per, level=cfg["max_voxel_level"])
for l, (x, level, logits) in enumerate(calls, start=1):
    rec[f"l{l}_x"], rec[f"l{l}_level"], rec[f"l{l}_logits"] = x, level, logits
rec["anchors"] = anchors
common.save(a.out, "octree", rec, {"num_gaussians": a.num_gaussians, "levels": len(calls)})

with torch.no_grad(), common.Timer("gaussian decoder"):
    feats = dec_gs(anchors, latent)
    g = dec_gs.to_representation(anchors, feats)[0]
xyz = g.xyz.numpy()
eps = 1e-6
ply = {
    "ply_xyz": torch.from_numpy(np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)),
    "ply_f_dc": g.features_dc.transpose(1, 2).flatten(1),
    "ply_opacity": inverse_sigmoid(g.opacity.clamp(eps, 1 - eps)),
    "ply_scale": torch.log(g.scaling.clamp_min(eps)),
    "ply_rot": torch.from_numpy(quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), g.rotation.numpy())),
}
common.save(a.out, "gaussians", {"latent": latent, "anchors": anchors, "features": feats, "xyz": g.xyz, "features_dc": g.features_dc,
                                 "scaling": g.scaling, "rotation": g.rotation, "opacity": g.opacity, **ply})
g.save_ply(os.path.join(a.out, "reference.ply"))
