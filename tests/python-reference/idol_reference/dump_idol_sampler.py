"""Short sampler runs on the CPU in float32 from the saved noise and condition, for the engine's sampling loops.

Usage: python -I dump_idol_sampler.py <denoiser dump dir> <out dir> [--only ...] [--budget 0.5]

Noise is denoiser_inputs.safetensors (latent, camera); the negative condition is zeros, as Idol's pipeline uses.
  idol                 idol.sampler.sample_flow, 4 steps, CFG 3, rescale_t 3 (the loop and grid, not the 50-step default)
  idol-turbo           the official phase-switched loop of idol-turbo's turbo.py (slow while t >= 0.5), 5 steps, no CFG
  budget3d-idol        budget3d.sampling.sampler.sample at --budget, 6 steps (layout built at step 0, rebuilt at step 5), CFG 3
  budget3d-idol-turbo  budget3d.sampling.turbo.sample_turbo at --budget, 5 steps (layout rebuilt every step)
Each writes <model>.safetensors with the final latent [1, 8192, 16] and camera [1, 1, 5]; the Budget3D runs also store the
final layout's indices. The turbo run's final latent is the input of dump_idol_decoders.py.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # common.py; python -I drops the script folder
import common

ap = argparse.ArgumentParser()
ap.add_argument("den")
ap.add_argument("out")
ap.add_argument("--only", default="idol,idol-turbo,budget3d-idol,budget3d-idol-turbo")
ap.add_argument("--budget", type=float, default=0.5)
a = ap.parse_args()
only = set(a.only.split(","))

inp = common.load_file(os.path.join(a.den, "denoiser_inputs.safetensors"))
z, cam = inp["latent"], inp["camera"]
cond = TensorDict({"feature1": inp["feature1"], "feature2": inp["feature2"]}, batch_size=1)
neg = cond.apply(torch.zeros_like)
comfy_dit = common.load_file(os.path.join(common.IDOL_DIR, "diffusion_models", "idol_fp16.safetensors"))
noise = lambda: TensorDict({"latent": z.clone(), "camera": cam.clone()}, batch_size=1)


def layout_tensors(L):
    return {"idx4": L.idx4, "idx2": L.idx2, "idx1": L.idx1, "k4k2": torch.tensor([L.k4, L.k2])}


if "idol" in only:
    from idol.sampler import sample_flow
    with common.Timer("idol 4 steps"):
        s = sample_flow(common.idol_denoiser(comfy_dit), noise(), cond, neg, steps=4, rescale_t=3.0, cfg_strength=3.0, verbose=False)
    common.save(a.out, "sampler-idol", {"latent": s["latent"], "camera": s["camera"]}, {"steps": 4, "cfg": 3.0, "rescale_t": 3.0})

if "idol-turbo" in only:
    spec = importlib.util.spec_from_file_location("idol_turbo_official", os.path.join(common.MODELS, "idol-turbo", "turbo.py"))
    turbo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(turbo)
    model = common.idol_denoiser(comfy_dit)
    turbo._inject(model, turbo.LORA_RANK, turbo.LORA_ALPHA)
    model.load_state_dict(common.f32(common.load_file(os.path.join(common.MODELS, "idol-turbo", "idol_turbo_lora.safetensors"))), strict=False)
    model.float()
    steps, rescale = 5, 3.0
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale * t_seq / (1 + (rescale - 1) * t_seq)
    s = noise()
    with torch.no_grad(), common.Timer("idol-turbo 5 steps"):
        for i in range(steps):     # the loop of turbo.py's turbo_sample_flow
            t, t_prev = t_seq[i], t_seq[i + 1]
            turbo._set_active(model, "slow" if t >= turbo.PHASE_THRESHOLD else "fast")
            v = model(s, torch.tensor([1000 * t], dtype=torch.float32), cond)
            s = TensorDict({k: s[k] - v[k] * (t - t_prev) for k in s.keys()}, batch_size=1)
    common.save(a.out, "sampler-idol-turbo", {"latent": s["latent"], "camera": s["camera"]}, {"steps": steps, "rescale_t": rescale})

if "budget3d-idol" in only:
    from budget3d.sampling.sampler import sample
    model = common.budget3d_model(os.path.join(common.MODELS, "budget3d-idol", "budget3d_idol.safetensors"))
    with common.Timer("budget3d-idol 6 steps"):
        lat, c, L = sample(model, cond, budget=a.budget, steps=6, cfg=3.0, rescale_t=3.0, device="cpu", noise=(z.clone(), cam.clone()))
    common.save(a.out, "sampler-budget3d-idol", {"latent": lat, "camera": c, **layout_tensors(L)},
                {"steps": 6, "cfg": 3.0, "budget": a.budget})

if "budget3d-idol-turbo" in only:
    from budget3d.sampling.turbo import sample_turbo
    model = common.budget3d_model(os.path.join(common.MODELS, "budget3d-idol-turbo", "budget3d_idol_turbo.safetensors"), turbo=True)
    with common.Timer("budget3d-idol-turbo 5 steps"):
        lat, c, L = sample_turbo(model, cond, budget=a.budget, steps=5, device="cpu", noise=(z.clone(), cam.clone()))
    common.save(a.out, "sampler-budget3d-idol-turbo", {"latent": lat, "camera": c, **layout_tensors(L)},
                {"steps": 5, "budget": a.budget})
