"""One denoiser forward per model on the CPU in float32, at the dense (8192-token) layout, with stage taps for bisecting.

Usage: python -I dump_idol_denoiser.py <conditioning dir> <out dir> [--only idol,idol-turbo,budget3d-idol,budget3d-idol-turbo]

Inputs (seed 1234): latent [1, 8192, 16] and camera [1, 1, 5] from torch.randn; the condition is feature1/feature2 from
dump_idol_conditioning.py. Writes denoiser_inputs.safetensors and one <model>.safetensors per model:
  idol                 Comfy-Org idol_fp16 (cast to float32) + pinned pos_pe, t = 731
  idol-turbo           the same base + speridlabs idol_turbo_lora through the official turbo.py, slow t = 800 and fast t = 200;
                       also checks that the Comfy-format slow/fast LoRA files hold the same deltas
  budget3d-idol        BudgetAny budget3d_idol (dense layout), t = 731
  budget3d-idol-turbo  BudgetAny budget3d_idol_turbo (dense layout), slow t = 800 and fast t = 200
Taps (idol and budget3d-idol): tokens after input+position embedding, after the noise refiner, the context stream after the
context refiner, the RePo rotation (cos, sin) of joint block 0, the joint stream after blocks 0 and 23.
"""
import argparse
import importlib.util
import os
import sys

import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # common.py; python -I drops the script folder
import common

ap = argparse.ArgumentParser()
ap.add_argument("cond")
ap.add_argument("out")
ap.add_argument("--only", default="idol,idol-turbo,budget3d-idol,budget3d-idol-turbo")
a = ap.parse_args()
only = set(a.only.split(","))

c = common.load_file(os.path.join(a.cond, "conditioning.safetensors"))
g = torch.Generator().manual_seed(1234)
latent = torch.randn(1, 8192, 16, generator=g)
camera = torch.randn(1, 1, 5, generator=g)
x = TensorDict({"latent": latent, "camera": camera}, batch_size=1)
cond = TensorDict({"feature1": c["feature1"], "feature2": c["feature2"]}, batch_size=1)
common.save(a.out, "denoiser_inputs", {"latent": latent, "camera": camera, "feature1": c["feature1"], "feature2": c["feature2"]})


def taps(model):
    """Forward hooks on the stages named in the docstring; returns (dict, remove)."""
    got, hs = {}, []

    def hook(name, pick=lambda o: o):
        def f(_m, _i, o):
            got[name] = pick(o).detach().float().clone()
        return f
    hs.append(model.noise_refiner[-1].register_forward_hook(hook("after_noise_refiner")))
    hs.append(model.context_refiner[-1].register_forward_hook(hook("after_context_refiner")))
    # [1, L, 16, 32, 2]: (cos, sin) of every rotation angle, whether RePo returned complex numbers or real pairs
    hs.append(model.repo_layers[0].register_forward_hook(hook("repo0_cos_sin", lambda o: torch.view_as_real(o) if o.is_complex() else o)))
    hs.append(model.blocks[0].register_forward_hook(hook("after_block0")))
    hs.append(model.blocks[-1].register_forward_hook(hook("after_block23")))
    hs.append(model.noise_refiner[0].register_forward_pre_hook(
        lambda _m, args, kwargs=None: got.__setitem__("embedded_tokens", args[0].detach().float().clone())))
    return got, lambda: [h.remove() for h in hs]


def run(model, t, tap=False):
    got, remove = taps(model) if tap else ({}, lambda: None)
    with torch.no_grad(), common.Timer(f"forward t={t}"):
        out = model(x, torch.full((1,), float(t)), cond)
    remove()
    return {"v_latent": out["latent"], "v_camera": out["camera"], **got}


def official_turbo():
    spec = importlib.util.spec_from_file_location("idol_turbo_official", os.path.join(common.MODELS, "idol-turbo", "turbo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


comfy_dit = common.load_file(os.path.join(common.IDOL_DIR, "diffusion_models", "idol_fp16.safetensors"))

if "idol" in only:
    common.save(a.out, "idol", run(common.idol_denoiser(comfy_dit), 731, tap=True), {"t": 731})

if "idol-turbo" in only:
    turbo = official_turbo()
    model = common.idol_denoiser(comfy_dit)
    turbo._inject(model, 32, 128.0)
    lora = common.f32(common.load_file(os.path.join(common.MODELS, "idol-turbo", "idol_turbo_lora.safetensors")))
    missing, unexpected = model.load_state_dict(lora, strict=False)
    assert not unexpected, unexpected[:4]
    model.float()
    res = {}
    for phase, t in (("slow", 800), ("fast", 200)):
        turbo._set_active(model, phase)
        r = run(model, t)
        res[f"{phase}_v_latent"], res[f"{phase}_v_camera"] = r["v_latent"], r["v_camera"]
    # the Comfy-format files: lora_up @ lora_down must equal (alpha / rank) * B @ A of the same expert
    worst = 0.0
    for phase in ("slow", "fast"):
        comfy = common.f32(common.load_file(os.path.join(common.MODELS, "idol-turbo", "loras", f"idol_turbo_{phase}_comfy.safetensors")))
        for k in [k for k in comfy if k.endswith(".lora_down.weight")]:
            base = k[: -len(".lora_down.weight")]
            d_comfy = comfy[base + ".lora_up.weight"] @ comfy[k]
            d_off = 4.0 * lora[f"{base}.{phase}_B"] @ lora[f"{base}.{phase}_A"]
            worst = max(worst, float((d_comfy - d_off).abs().max() / d_off.abs().max().clamp_min(1e-12)))
    print(f"comfy-format LoRA vs idol_turbo_lora: worst relative max |delta difference| = {worst:.3e}")
    common.save(a.out, "idol-turbo", res, {"t_slow": 800, "t_fast": 200, "comfy_lora_rel_diff": worst})

if "budget3d-idol" in only:
    model = common.budget3d_model(os.path.join(common.MODELS, "budget3d-idol", "budget3d_idol.safetensors"))
    common.save(a.out, "budget3d-idol", run(model, 731, tap=True), {"t": 731})

if "budget3d-idol-turbo" in only:
    model = common.budget3d_model(os.path.join(common.MODELS, "budget3d-idol-turbo", "budget3d_idol_turbo.safetensors"), turbo=True)
    res = {}
    for phase, t in (("slow", 800), ("fast", 200)):
        r = run(model, t)
        res[f"{phase}_v_latent"], res[f"{phase}_v_camera"] = r["v_latent"], r["v_camera"]
    common.save(a.out, "budget3d-idol-turbo", res, {"t_slow": 800, "t_fast": 200})
