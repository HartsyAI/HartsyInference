"""Reference values for the Hunyuan3D-Paint schedule, samplers and guidance, produced by diffusers itself on the CPU.

Paint 2.1 builds `UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")` from the released
`scheduler/scheduler_config.json` (a DDIM config: scaled-linear betas 0.00085 to 0.012, v-prediction, `rescale_betas_zero_snr`) and runs 15 steps
(`utils/multiview_utils.py` of Tencent-Hunyuan/Hunyuan3D-2.1). Paint 2.0 uses `EulerAncestralDiscreteScheduler` with the same config and 30 steps. This script
runs both schedulers from diffusers over a fixed random sample with a fixed synthetic model, and transcribes the guidance of
`hunyuanpaintpbr/pipeline.py` (the `cam_mapping` view scale and the three-way combination) so the C# side has numbers to match.

    python dump_paint_sampler.py --scheduler-config <hunyuan3d-paint21>/scheduler/scheduler_config.json --out <fixture.safetensors>

The synthetic model is `v(x, t) = 0.5 tanh(x) - 0.2 x cos(t / 150) + 0.1 sin(t / 37)`, evaluated in float32 here and in the C# test, so the comparison covers the
scheduler arithmetic and not a network. Everything is stored as float32 (timesteps and sigmas included; a sigma of 4096 or an alpha of 6e-8 keeps 7 digits). Fixture keys: `alphas_cumprod`, `init_sample`, and per step count N in (15, 30) `unipc.<N>.timesteps`, `unipc.<N>.sigmas`,
`unipc.<N>.sample` (N + 1 rows: the sample before each step and after the last), `euler.<N>.*` the same with `euler.<N>.model_input`, `euler.<N>.noise` (the noise
each step adds, injected by replacing diffusers' `randn_tensor`) and `euler.init_noise_sigma`, and `guidance.*` (the three branch predictions, the azimuths, the view
scales and the guided result).
"""
import argparse
import json

import numpy as np
import torch
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler
from diffusers.schedulers import scheduling_euler_ancestral_discrete as euler_module
from safetensors.torch import save_file

SHAPE = (2, 4, 8, 8)


def model(x, t):
    return 0.5 * torch.tanh(x) - 0.2 * x * float(np.cos(t / 150.0)) + 0.1 * float(np.sin(t / 37.0))


def unipc(config, steps, sample0, out):
    scheduler = UniPCMultistepScheduler.from_config(config, timestep_spacing="trailing")
    scheduler.set_timesteps(steps)
    out[f"unipc.{steps}.timesteps"] = scheduler.timesteps.to(torch.float32)
    out[f"unipc.{steps}.sigmas"] = scheduler.sigmas.to(torch.float32)
    sample = sample0.clone()
    rows = [sample.clone()]
    for t in scheduler.timesteps:
        v = model(scheduler.scale_model_input(sample, t), float(t))
        sample = scheduler.step(v, t, sample, return_dict=False)[0]
        rows.append(sample.clone())
    out[f"unipc.{steps}.sample"] = torch.stack(rows).reshape(steps + 1, -1)


def euler(config, steps, sample0, noise_source, out):
    scheduler = EulerAncestralDiscreteScheduler.from_config(config, timestep_spacing="trailing")
    scheduler.set_timesteps(steps)
    out[f"euler.{steps}.timesteps"] = scheduler.timesteps.to(torch.float32)
    out[f"euler.{steps}.sigmas"] = scheduler.sigmas.to(torch.float32)
    out["euler.init_noise_sigma"] = torch.tensor([float(scheduler.init_noise_sigma)], dtype=torch.float32)
    noises = []
    queue = [torch.randn(SHAPE, generator=noise_source) for _ in range(steps)]

    def fake_randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
        n = queue[len(noises)]
        noises.append(n)
        return n.to(dtype) if dtype is not None else n

    original = euler_module.randn_tensor
    euler_module.randn_tensor = fake_randn_tensor
    try:
        sample = sample0.clone()
        rows, inputs = [sample.clone()], []
        for t in scheduler.timesteps:
            scaled = scheduler.scale_model_input(sample, t)
            inputs.append(scaled.clone())
            sample = scheduler.step(model(scaled, float(t)), t, sample, return_dict=False)[0]
            rows.append(sample.clone())
    finally:
        euler_module.randn_tensor = original
    out[f"euler.{steps}.sample"] = torch.stack(rows).reshape(steps + 1, -1)
    out[f"euler.{steps}.model_input"] = torch.stack(inputs).reshape(steps, -1)
    out[f"euler.{steps}.noise"] = torch.stack(noises).reshape(steps, -1)


def cam_mapping(azim):
    # hunyuanpaintpbr/pipeline.py, the nested function of the denoising loop
    if azim < 90 and azim >= 0:
        return float(azim) / 90.0 + 1
    elif azim >= 90 and azim < 330:
        return 2.0
    else:
        return -float(azim) / 90.0 + 5.0


def guidance(out, generator):
    # pipeline.py: noise_pred.chunk(3) -> uncond, ref, full; view_scale per (material, view); two accumulating terms
    n_pbr, views, scale = 2, 8, 3.0
    azims = [0, 45, 89, 90, 180, 329, 330, 345]
    branch = torch.randn((3 * n_pbr * views, 4, 8, 8), generator=generator)
    uncond, ref, full = branch.chunk(3)
    view_scale = torch.from_numpy(np.asarray([cam_mapping(a) for a in azims])).unsqueeze(0).repeat(n_pbr, 1).view(-1).to(uncond)[:, None, None, None]
    pred = uncond + scale * view_scale * (ref - uncond)
    pred += scale * view_scale * (full - ref)
    out["guidance.branches"] = branch.reshape(3, -1)
    out["guidance.azimuths"] = torch.tensor(azims, dtype=torch.float32)
    out["guidance.view_scale"] = view_scale.reshape(-1).to(torch.float32)
    out["guidance.result"] = pred.reshape(-1)
    out["guidance.scale"] = torch.tensor([scale], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.scheduler_config, encoding="utf-8") as handle:
        config = json.load(handle)

    generator = torch.Generator().manual_seed(0)
    out = {}
    # The table the samplers run on (UniPC and Euler ancestral both set the last entry to 2^-24; DDIMScheduler leaves it 0 and is not used by Paint).
    out["alphas_cumprod"] = UniPCMultistepScheduler.from_config(config).alphas_cumprod.to(torch.float32)
    sample0 = torch.randn(SHAPE, generator=generator)
    out["init_sample"] = sample0.reshape(-1)
    for steps in (15, 30):
        unipc(config, steps, sample0, out)
        euler(config, steps, sample0, generator, out)
    guidance(out, generator)
    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
