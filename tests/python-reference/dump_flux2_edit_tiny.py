"""Tiny diffusers Flux2 reference-edit fixture for the C# parity tests (Flux2EditParityTests).

Builds a random tiny diffusers `Flux2Transformer2DModel` (hidden 32, 1 double + 2 single blocks, the layout and key
names the Open Vision Banana release ships), runs it the way `Flux2KleinPipeline` does with one reference image
(target ids T=0, reference ids T=10, shared timestep, CFG against a second prompt, FlowMatchEuler with the base repo's
scheduler config and compute_empirical_mu), and saves the weights, inputs, per-call velocities, per-step latents and
the sigma schedules. Run in a CPU container with torch, safetensors and diffusers (`pip install 'diffusers>=0.37'`; the project's `hartsy-ref:cpu` image is local to
the maintainers' machine, any equivalent works):

    python tests/python-reference/dump_flux2_edit_tiny.py --out tests/HartsyInference.Diffusion.Tests/Fixtures/flux2_edit_tiny

Everything is float32 on CPU. meta.json records the torch and diffusers versions used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

import diffusers
from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

# The FLUX.2 klein base repos' scheduler_config.json (black-forest-labs/FLUX.2-klein-base-4B @ a3b4f48).
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

PAT_H, PAT_W, CHANNELS, CONTEXT, TEXT_LEN = 2, 3, 8, 12, 5
STEPS, GUIDANCE = 3, 3.5


def timesteps_and_sigmas(tokens: int, steps: int):
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu = compute_empirical_mu(image_seq_len=tokens, num_steps=steps)
    scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
    return scheduler, mu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)

    model = Flux2Transformer2DModel(
        patch_size=1, in_channels=CHANNELS, num_layers=1, num_single_layers=2, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=CONTEXT, timestep_guidance_channels=256, mlp_ratio=3.0,
        axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, eps=1e-6, guidance_embeds=False,
    ).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn_like(p) * 0.3)
            else:  # RMSNorm scales
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "weights.safetensors"))

    grid = PAT_H * PAT_W
    noise = torch.randn(1, CHANNELS, PAT_H, PAT_W)                    # unpacked, as TextToImageRequest.InitialNoise
    ref_tokens = torch.randn(1, grid, CHANNELS)                       # the packed reference latent
    cond = torch.randn(1, TEXT_LEN, CONTEXT)
    uncond = torch.randn(1, TEXT_LEN, CONTEXT)

    def ids(times):
        rows = [torch.tensor([[0, 0, 0, i] for i in range(TEXT_LEN)], dtype=torch.float32)]
        for t in times:
            rows_img = [[t, h, w, 0] for h in range(PAT_H) for w in range(PAT_W)]
            rows.append(torch.tensor(rows_img, dtype=torch.float32))
        return rows[0].unsqueeze(0), torch.cat(rows[1:], dim=0).unsqueeze(0)

    txt_ids, img_ids = ids([0, 10])
    _, img_ids_target_only = ids([0])

    def velocity(latents, text, sigma, with_ref=True):
        x = torch.cat([latents, ref_tokens], dim=1) if with_ref else latents
        i = img_ids if with_ref else img_ids_target_only
        t = torch.tensor([sigma], dtype=torch.float32)
        with torch.no_grad():
            v = model(hidden_states=x, encoder_hidden_states=text, timestep=t, img_ids=i, txt_ids=txt_ids,
                      guidance=None, return_dict=False)[0]
        return v

    tensors = {"noise": noise, "ref_tokens": ref_tokens, "cond": cond, "uncond": uncond,
               "txt_ids": txt_ids, "img_ids": img_ids}

    # One raw call at a fixed sigma, with and without the reference (full output incl. reference rows).
    packed_noise = noise.flatten(2).transpose(1, 2).contiguous()      # == _pack_latents on a [1,C,H,W] latent
    tensors["velocity_ref_sigma0p6"] = velocity(packed_noise, cond, 0.6)
    tensors["velocity_noref_sigma0p6"] = velocity(packed_noise, cond, 0.6, with_ref=False)

    # The same call in float64: the float32 noise floor of this tiny model, so a C# difference can be told from rounding.
    model64 = Flux2Transformer2DModel.from_config(model.config).double().eval()
    model64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    with torch.no_grad():
        v64 = model64(hidden_states=torch.cat([packed_noise, ref_tokens], dim=1).double(), encoder_hidden_states=cond.double(),
                      timestep=torch.tensor([0.6], dtype=torch.float64), img_ids=img_ids.double(), txt_ids=txt_ids.double(),
                      guidance=None, return_dict=False)[0]
    tensors["velocity_ref_sigma0p6_f64"] = v64.float()
    f32_vs_f64 = float(((tensors["velocity_ref_sigma0p6"] - v64.float()).norm() / v64.float().norm()))

    # The pipeline loop: CFG (neg + g * (pos - neg)), only the target rows stepped, FlowMatchEuler.
    scheduler, mu = timesteps_and_sigmas(grid, STEPS)
    latents = packed_noise.clone()
    for i, t in enumerate(scheduler.timesteps):
        sigma = float(t) / 1000.0
        pos = velocity(latents, cond, sigma)[:, :grid]
        neg = velocity(latents, uncond, sigma)[:, :grid]
        noise_pred = neg + GUIDANCE * (pos - neg)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        tensors[f"latents_step_{i}"] = latents.clone()

    # Schedules the C# scheduler must reproduce: (target tokens, steps) -> mu and sigmas (len steps + 1).
    schedules = {}
    for tokens, steps in [(grid, STEPS), (4096, 28), (1024, 28), (2025, 20)]:
        sched, m = timesteps_and_sigmas(tokens, steps)
        schedules[f"{tokens}_{steps}"] = {"mu": float(m), "sigmas": [float(s) for s in sched.sigmas],
                                          "timesteps": [float(t) for t in sched.timesteps]}

    save_file({k: v.contiguous().float() for k, v in tensors.items()}, str(out / "reference.safetensors"))
    (out / "meta.json").write_text(json.dumps({
        "torch": torch.__version__, "diffusers": diffusers.__version__, "patH": PAT_H, "patW": PAT_W,
        "channels": CHANNELS, "context": CONTEXT, "textLen": TEXT_LEN, "steps": STEPS, "guidance": GUIDANCE,
        "mu": float(mu), "velocityF32VsF64RelRms": f32_vs_f64, "schedules": schedules, "referenceTimes": [0, 10],
        "note": "random weights; hidden 32, 1 double + 2 single blocks, heads 2 x 16, axes [4,4,4,4], rope_theta 2000",
    }, indent=1))
    print("wrote", out, "diffusers", diffusers.__version__)


if __name__ == "__main__":
    main()
