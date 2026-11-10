# Regenerates the sana_wm_schedule_*.json fixtures from the pinned diffusers venv.
# Run: CUDA_VISIBLE_DEVICES= <venv>/bin/python generate_sana_wm_schedule_fixtures.py <Fixtures dir>
import json
import os
import sys

import numpy as np
import torch
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be set empty"
OUT_DIR = sys.argv[1]
CASES = [(60, 9.8), (60, 8.0), (30, 9.8), (1, 9.8), (2, 9.8)]


def float64_replica(steps, shift):
    """Re-derives set_timesteps in float64 numpy, before the final float32 cast."""
    probe = FlowMatchEulerDiscreteScheduler(shift=shift)
    start = probe.sigma_max * 1000
    stop = probe.sigma_min * 1000
    ts = np.linspace(start, stop, steps)
    sig = ts / 1000
    sig = shift * sig / (1 + (shift - 1) * sig)
    return probe.sigma_max, probe.sigma_min, start, stop, sig


for steps, shift in CASES:
    sched = FlowMatchEulerDiscreteScheduler(shift=shift)
    timesteps, n = retrieve_timesteps(sched, steps, "cpu", None)
    assert n == steps
    sigmas32 = sched.sigmas.detach().cpu().numpy().astype(np.float32)
    ts32 = timesteps.detach().cpu().numpy().astype(np.float32)
    assert sigmas32.shape == (steps + 1,) and sigmas32[-1] == 0.0
    assert ts32.shape == (steps,)

    smax, smin, start, stop, sig64 = float64_replica(steps, shift)
    assert np.array_equal(sig64.astype(np.float32), sigmas32[:steps]), "float64 replica does not reproduce float32 sigmas"
    assert np.array_equal(ts32, sigmas32[:steps] * np.float32(1000.0)), "timesteps are not sigma*1000 in float32"

    # Sign-convention probes: one scalar-path step, and a per-token step as flow_euler_ltx calls it (-noise_pred in).
    x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32)
    v = torch.linspace(0.5, -0.25, 8, dtype=torch.float32)
    probe = FlowMatchEulerDiscreteScheduler(shift=shift)
    probe.set_timesteps(steps, device="cpu")
    t0 = probe.timesteps[0]
    s0, s1 = float(probe.sigmas[0]), float(probe.sigmas[1])
    std_out = probe.step(v, t0, x, return_dict=False)[0]
    std_expect = x + (s1 - s0) * v
    probe2 = FlowMatchEulerDiscreteScheduler(shift=shift)
    probe2.set_timesteps(steps, device="cpu")
    ltx_out = probe2.step(-v.reshape(1, 8, 1), t0, x.reshape(1, 8, 1),
                          per_token_timesteps=torch.full((1, 8), float(t0)), return_dict=False)[0].reshape(8)
    sign_std = float((std_out - std_expect).abs().max())
    sign_ltx = float((ltx_out - std_expect).abs().max())

    doc = {
        "description": "SANA-WM flow_euler_ltx sigma schedule, produced by diffusers FlowMatchEulerDiscreteScheduler"
                       " .set_timesteps via retrieve_timesteps (stable_diffusion_3 pipeline), CPU, no shift_terminal.",
        "generator": "tests/HartsyInference.Diffusion.Tests/Fixtures/generate_sana_wm_schedule_fixtures.py",
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "steps": steps,
        "shift": shift,
        "endpoints": {
            "init_sigma_max_float32": float(smax),
            "init_sigma_min_float32": float(smin),
            "linspace_start_float64": float(start),
            "linspace_stop_float64": float(stop),
        },
        "sigmas_float64": [float(v) for v in sig64] + [0.0],
        "sigmas_float32": [float(v) for v in sigmas32],
        "timesteps_float32": [float(v) for v in ts32],
        "sign_probe_max_abs_error": {"scalar_step_vs_x_plus_dsigma_v": sign_std,
                                     "per_token_ltx_step_vs_x_plus_dsigma_v": sign_ltx},
    }
    name = f"sana_wm_schedule_steps{steps}_shift{str(shift).replace('.', '')}.json"
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(name, "last nonzero sigma", sigmas32[steps - 1], "sigma0", sigmas32[0],
          "sigma_max/min32", smax, smin, "sign err", sign_std, sign_ltx, "n", steps)
