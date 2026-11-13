"""Reference values for Hunyuan3D-Omni's sampling loop: upstream's own `Sampler.sample_ode` (transport) with its `denoise_with_cfg`, on the CPU, with a toy velocity.

Runs `hy3dshape/models/diffusion/transport` of a checkout of Tencent-Hunyuan/Hunyuan3D-Omni (the one the other Omni oracle uses) exactly as the released
`scheduler/config.json` builds it (`create_transport(path_type="Linear", prediction="velocity", train_sample_type="logit_normal", mean=0, std=1, shift_scale=0.125)`,
`Sampler(transport)`, `sample_ode(sampling_method="euler", num_steps=50)`) and the classifier-free guidance of `pipeline_generation_sit_omni.py`
(`denoise_with_cfg`: both halves of the batch get `uncond + guidance * (cond - uncond)`; the latent is duplicated first and the first half is kept).

`torchdiffeq` is not installed here, so the one function the sampler takes from it, `odeint(..., method="euler")`, is a transcription of torchdiffeq's fixed-grid Euler:
over the time grid `t` (a float32 `linspace(t0, t1, num_steps)`), `y1 = y0 + (t1 - t0) * f(t0, y0)` for every interval, the solution stacking the states. Nothing else of
the transport package is replaced (`tqdm` is a stand-in because only the SDE sampler uses it).

    python dump_omni_flow.py --upstream <Hunyuan3D-Omni checkout>/hy3dshape --out <fixture.npz>

The toy velocity is `v(x, t, c) = (0.3 c - 0.5 t) x + 0.1 c t^2` with `c` 1 for the conditional half and 0.3 for the unconditional half, so only the sampler and the guidance
are compared, not a network. Fixture keys: `init` (the latent, `[1, 16, 8]`), `t` (the 50 time points the sampler integrates over), `t0` and `t1`, `guidance` (4.5, upstream's
`inference.py` value), `final` (the guided latent after the loop) and `trajectory` (the 50 states, the last being `final`).
"""
import argparse
import importlib.util
import os
import sys
import types

import torch


def euler_odeint(func, y0, t, atol=None, rtol=None, method=None, **_):
    """torchdiffeq's fixed-grid euler: y1 = y0 + (t1 - t0) * f(t0, y0), the states of every grid point stacked."""
    assert method == "euler", method
    solution = [y0]
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0
        y0 = y0 + dt * func(t0, y0)
        solution.append(y0)
    return torch.stack(solution)


def load_transport(root):
    """Loads hy3dshape/models/diffusion/transport as a package without running hy3dshape's own __init__ files."""
    torchdiffeq = types.ModuleType("torchdiffeq")
    torchdiffeq.odeint = euler_odeint
    sys.modules["torchdiffeq"] = torchdiffeq
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable=None, **kw: iterable
    sys.modules["tqdm"] = tqdm
    for name, sub in (("hy3dshape", ""), ("hy3dshape.models", "models"), ("hy3dshape.models.diffusion", "models/diffusion")):
        package = types.ModuleType(name)
        package.__path__ = [os.path.join(root, sub)] if sub else [root]
        sys.modules[name] = package
    full = "hy3dshape.models.diffusion.transport"
    path = os.path.join(root, "models", "diffusion", "transport")
    spec = importlib.util.spec_from_file_location(full, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    import numpy as np

    transport = load_transport(args.upstream)
    state = transport.create_transport(path_type="Linear", prediction="velocity", train_sample_type="logit_normal", mean=0, std=1.0, shift_scale=0.125)
    scheduler = transport.Sampler(state)
    steps, guidance = 50, 4.5
    sample_fn = scheduler.sample_ode(num_steps=steps, sampling_method="euler")

    def model(inputs, ts, contexts, guidance=None):
        c = contexts["cond"].reshape(-1, 1, 1)  # [2, 1, 1]: 1.0 for the conditional half, 0.3 for the unconditional half
        t = ts.reshape(-1, 1, 1)
        return (0.3 * c - 0.5 * t) * inputs + 0.1 * c * t * t

    def denoise_with_cfg(inputs, ts, contexts):  # pipeline_generation_sit_omni.py, lines 271-277
        noise_pred = model(inputs, ts, contexts, guidance=None)
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
        return torch.cat([noise_pred, noise_pred], dim=0)

    generator = torch.Generator().manual_seed(7)
    init = torch.randn(1, 16, 8, generator=generator)
    contexts = {"cond": torch.tensor([1.0, 0.3])}
    latents = torch.cat([init, init])
    trajectory = sample_fn(latents, denoise_with_cfg, contexts=contexts)
    final = trajectory[-1].chunk(2, dim=0)[0]
    t0, t1 = state.check_interval(state.train_eps, state.sample_eps, sde=False, eval=True, reverse=False, last_step_size=0.0)
    grid = torch.linspace(t0, t1, steps)
    arrays = {
        "init": init, "t": grid, "t0": torch.tensor([float(t0)]), "t1": torch.tensor([float(t1)]),
        "guidance": torch.tensor([guidance]), "final": final, "trajectory": trajectory.chunk(2, dim=1)[0],
    }
    np.savez(args.out, **{k: v.contiguous().numpy().astype(np.float32) for k, v in arrays.items()})
    print(f"interval [{t0}, {t1}], {steps} points, final RMS {final.pow(2).mean().sqrt():.4f}; wrote {args.out}")


if __name__ == "__main__":
    main()
