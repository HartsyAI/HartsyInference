"""Reference values for the X-Part flow loop: upstream's own scheduler and denoising loop on the CPU, with a toy velocity in place of the DiT.

Runs `FlowMatchEulerDiscreteScheduler` of `XPart/partgen/models/diffusion/schedulers.py` and the schedule, timestep and `scheduler.step` arithmetic of `XPartPipeline.__call__`
(`partformer_pipeline.py`, Tencent-Hunyuan/Hunyuan3D-Part, pinned in docs/Research/HUNYUAN3D_PART_ARCHITECTURE.md): `sigmas = np.linspace(0, 1, N)`, `retrieve_timesteps`, `timestep = t / 1000`
handed to the model, `latents = scheduler.step(v, t, latents).prev_sample`, in float32. The DiT is replaced by a fixed function of (latents, timestep) so the loop can be compared without the 6.6 GB
model: `v = -0.5 * x + 0.3 * sin(3 * timestep) + 0.1 * cos(7 * x)`. The three diffusers classes the scheduler file imports are replaced by the few lines it needs from them.

    python -I dump_xpart_flow.py --upstream <Hunyuan3D-Part checkout>/XPart/partgen --out <fixture.npz>

The fixture holds, for 50 steps and for 7 steps, the `noise` (parts x 1024 x 64), the `timesteps` given to the model (`timestep`, after the division by 1000), the scheduler's `sigmas`
(with the trailing 1) and the final `latents`.
"""
import argparse
import importlib.util
import os
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch


def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


class SchedulerMixin:
    pass


class ConfigMixin:
    pass


class BaseOutput:
    pass


def register_to_config(init):
    def wrapped(self, *args, **kwargs):
        import inspect

        names = list(inspect.signature(init).parameters)[1:]
        values = dict(zip(names, args))
        values.update(kwargs)
        defaults = {k: v.default for k, v in inspect.signature(init).parameters.items() if v.default is not inspect.Parameter.empty}
        self.config = types.SimpleNamespace(**{**defaults, **values})
        init(self, *args, **kwargs)

    return wrapped


def retrieve_timesteps(scheduler, num_inference_steps, device, sigmas):
    # The sigmas branch of retrieve_timesteps in partformer_pipeline.py.
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps, len(scheduler.timesteps)


def velocity(x, timestep):
    return -0.5 * x + 0.3 * torch.sin(3 * timestep) + 0.1 * torch.cos(7 * x)


def run(scheduler_module, steps, noise):
    scheduler = scheduler_module.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    sigmas = np.linspace(0, 1, steps)
    timesteps, steps = retrieve_timesteps(scheduler, steps, "cpu", sigmas)
    latents = noise.clone()
    given = []
    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        timestep = timestep / scheduler.config.num_train_timesteps
        given.append(float(timestep[0]))
        v = velocity(latents, timestep.reshape(-1, 1, 1))
        latents = scheduler.step(v, t, latents).prev_sample
    return np.array(given, dtype=np.float32), scheduler.sigmas.numpy(), latents.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="the XPart/partgen folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    stub("diffusers"), stub("diffusers.configuration_utils", ConfigMixin=ConfigMixin, register_to_config=register_to_config)
    stub("diffusers.schedulers"), stub("diffusers.schedulers.scheduling_utils", SchedulerMixin=SchedulerMixin)
    import logging
    stub("diffusers.utils", BaseOutput=BaseOutput, logging=types.SimpleNamespace(get_logger=lambda name: logging.getLogger(name)))

    path = os.path.join(args.upstream, "models", "diffusion", "schedulers.py")
    spec = importlib.util.spec_from_file_location("partgen_schedulers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["partgen_schedulers"] = module
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    noise = torch.randn(3, 1024, 64)
    out = {"noise": noise.numpy()}
    for steps in (50, 7):
        timesteps, sigmas, latents = run(module, steps, noise)
        out[f"timesteps{steps}"], out[f"sigmas{steps}"], out[f"latents{steps}"] = timesteps, sigmas, latents
        print(f"{steps} steps: timesteps {timesteps[:3]}..{timesteps[-2:]}, sigmas ends {sigmas[-3:]}, latents std {latents.std():.4f}")
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
