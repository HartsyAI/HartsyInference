"""Dumps ComfyUI sampler trajectories on a synthetic denoiser for SamplerParityTests.

Runs ComfyUI's own sample_* functions (the checkout passed as --comfy) against a deterministic nonlinear denoiser,
records every model-query sigma and every noise draw, and writes one JSON fixture the C# suite replays. The committed
fixture was generated against ComfyUI 0.37.0; regenerate it when a ComfyUI update changes a sampler.

    <comfy venv python> dump_k_samplers.py --comfy <ComfyUI dir> \
        --out ../HartsyInference.Diffusion.Tests/Fixtures/Samplers/k_sampler_parity.json
"""
import argparse
import inspect
import json
import sys

SHAPE = (1, 2, 4, 4)
STEPS = 8
GUIDANCE = 2.5
IMG2IMG_START = 2
FLOW_SHIFT = 3.0

SAMPLERS = [
    "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddpm", "ipndm", "ipndm_v",
    "deis", "res_multistep", "gradient_estimation", "er_sde", "seeds_2", "seeds_3", "sa_solver", "euler_cfg_pp",
    "uni_pc", "uni_pc_bh2",
]


def _stub_missing_optional_modules():
    """Stubs packages the sampler code never calls into, so a venv older than the checkout can still import it."""
    import importlib.abc
    import importlib.machinery
    import types

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            value = _Stub(f"{self.__name__}.{name}")
            setattr(self, name, value)
            return value

        def __call__(self, *a, **k):
            return None

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        prefixes = ("comfy_aimdo",)

        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in self.prefixes:
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return _Stub(spec.name)

        def exec_module(self, module):
            module.__path__ = []

    sys.meta_path.insert(0, _Finder())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.comfy)
    _stub_missing_optional_modules()

    import torch
    import comfy.model_sampling as ms
    import comfy.samplers as samplers
    from comfy.k_diffusion import sampling as kd
    from comfy.extra_samplers import uni_pc

    class EpsSampling(ms.ModelSamplingDiscrete, ms.EPS):
        pass

    class FlowSampling(ms.ModelSamplingDiscreteFlow, ms.CONST):
        pass

    gen = torch.Generator().manual_seed(1234)
    pattern = torch.randn(SHAPE, generator=gen)

    def denoise_pair(x, sigma):
        s = sigma.view(-1, 1, 1, 1)
        cond = x / (1 + s * s) + 0.3 * s / (1 + s) * torch.tanh(x) + 0.1 * pattern
        uncond = cond - 0.2 * s / (1 + s) * torch.tanh(0.5 * x) + 0.05 * pattern
        return cond, uncond

    class Patcher:
        def __init__(self, sampling):
            self.sampling = sampling

        def get_model_object(self, name):
            assert name == "model_sampling"
            return self.sampling

    class Inner:
        def __init__(self, sampling):
            self.model_sampling = sampling

    class Wrapped:
        def __init__(self, sampling):
            self.inner_model = Inner(sampling)
            self.model_patcher = Patcher(sampling)

    class Model:
        def __init__(self, sampling):
            self.inner_model = Wrapped(sampling)
            self.queries = []

        def __call__(self, x, sigma, **extra_args):
            self.queries.append(float(sigma.reshape(-1)[0]))
            cond, uncond = denoise_pair(x, sigma.reshape(-1))
            combined = uncond + GUIDANCE * (cond - uncond)
            for fn in extra_args.get("model_options", {}).get("sampler_post_cfg_function", []):
                combined = fn({"denoised": combined, "cond_denoised": cond, "uncond_denoised": uncond, "sigma": sigma,
                               "input": x, "model_options": extra_args.get("model_options", {})})
            return combined

    class RecordingNoise:
        def __init__(self, seed):
            self.gen = torch.Generator().manual_seed(seed)
            self.draws = []

        def __call__(self, sigma, sigma_next):
            n = torch.randn(SHAPE, generator=self.gen)
            self.draws.append({"sigma": float(sigma), "sigmaNext": float(sigma_next),
                               "data": [float(v) for v in n.flatten()]})
            return n

    def eps_sigmas():
        return kd.get_sigmas_karras(STEPS, 0.0291675, 14.614642)

    def flow_sigmas():
        t = torch.linspace(1.0, 0.0, STEPS + 1)
        s = FLOW_SHIFT * t / (1 + (FLOW_SHIFT - 1) * t)
        return s.float()

    cases = []
    for model_kind in ("eps", "flow"):
        sampling = EpsSampling() if model_kind == "eps" else FlowSampling()
        if model_kind == "flow":
            sampling.set_parameters(shift=FLOW_SHIFT)
        full = eps_sigmas() if model_kind == "eps" else flow_sigmas()
        percents = {str(p): float(sampling.percent_to_sigma(p)) for p in (1e-4, 0.2, 0.8)}
        for start in (0, IMG2IMG_START):
            for name in SAMPLERS:
                sigmas = full[start:].clone()
                x = torch.randn(SHAPE, generator=torch.Generator().manual_seed(99 + start))
                x = x * (float(sigmas[0]) if model_kind == "eps" else 1.0)
                x0 = x.clone()
                model = Model(sampling)
                noise = RecordingNoise(4321 + start)
                extra = {"model_options": {}, "seed": 7}
                if name == "dpm_fast":
                    sigma_min = sigmas[-1] if float(sigmas[-1]) != 0.0 else sigmas[-2]
                    out = kd.sample_dpm_fast(model, x, sigma_min, sigmas[0], len(sigmas) - 1, extra_args=extra, disable=True)
                elif name == "dpm_adaptive":
                    sigma_min = sigmas[-1] if float(sigmas[-1]) != 0.0 else sigmas[-2]
                    out = kd.sample_dpm_adaptive(model, x, sigma_min, sigmas[0], extra_args=extra, disable=True)
                elif name in ("uni_pc", "uni_pc_bh2"):
                    fn = uni_pc.sample_unipc if name == "uni_pc" else uni_pc.sample_unipc_bh2
                    work = sigmas.clone()
                    out = fn(model, x, work, extra_args=extra, disable=True)
                    out = sampling.inverse_noise_scaling(work[-1], out)
                else:
                    fn = getattr(kd, "sample_" + name)
                    kwargs = {}
                    if "noise_sampler" in inspect.signature(fn).parameters:
                        kwargs["noise_sampler"] = noise
                    out = fn(model, x, sigmas, extra_args=extra, disable=True, **kwargs)
                    out = sampling.inverse_noise_scaling(sigmas[-1], out)
                cases.append({
                    "sampler": name, "model": model_kind, "start": start,
                    "sigmas": [float(v) for v in full], "x0": [float(v) for v in x0.flatten()],
                    "noise": noise.draws, "queries": model.queries, "percents": percents,
                    "final": [float(v) for v in out.flatten()],
                })
                print(f"{model_kind:4s} start={start} {name:22s} evals={len(model.queries):3d} draws={len(noise.draws):2d}")

    fixture = {"shape": list(SHAPE), "guidance": GUIDANCE, "pattern": [float(v) for v in pattern.flatten()], "cases": cases}
    with open(args.out, "w") as f:
        json.dump(fixture, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
