"""
DIAMOND (eloialonso/diamond) Atari parity oracle.

Runs the reference InnerModel (U-Net) + EDM preconditioning on the real Breakout checkpoint (denoiser.* keys)
and dumps the denoise IO + per-stage taps for the C# parity test.

  diamond_ref_io.safetensors:
    obs        [1,12,64,64]   4 context frames (3ch each), in [-1,1]
    act        [1,4]          4 context actions (int, as float)
    noisy      [1,3,64,64]    noisy next-obs
    sigma      [1]            noise level
    cond       [1,256]        conditioning vector (cond_proj(noise_emb + act_emb))
    innerout   [1,3,64,64]    raw inner_model output
    denoised   [1,3,64,64]    EDM-wrapped denoise output (pre-quantize + quantized)

Usage: python3 tests/python-reference/diamond_reference/dump_diamond_reference.py
Env: DIAMOND_CKPT (Breakout.pt), DIAMOND_SRC (/tmp/diamond/src)
"""
import os
import sys
import types
import importlib.machinery
import math
import torch
from safetensors.torch import save_file

CKPT = os.environ.get("DIAMOND_CKPT", "/tmp/diamond_ckpt/atari_100k/models/Breakout.pt")
SRC = os.environ.get("DIAMOND_SRC", "/tmp/diamond/src")
OUT = os.path.dirname(os.path.abspath(__file__))
SIGMA_DATA, SIGMA_OFFSET = 0.5, 0.3

sys.path.insert(0, SRC)
# Register minimal package stubs so models.diffusion.inner_model imports without the heavy __init__ (data/utils).
for name, sub in [("models", "models"), ("models.diffusion", "models/diffusion")]:
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(SRC, sub)]
    m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    sys.modules[name] = m
import importlib  # noqa: E402
inner = importlib.import_module("models.diffusion.inner_model")


def add_dims(x, n):
    return x.reshape(x.shape + (1,) * (n - x.ndim))


@torch.no_grad()
def main():
    torch.manual_seed(0)
    cfg = inner.InnerModelConfig(img_channels=3, num_steps_conditioning=4, cond_channels=256,
                                 depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0], num_actions=4)
    model = inner.InnerModel(cfg).eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k[len("denoiser.inner_model."):]: v for k, v in ck.items() if k.startswith("denoiser.inner_model.")}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("missing:", miss[:6], "unexpected:", unexp[:6])

    obs = (torch.rand(1, 12, 64, 64) * 2 - 1)      # 4 frames x 3ch
    act = torch.randint(0, 4, (1, 4))
    noisy = torch.randn(1, 3, 64, 64)
    sigma = torch.tensor([1.3])

    # EDM conditioners (denoiser.compute_conditioners)
    s = (sigma**2 + SIGMA_OFFSET**2).sqrt()
    c_in = add_dims(1 / (s**2 + SIGMA_DATA**2).sqrt(), 4)
    c_skip = add_dims(SIGMA_DATA**2 / (s**2 + SIGMA_DATA**2), 4)
    c_out = add_dims(s * (SIGMA_DATA**2 / (s**2 + SIGMA_DATA**2)).sqrt(), 4)
    c_noise = s.log() / 4                            # [1]

    cond = model.cond_proj(model.noise_emb(c_noise) + model.act_emb(act))
    innerout = model(noisy * c_in, c_noise, obs / SIGMA_DATA, act)
    d = c_skip * noisy + c_out * innerout
    denoised = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)

    # --- full EDM sampler (Karras rho + Euler, s_churn=0), fixed init noise ---
    NUM_STEPS, SIGMA_MIN, SIGMA_MAX, RHO = 3, 2e-3, 5.0, 7
    min_inv, max_inv = SIGMA_MIN ** (1 / RHO), SIGMA_MAX ** (1 / RHO)
    ll = torch.linspace(0, 1, NUM_STEPS)
    sigmas = (max_inv + ll * (min_inv - max_inv)) ** RHO
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

    def denoise(xx, sg):
        s2 = sg**2 + SIGMA_OFFSET**2
        ci = 1 / (s2 + SIGMA_DATA**2).sqrt()
        csk = SIGMA_DATA**2 / (s2 + SIGMA_DATA**2)
        co = s2.sqrt() * csk.sqrt()
        cn = s2.sqrt().log() / 4
        out = model(xx * ci, cn, obs / SIGMA_DATA, act)
        dd = csk * xx + co * out
        return dd.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)

    xinit = torch.randn(1, 3, 64, 64)
    x = xinit.clone()
    for sg, nsg in zip(sigmas[:-1], sigmas[1:]):
        den = denoise(x, sg.view(1))
        dcur = (x - den) / sg
        x = x + dcur * (nsg - sg)
    sampled = x

    refs = {"obs": obs, "act": act.float(), "noisy": noisy, "sigma": sigma,
            "cond": cond, "innerout": innerout, "denoised_raw": d, "denoised": denoised,
            "sampler_xinit": xinit, "sampler_out": sampled}
    refs = {k: v.float().contiguous() for k, v in refs.items()}
    os.makedirs(OUT, exist_ok=True)
    save_file(refs, os.path.join(OUT, "diamond_ref_io.safetensors"))
    for k, v in refs.items():
        print(f"  {k:12s} {tuple(v.shape)}")
    print("wrote diamond_ref_io.safetensors ->", OUT)


if __name__ == "__main__":
    main()
