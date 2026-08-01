"""SeedVR2-3B Python reference benchmark (Part G).

Times the reference pipeline END-TO-END (preprocess -> VAE encode -> NaDiT 1 step -> VAE decode) on a
frames dir, at the same target area as the C# matrix runs. Runs in the reference's PRODUCTION shape:
bf16 weights/activations + causal slicing (the fp32 whole-clip path needs ~18+ GB of activations at
720p-area and OOMs a 24 GB card — see run_seedvr2_e2e_reference.py). Models load once; then 1 discarded
warmup + N timed trials (repo contract: N=5, 95% CI via Student-t).

The DiT state dict loads from the converted safetensors (the .pth was deleted for disk hygiene;
identical tensors). flash_attn/apex shims as in the E2E driver.

Usage: <venv-python> bench_seedvr2_python.py <SeedVR-checkout> <weights-dir> <frames-dir> [trials=5] [area=518400]
"""
import importlib.machinery
import os
import sys
import time
import types

import torch

def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                 dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        s, e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        qi = q[s:e].transpose(0, 1).unsqueeze(0)
        ki = k[s:e].transpose(0, 1).unsqueeze(0)
        vi = v[s:e].transpose(0, 1).unsqueeze(0)
        oi = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)
        outs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outs)

_fa = types.ModuleType("flash_attn")
_fa.flash_attn_varlen_func = _sdpa_varlen
_fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
sys.modules["flash_attn"] = _fa

from diffusers.models.normalization import RMSNorm  # noqa: E402


class _FusedRMSNorm(RMSNorm):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


class _FusedLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kw):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


_apex = types.ModuleType("apex")
_apex_norm = types.ModuleType("apex.normalization")
_apex_norm.FusedRMSNorm = _FusedRMSNorm
_apex_norm.FusedLayerNorm = _FusedLayerNorm
_apex.normalization = _apex_norm
_apex.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
_apex_norm.__spec__ = importlib.machinery.ModuleSpec("apex.normalization", loader=None)
sys.modules["apex"] = _apex
sys.modules["apex.normalization"] = _apex_norm

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from models.dit_v2 import na, rope as _rope_mod  # noqa: E402
from models.dit_v2.nadit import NaDiT  # noqa: E402
from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper  # noqa: E402
from data.image.transforms.area_resize import AreaResize  # noqa: E402
from data.image.transforms.divisible_crop import DivisibleCrop  # noqa: E402
from run_seedvr2_e2e_reference import DIT_ARGS, VAE_CFG, _get_freqs_small  # noqa: E402

_rope_mod.NaMMRotaryEmbedding3d.get_freqs = _get_freqs_small

WEIGHTS, FRAMES = sys.argv[2], sys.argv[3]
TRIALS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
AREA = float(sys.argv[5]) if len(sys.argv) > 5 else 518400.0
SCALE = 0.9152
DEV = "cuda"


def load_frames() -> torch.Tensor:
    files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".png"))
    return torch.stack([
        torch.from_numpy(np.array(Image.open(os.path.join(FRAMES, f)).convert("RGB")))
        for f in files]).permute(0, 3, 1, 2).float() / 255.0


def run_once(vae, dit, pos, video) -> tuple:
    x = AreaResize(max_area=AREA, downsample_only=False)(video)
    x = torch.clamp(x, 0.0, 1.0)
    x = DivisibleCrop((16, 16))(x)
    x = ((x - 0.5) / 0.5).permute(1, 0, 2, 3).contiguous().to(DEV, torch.bfloat16)
    with torch.no_grad():
        posterior = vae.encode(x.unsqueeze(0)).posterior
        z = posterior.sample() * SCALE
        _, _, lt, lh, lw = z.shape
        noise = torch.randn_like(z)
        z_last = z[0].permute(1, 2, 3, 0)
        cond = torch.cat([z_last, torch.ones(lt, lh, lw, 1, device=DEV, dtype=z_last.dtype)], dim=-1)
        dit_in = torch.cat([noise[0].permute(1, 2, 3, 0), cond], dim=-1)
        # Production dit_offload behavior: VAE off-device around the DiT forward (co-resident bf16
        # vae+dit peaked 18.2 GB and OOM'd the warmup).
        vae.to("cpu")
        torch.cuda.empty_cache()
        vid_flat, vid_shape = na.flatten([dit_in])
        txt_flat, txt_shape = na.flatten([pos])
        v = dit(vid=vid_flat, txt=txt_flat, vid_shape=vid_shape, txt_shape=txt_shape,
                timestep=torch.tensor([1000.0], device=DEV)).vid_sample
        vae.to(DEV)
        x0 = (noise[0].permute(1, 2, 3, 0).reshape(-1, 16) - v) / SCALE
        x0 = x0.T.reshape(1, 16, lt, lh, lw).to(torch.bfloat16)
        pixels = vae.decode(x0)[0]
    torch.cuda.synchronize()
    return pixels.shape


def main() -> None:
    vae = VideoAutoencoderKLWrapper(**VAE_CFG).eval().to(DEV, torch.bfloat16)
    vae.load_state_dict(torch.load(os.path.join(WEIGHTS, "ema_vae.pth"),
                                   map_location="cpu", weights_only=True), strict=True)
    vae.set_causal_slicing(split_size=4, memory_device="same")

    dit = NaDiT(**DIT_ARGS).eval()
    sd = load_file(os.path.join(WEIGHTS, "seedvr2_3b_dit_f32.safetensors"))
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    assert not unexpected
    dit = dit.to(DEV, torch.bfloat16)
    del sd

    pos = torch.load(os.path.join(WEIGHTS, "pos_emb.pt"),
                     map_location="cpu", weights_only=True).to(DEV, torch.bfloat16)
    video = load_frames()
    n_frames = video.shape[0]
    print(f"frames={n_frames} area={AREA:.0f} dtype=bf16 slicing=on")

    shape = run_once(vae, dit, pos, video)   # warmup (discarded)
    print(f"warmup done, out={tuple(shape)}")

    times = []
    for i in range(TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once(vae, dit, pos, video)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"trial {i + 1}: {dt:.2f}s  ({dt / n_frames:.2f} s/frame)  "
              f"vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
        torch.cuda.reset_peak_memory_stats()

    mean = float(np.mean(times))
    sd_ = float(np.std(times, ddof=1))
    ci = 2.776 * sd_ / len(times) ** 0.5      # Student-t, df=4, 95%
    print(f"MEAN {mean:.2f}s ± {ci:.2f} (95% CI)  |  {mean / n_frames:.3f} s/frame")


if __name__ == "__main__":
    main()
