"""Real-weight SeedVR2-3B E2E reference restoration (Part A7 gate).

Faithful re-drive of projects/inference_seedvr2_3b.py minus the distributed/mediapy scaffolding:
preprocess (AreaResize/clamp/DivisibleCrop/Normalize + cut_videos) -> VAE encode (basic path, no causal
slicing) -> posterior sample -> 33ch condition (noise|latent|mask=1, sr task) -> ONE NaDiT forward at
t=1000 (cfg 1.0, no second pass) -> x0 = noise - v -> VAE decode. FP32 end-to-end so the C# comparison is
dtype-clean. Saves the two noises (cell-major [cells,16]) and the output pixels for the C# side.

flash_attn and apex are absent -> math-identical shims injected (SDPA varlen; diffusers RMSNorm /
nn.LayerNorm for the fused apex norms).

Usage: <venv-python> run_seedvr2_e2e_reference.py <SeedVR-checkout> <weights-dir> <frames-dir> <out.safetensors> [cuda|cpu] [stage]

`stage` ∈ {all, encode, dit, decode}. The default `all` re-execs each stage as a SEPARATE process —
the 13.6 GB f32 DiT otherwise stays pinned on-device through VAE decode (in-process del/gc did not
release it) and the three phases only fit the 24 GB card one at a time.
"""
import importlib.machinery
import os
import sys
import types

import torch

# ---- shims BEFORE model imports ----
def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                 dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        s, e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        qi = q[s:e].transpose(0, 1).unsqueeze(0).float()
        ki = k[s:e].transpose(0, 1).unsqueeze(0).float()
        vi = v[s:e].transpose(0, 1).unsqueeze(0).float()
        oi = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)
        outs.append(oi.squeeze(0).transpose(0, 1).to(q.dtype))
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
from models.dit_v2 import rope as _rope_mod  # noqa: E402


def _get_freqs_small(self, vid_shape, txt_shape):
    """Reference get_freqs builds a (1024,128,128) axial table — 8.4 GB in f32, OOM alongside the 13.6 GB
    DiT. Frequencies depend only on position index, so a table sized to the LARGEST window sliced
    identically is bit-equal. Mirrors the original per-row loop (windowed attention passes one row PER
    WINDOW, not one per sample)."""
    t_max = int((txt_shape[:, 0] + vid_shape[:, 0]).max())
    h_max = int(vid_shape[:, 1].max())
    w_max = int(vid_shape[:, 2].max())
    vid_freqs = self.get_axial_freqs(t_max, h_max, w_max)
    txt_freqs = self.get_axial_freqs(1024)
    vid_list, txt_list = [], []
    for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
        vid_list.append(vid_freqs[l:l + f, :h, :w].reshape(-1, vid_freqs.size(-1)))
        txt_list.append(txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1)))
    return torch.cat(vid_list, dim=0), torch.cat(txt_list, dim=0)


_rope_mod.NaMMRotaryEmbedding3d.get_freqs = _get_freqs_small
from PIL import Image  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from models.dit_v2 import na  # noqa: E402
from models.dit_v2.nadit import NaDiT  # noqa: E402
from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper  # noqa: E402
from data.image.transforms.area_resize import AreaResize  # noqa: E402
from data.image.transforms.divisible_crop import DivisibleCrop  # noqa: E402

WEIGHTS, FRAMES, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
DEV = sys.argv[5] if len(sys.argv) > 5 else "cuda"
SCALE = 0.9152
# f32 whole-clip VAE at 720p-area needs ~18+ GB of activations (upstream runs bf16 + causal slicing).
# The parity gate runs f32 both sides at a reduced area — identical code paths, fits the 24 GB card.
AREA = float(os.environ.get("SEEDVR2_AREA", str(1280 * 720)))

DIT_ARGS = dict(
    vid_in_channels=33, vid_out_channels=16, vid_dim=2560, txt_in_dim=5120, txt_dim=2560,
    emb_dim=15360, heads=20, head_dim=128, expand_ratio=4, norm="fusedrms", norm_eps=1e-5,
    ada="single", qk_bias=False, qk_norm="fusedrms", patch_size=(1, 2, 2), num_layers=32,
    block_type="mmdit_sr", mm_layers=10, mlp_type="swiglu",
    window=[(4, 3, 3)] * 32,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
    rope_type="mmrope3d", rope_dim=128, vid_out_norm="fusedrms",
)
VAE_CFG = dict(
    act_fn="silu", block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock3D"] * 4, in_channels=3, latent_channels=16,
    layers_per_block=2, norm_num_groups=32, out_channels=3,
    up_block_types=["UpDecoderBlock3D"] * 4, use_quant_conv=False, use_post_quant_conv=False,
    spatial_downsample_factor=8, temporal_downsample_factor=4,
    slicing_sample_min_size=4, temporal_scale_num=2, inflation_mode="pad", freeze_encoder=False,
)


STAGE = sys.argv[6] if len(sys.argv) > 6 else "all"
TMP = OUT + ".stage"


def main() -> None:
    if STAGE == "all":
        import subprocess
        for stage in ("encode", "dit", "decode"):
            subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:6], stage],
                           check=True, cwd=sys.argv[1])
        return
    if STAGE == "encode":
        stage_encode()
    elif STAGE == "dit":
        stage_dit()
    else:
        stage_decode()


def stage_encode() -> None:
    files = sorted(f for f in os.listdir(FRAMES) if f.endswith(".png"))
    video = torch.stack([
        torch.from_numpy(np.array(Image.open(os.path.join(FRAMES, f)).convert("RGB")))
        for f in files]).permute(0, 3, 1, 2).float() / 255.0            # TCHW
    print("input", tuple(video.shape))

    x = AreaResize(max_area=AREA, downsample_only=False)(video)
    x = torch.clamp(x, 0.0, 1.0)
    x = DivisibleCrop((16, 16))(x)
    x = ((x - 0.5) / 0.5).permute(1, 0, 2, 3).contiguous()               # CTHW
    t_frames, out_h, out_w = x.shape[1], x.shape[2], x.shape[3]
    print("preprocessed", tuple(x.shape))

    vae = VideoAutoencoderKLWrapper(**VAE_CFG).float().eval().to(DEV)
    vae.load_state_dict(torch.load(os.path.join(WEIGHTS, "ema_vae.pth"),
                                   map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        posterior = vae.encode(x.unsqueeze(0).to(DEV)).posterior
        mean, logvar = posterior.mean.cpu(), posterior.logvar.cpu()
    save_file({"mean": mean.contiguous(), "logvar": logvar.contiguous()}, TMP + ".enc")
    print("encode done", tuple(mean.shape))


def stage_dit() -> None:
    from safetensors.torch import load_file
    enc = load_file(TMP + ".enc")
    mean, logvar = enc["mean"], enc["logvar"]
    _, _, lt, lh, lw = mean.shape
    cells = lt * lh * lw
    gen = torch.Generator().manual_seed(666)
    eps = torch.randn((cells, 16), generator=gen)
    init = torch.randn((cells, 16), generator=gen)

    std = torch.exp(0.5 * logvar.clamp(-30, 20))[0]                      # (16,lt,lh,lw)
    eps_chw = eps.T.reshape(16, lt, lh, lw)
    z = (mean[0] + std * eps_chw) * SCALE                                # channels-first latent, scaled
    z_last = z.permute(1, 2, 3, 0)                                       # (lt,lh,lw,16)
    init_last = init.T.reshape(16, lt, lh, lw).permute(1, 2, 3, 0)

    cond = torch.cat([z_last, torch.ones(lt, lh, lw, 1)], dim=-1)
    dit_in = torch.cat([init_last, cond], dim=-1)                        # (lt,lh,lw,33)

    dit = NaDiT(**DIT_ARGS).float().eval()
    sd = torch.load(os.path.join(WEIGHTS, "seedvr2_ema_3b.pth"), map_location="cpu", weights_only=True)
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(m.endswith("rope.rope.freqs") for m in missing) or not missing, missing[:5]
    dit = dit.to(DEV)

    pos = torch.load(os.path.join(WEIGHTS, "pos_emb.pt"), map_location="cpu", weights_only=True).float()
    vid_flat, vid_shape = na.flatten([dit_in.to(DEV)])
    txt_flat, txt_shape = na.flatten([pos.to(DEV)])
    with torch.no_grad():
        v = dit(vid=vid_flat, txt=txt_flat, vid_shape=vid_shape, txt_shape=txt_shape,
                timestep=torch.tensor([1000.0], device=DEV)).vid_sample.cpu()

    x0 = (init.reshape(cells, 16) - v.reshape(cells, 16)) / SCALE
    x0 = x0.T.reshape(1, 16, lt, lh, lw)
    save_file({"x0": x0.contiguous(), "posterior_noise": eps.contiguous(),
               "init_noise": init.contiguous()}, TMP + ".dit")
    print("dit done", tuple(x0.shape))


def stage_decode() -> None:
    from safetensors.torch import load_file
    st = load_file(TMP + ".dit")
    vae = VideoAutoencoderKLWrapper(**VAE_CFG).float().eval().to(DEV)
    vae.load_state_dict(torch.load(os.path.join(WEIGHTS, "ema_vae.pth"),
                                   map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        pixels = vae.decode(st["x0"].to(DEV))[0].cpu()
    pixels = pixels[0] if pixels.ndim == 5 else pixels
    print("decoded", tuple(pixels.shape))

    save_file({
        "posterior_noise": st["posterior_noise"],
        "init_noise": st["init_noise"],
        "output": pixels.float().contiguous(),                           # (3,F,H,W) in [-1,1]
    }, OUT)
    u8 = ((pixels.clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte().permute(1, 2, 3, 0).numpy()
    for i in range(u8.shape[0]):
        Image.fromarray(u8[i]).save(os.path.join(os.path.dirname(OUT), f"ref_out_{i:02d}.png"))
    print("saved ->", OUT, "| frames:", u8.shape[0], f"{u8.shape[2]}x{u8.shape[1]}")


if __name__ == "__main__":
    main()
