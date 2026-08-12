"""Tiny-config reference dump for the LTX-2.5 `NADiffusionDecoder` C# port.

Builds a small random-weight decoder with the SAME module wiring as ComfyUI's
`comfy/ldm/lightricks/vae/na_diffusion_decoder.py`, but sources the two numerically
tricky pieces independently so a shared bug cannot hide:

  * neighborhood attention -> `na_eager.na3d` (comfy-kitchen's vendored eager backend)
  * absolute RoPE          -> `rot_abs_axis_impl` / `rope_inv_freqs` / `default_rope_dim_split`
                              from the official `ltx_core...transformer/rope_math.py`

Both files are third-party and not vendored into this repo. Point at them with:
    NA_EAGER_DIR   = directory containing na_eager.py
    LTX_ROPE_MATH  = path to ltx_core/model/video_vae/transformer/rope_math.py
Torch lives only in the ComfyUI venv on this box, e.g.
    "<SwarmUI>/dlbackend/ComfyUI/venv/bin/python3" ltx25_diffusion_decoder_reference.py

Writes `ltx25_diffusion_decoder_reference.bin` (gitignored) next to this file:
config ints, then a name -> f32 tensor table holding the checkpoint-named weights,
the input latent, the injected noise, the stage-1..4 context and the final pixels.
Weight names are the real checkpoint key names, so the C# parity test exercises key
mapping as well as numerics.
"""

import importlib.util
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_rope_math():
    candidates = [os.environ.get("LTX_ROPE_MATH"), os.path.join(_HERE, "rope_math.py")]
    for path in candidates:
        if path and os.path.exists(path):
            spec = importlib.util.spec_from_file_location("ltx_rope_math", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("rope_math.py not found - set LTX_ROPE_MATH to the ltx-core copy.")


_rope_math = _load_rope_math()
default_rope_dim_split = _rope_math.default_rope_dim_split
rope_inv_freqs = _rope_math.rope_inv_freqs
rot_abs_axis_impl = _rope_math.rot_abs_axis_impl

for _dir in (os.environ.get("NA_EAGER_DIR"), _HERE):
    if _dir and os.path.exists(os.path.join(_dir, "na_eager.py")):
        sys.path.insert(0, _dir)
        break
try:
    from na_eager import na3d
except ImportError as exc:  # pragma: no cover
    raise SystemExit("na_eager.py not found - set NA_EAGER_DIR to its directory.") from exc


# --- reference modules (wiring mirrors comfy's na_diffusion_decoder.py) ---


def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos, downscale_freq_shift, scale=1, max_period=10000):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


def patchify(x, patch_size_hw, patch_size_t=1):
    return rearrange(x, "b c (f p) (h q) (w r) -> b (c p r q) f h w",
                     p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


def unpatchify(x, patch_size_hw, patch_size_t=1):
    return rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)",
                     p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


class NeighborhoodAttention3D(nn.Module):
    def __init__(self, dim, kernel_size, head_dim=64, rope_base=10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim ** -0.5
        self.rope_split = default_rope_dim_split(head_dim)
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def _rope(self, x):
        d_t, d_h, _ = self.rope_split
        inv = [rope_inv_freqs(d, self.rope_base) for d in self.rope_split]
        t, h, w = x.shape[1], x.shape[2], x.shape[3]
        pos_t = torch.arange(t, dtype=torch.float32)
        pos_h = torch.arange(h, dtype=torch.float32)
        pos_w = torch.arange(w, dtype=torch.float32)
        xt = rot_abs_axis_impl(x[..., :d_t], pos_t, inv[0], axis=1, compute_dtype=torch.float32)
        xh = rot_abs_axis_impl(x[..., d_t:d_t + d_h], pos_h, inv[1], axis=2, compute_dtype=torch.float32)
        xw = rot_abs_axis_impl(x[..., d_t + d_h:], pos_w, inv[2], axis=3, compute_dtype=torch.float32)
        return torch.cat([xt, xh, xw], dim=-1)

    def forward(self, x):
        batch, t, h, w, _ = x.shape
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)
        q, k = self._rope(q), self._rope(k)
        out = na3d(q.contiguous(), k.contiguous(), v.contiguous(), list(self.kernel_size), None, 1.0)
        return self.proj(out.reshape(batch, t, h, w, self.dim))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def swiglu_hidden(dim, mlp_ratio=4.0):
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class NABlock(nn.Module):
    def __init__(self, dim, kernel_size, head_dim=64):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, swiglu_hidden(dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def modulate(x, scale, shift):
    return x * (1.0 + scale) + shift


class AdaLNZero(nn.Module):
    NUM_CHUNKS = 7

    def __init__(self, dim, t_emb_dim):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb):
        h = self.proj(F.silu(t_emb))
        return tuple(c[:, None, None, None, :] for c in h.chunk(self.NUM_CHUNKS, dim=-1))


class DiffusionNABlock(nn.Module):
    def __init__(self, dim, kernel_size, context_channels, head_dim=64):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, swiglu_hidden(dim))

    def forward(self, x, latent_context, modulation):
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        x = x + self.context_proj(latent_context)
        x = x + self.attn(modulate(self.norm1(x), scale_msa, shift_msa))
        return x + self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, stride, out_channels_reduction_factor=1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, x, drop_leading_frame=True):
        p1, p2, p3 = self.stride
        out = rearrange(self.proj(x), "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c", p1=p1, p2=p2, p3=p3)
        if p1 == 2 and drop_leading_frame:
            out = out[:, 1:]
        return out


class TimestepEmbedder(nn.Module):
    def __init__(self, t_emb_dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, t_emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim, bias=True),
        )

    def forward(self, timestep):
        emb = get_timestep_embedding(timestep.flatten(), self.freq_dim, True, 0, 1)
        return self.mlp(emb)


class NADiffusionDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, head_dim, stage_channels, stage_depths,
                 stage_kernels, upsamples, stage5_kernel, t_emb_dim, timestep_scale_multiplier, freq_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.temporal_upscale = math.prod(s[0] for s, _ in upsamples)
        self.spatial_upscale = math.prod(s[1] for s, _ in upsamples) * patch_size
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)
        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(stage_channels) - 1):
            c = stage_channels[i]
            self.det_stages.append(nn.ModuleList(
                [NABlock(c, stage_kernels[i], head_dim=head_dim) for _ in range(stage_depths[i])]))
            stride, reduction = upsamples[i]
            self.upsamples.append(LinearPixelShuffleUpsample(c, stride, out_channels_reduction_factor=reduction))

        self.t_embedder = TimestepEmbedder(t_emb_dim, freq_dim=freq_dim)
        c5 = stage_channels[-1]
        noised_pixel_channels = out_channels * (patch_size ** 2)
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, c5, bias=True)
        self.shared_adaln = AdaLNZero(c5, t_emb_dim)
        self.diff_blocks = nn.ModuleList([
            DiffusionNABlock(c5, stage5_kernel, context_channels=c5, head_dim=head_dim)
            for _ in range(stage_depths[-1])])
        self.norm_out = RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, noised_pixel_channels, bias=True)
        self.type_emb = nn.Parameter(torch.zeros(in_channels))

    def forward_pre_diffusion(self, z):
        n = self.trailing_pad_latent_frames
        z = torch.cat([z, z[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)
        x = self.conv_in(z.permute(0, 2, 3, 4, 1))
        for i, blocks in enumerate(self.det_stages):
            for block in blocks:
                x = block(x)
            x = self.upsamples[i](x, drop_leading_frame=True)
        return x[:, :-(n * self.temporal_upscale)]

    def forward_diff_step(self, context, x_t, t):
        x = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(x.permute(0, 2, 3, 4, 1))
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(x, context, modulation)
        x = self.conv_out(self.norm_out(x)).permute(0, 4, 1, 2, 3)
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)


# --- tiny config + dump ---

CONFIG = dict(
    in_channels=24,
    out_channels=3,
    patch_size=4,
    head_dim=16,
    stage_channels=(128, 64, 32, 32, 16),
    stage_depths=(2, 1, 1, 1, 2),
    # Deliberately asymmetric per axis, and sized to fit every stage's grid without NATTEN's kernel clamping,
    # so a T/H/W mix-up cannot hide behind a square window.
    stage_kernels=((3, 3, 5), (3, 5, 3), (3, 3, 5), (3, 5, 3), (5, 3, 3)),
    upsamples=(((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
    stage5_kernel=(5, 3, 3),
    t_emb_dim=32,
    timestep_scale_multiplier=1000.0,
    freq_dim=64,
)
LATENT_SHAPE = (1, CONFIG["in_channels"], 2, 4, 5)


def randomize(model, gen):
    """Replaces the all-ones RMSNorm weights and the all-zeros scale_shift_table with real values, so a
    dropped norm weight or a dropped modulation table cannot pass."""
    for name, param in model.named_parameters():
        with torch.no_grad():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") \
                    or name.endswith("q_norm.weight") or name.endswith("k_norm.weight") \
                    or name.endswith("norm_out.weight"):
                param.copy_(1.0 + 0.25 * torch.randn(param.shape, generator=gen))
            elif name.endswith("scale_shift_table") or name.endswith("type_emb"):
                param.copy_(0.1 * torch.randn(param.shape, generator=gen))


def write_tensor(f, name, tensor):
    raw = name.encode("utf-8")
    f.write(struct.pack("<i", len(raw)))
    f.write(raw)
    dims = list(tensor.shape)
    f.write(struct.pack("<i", len(dims)))
    for d in dims:
        f.write(struct.pack("<i", int(d)))
    f.write(tensor.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)
    model = NADiffusionDecoder(**CONFIG)
    randomize(model, gen)
    model.eval()

    latent = torch.randn(LATENT_SHAPE, generator=gen, dtype=torch.float32)
    with torch.no_grad():
        context = model.forward_pre_diffusion(latent)
        batch, t5, h5, w5, _ = context.shape
        noise = torch.randn((batch, CONFIG["out_channels"], t5, h5 * model.patch_size, w5 * model.patch_size),
                            generator=gen, dtype=torch.float32)
        pixels = model.forward_diff_step(context, noise, torch.ones(batch, dtype=torch.float32))

    out_path = os.path.join(_HERE, "ltx25_diffusion_decoder_reference.bin")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<4i", CONFIG["in_channels"], CONFIG["out_channels"],
                            CONFIG["patch_size"], CONFIG["head_dim"]))
        f.write(struct.pack("<i", len(CONFIG["stage_channels"])))
        for c in CONFIG["stage_channels"]:
            f.write(struct.pack("<i", c))
        for d in CONFIG["stage_depths"]:
            f.write(struct.pack("<i", d))
        for k in CONFIG["stage_kernels"]:
            f.write(struct.pack("<3i", *k))
        f.write(struct.pack("<i", len(CONFIG["upsamples"])))
        for stride, reduction in CONFIG["upsamples"]:
            f.write(struct.pack("<4i", stride[0], stride[1], stride[2], reduction))
        f.write(struct.pack("<3i", *CONFIG["stage5_kernel"]))
        f.write(struct.pack("<2i", CONFIG["t_emb_dim"], CONFIG["freq_dim"]))
        f.write(struct.pack("<f", CONFIG["timestep_scale_multiplier"]))

        named = [(f"decoder.{n}", p) for n, p in model.state_dict().items()]
        named += [("__latent__", latent), ("__noise__", noise), ("__context__", context), ("__pixels__", pixels)]
        f.write(struct.pack("<i", len(named)))
        for name, tensor in named:
            write_tensor(f, name, tensor)

    print(f"context {tuple(context.shape)} pixels {tuple(pixels.shape)} "
          f"absmax {pixels.abs().max():.6f} -> {out_path} ({os.path.getsize(out_path)} bytes)")
    for name, _ in named[:6]:
        print("  key:", name)


if __name__ == "__main__":
    main()
