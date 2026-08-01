"""Reference dump for the SeedVR2 causal video VAE (Part A3 parity).

Instantiates ByteDance's VideoAutoencoderKLWrapper with the REAL ema_vae.pth on CPU/f32 and dumps:
  - encoder mean/logvar for a seeded [1,3,5,64,64] input (deterministic — no posterior sampling), and
  - decoder RGB output for a seeded [1,16,2,8,8] latent,
plus both inputs, to one safetensors file. No causal slicing (basic_forward path, memory_limit=inf),
matching the C# port's whole-clip forward.

Usage:
    <seedvr2-venv-python> dump_seedvr2_vae_reference.py <SeedVR-checkout> <ema_vae.pth> <out.safetensors>
"""
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, sys.argv[1])
from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper  # noqa: E402

CFG = dict(
    act_fn="silu",
    block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock3D"] * 4,
    in_channels=3,
    latent_channels=16,
    layers_per_block=2,
    norm_num_groups=32,
    out_channels=3,
    up_block_types=["UpDecoderBlock3D"] * 4,
    use_quant_conv=False,
    use_post_quant_conv=False,
    spatial_downsample_factor=8,
    temporal_downsample_factor=4,
    slicing_sample_min_size=4,
    temporal_scale_num=2,
    inflation_mode="pad",
    freeze_encoder=False,
)


def main() -> None:
    torch.manual_seed(20260801)
    model = VideoAutoencoderKLWrapper(**CFG)
    state = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True), None
    model = model.float().eval()

    gen = torch.Generator().manual_seed(42)
    video = (torch.rand((1, 3, 5, 64, 64), generator=gen) * 2.0 - 1.0).float()
    latent = torch.randn((1, 16, 2, 8, 8), generator=gen).float()

    with torch.no_grad():
        h = model._encode(video) if hasattr(model, "_encode") else None
        if h is None:
            posterior = model.encode(video.clone()).posterior
            mean, logvar = posterior.mean, posterior.logvar
        else:
            mean, logvar = h.chunk(2, dim=1)
        decoded = model.decode(latent.clone())[0]

    out = {
        "enc.input": video,
        "enc.mean": mean.float(),
        "enc.logvar": logvar.float(),
        "dec.input": latent,
        "dec.output": decoded.float(),
    }
    save_file(out, sys.argv[3])
    for k, v in out.items():
        print(k, tuple(v.shape), f"absmax={v.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
