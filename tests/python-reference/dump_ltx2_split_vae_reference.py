"""CPU-only reference dump for the LTX-2.3 split-checkpoint video VAE decoder (Kijai
`LTX23_video_vae_bf16.safetensors`), used to root-cause the 2026-07-21 CLI-path noise/checkerboard bug
(see docs/Checklists/MODEL_STATUS_VIDEO.md, LTX-2.3 row). Loads the REAL weights into the vendored
diffusers `LTX2VideoDecoder3d`, decodes a synthetic latent through the exact same op granularity as the
engine's `HARTSY_LTX2_PROBE=1` trace (conv_in / mid_resnets / up_stage_0..3 / norm_out / conv_out), and
prints per-stage min/max/mean/nan/inf stats for a direct diff against the C# probe log.

Run with the SwarmUI ComfyUI venv's python (has torch + diffusers installed): CPU only, never touches CUDA.
"""
import json
import struct
import sys

import torch
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoDecoder3d

VAE_PATH = "/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/LTX23_video_vae_bf16.safetensors"


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        tensors = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[meta["dtype"]]
            shape = meta["shape"]
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
            tensors[key] = t.to(torch.float32)
    return tensors


def regroup_up_block(key: str) -> str:
    """Mirrors LtxVideo2CheckpointConverter.RegroupUpBlock: flat up_blocks.{i} -> diffusers grouping by
    index parity (0 = mid_block, odd = upsampler, even>0 = the (i/2-1)-th up-stage's resnets)."""
    tok = "up_blocks."
    at = key.find(tok)
    if at < 0:
        return key
    rest = key[at + len(tok):]
    num_str = ""
    for ch in rest:
        if ch.isdigit():
            num_str += ch
        else:
            break
    if not num_str:
        return key
    i = int(num_str)
    if i == 0:
        mapped = "mid_block"
    elif i % 2 == 1:
        mapped = f"up_blocks.{(i - 1) // 2}.upsamplers.0"
    else:
        mapped = f"up_blocks.{i // 2 - 1}"
    return key[:at] + mapped + rest[len(num_str):]


def build_state_dict(raw):
    sd = {}
    for key, tensor in raw.items():
        if not key.startswith("decoder."):
            continue
        k = key[len("decoder."):]
        k = regroup_up_block(k)
        k = k.replace("res_blocks", "resnets")
        sd[k] = tensor
    return sd


def stats(label, t):
    finite = t[torch.isfinite(t)]
    nan = torch.isnan(t).sum().item()
    inf = torch.isinf(t).sum().item()
    mn = finite.min().item() if finite.numel() else float("nan")
    mx = finite.max().item() if finite.numel() else float("nan")
    mean = finite.mean().item() if finite.numel() else float("nan")
    print(f"[ltx2-vae-probe-py] {label}: shape={list(t.shape)} min={mn:.4f} max={mx:.4f} mean={mean:.4f} nan={nan} inf={inf}", flush=True)


def main():
    torch.manual_seed(42)
    raw = load_safetensors(VAE_PATH)
    print(f"loaded {len(raw)} raw tensors", file=sys.stderr)

    latents_mean = raw["per_channel_statistics.mean-of-means"]
    latents_std = raw["per_channel_statistics.std-of-means"]
    print(f"latents_mean[:4]={latents_mean[:4].tolist()} latents_std[:4]={latents_std[:4].tolist()}", file=sys.stderr)

    sd = build_state_dict(raw)
    print(f"remapped {len(sd)} decoder tensors", file=sys.stderr)

    # Config derived from the real checkpoint's weight shapes (see MODEL_STATUS_VIDEO.md investigation):
    # 4 up-stages, channels [1024 mid ->512->512->256->128], layers [2,2,4,6,4],
    # upsample types [spatiotemporal, spatiotemporal, temporal, spatial], upscale factors [2,1,2,2].
    decoder = LTX2VideoDecoder3d(
        in_channels=128,
        out_channels=3,
        block_out_channels=(256, 512, 512, 1024),
        spatio_temporal_scaling=(True, True, True, True),
        layers_per_block=(4, 6, 4, 2, 2),
        upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
        patch_size=4,
        patch_size_t=1,
        is_causal=False,
        timestep_conditioning=False,
        inject_noise=False,
        upsample_residual=(True, True, True, True),
        upsample_factor=(2, 2, 1, 2),
        spatial_padding_mode="reflect",
    )
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    print(f"missing={missing}", file=sys.stderr)
    print(f"unexpected={unexpected}", file=sys.stderr)
    if missing or unexpected:
        print("!!! state_dict mismatch -- config/remap is WRONG, stats below are not trustworthy !!!", file=sys.stderr)

    decoder.eval()

    # Small synthetic latent (same distribution shape as a real denorm'd latent: unit-scale-ish).
    with torch.no_grad():
        latent = torch.randn(1, 128, 3, 4, 4)
        denorm = latent * latents_std.view(1, -1, 1, 1, 1) + latents_mean.view(1, -1, 1, 1, 1)
        stats("denorm", denorm)

        h = decoder.conv_in(denorm, causal=decoder.is_causal)
        stats("conv_in", h)

        h = decoder.mid_block(h, None, causal=decoder.is_causal)
        stats("mid_resnets", h)

        for i, up_block in enumerate(decoder.up_blocks):
            h = up_block(h, None, causal=decoder.is_causal)
            stats(f"up_stage_{i}", h)

        h = decoder.norm_out(h)
        stats("norm_out", h)
        h = decoder.conv_act(h)
        h = decoder.conv_out(h, causal=decoder.is_causal)
        stats("conv_out", h)


if __name__ == "__main__":
    main()
