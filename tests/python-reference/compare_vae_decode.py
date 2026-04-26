"""
Compares VAE decode between SharpInference (C#) and diffusers (Python).
Loads the C# unpacked latent from a binary file, decodes it using the VAE
weights from the same safetensors checkpoint, and reports per-channel stats.

Usage: python compare_vae_decode.py <checkpoint_path> <latent_bin_path>
  checkpoint_path: path to flux1-dev-fp8.safetensors (or any Flux safetensors)
  latent_bin_path: path to unpacked_latent.bin (saved by C# FluxPipeline)

Requires: pip install diffusers torch safetensors numpy
"""
import sys
import os
import numpy as np
import torch
from safetensors import safe_open


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_vae_decode.py <checkpoint_path> <latent_bin_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    latent_bin_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(latent_bin_path)

    # 1. Load C# latent [1, 16, H, W] as float32
    print(f"Loading C# latent from {latent_bin_path}...")
    raw = np.fromfile(latent_bin_path, dtype=np.float32)
    # For 512x512: latent is [1, 16, 64, 64] = 65536 elements
    total = raw.shape[0]
    channels = 16
    # Infer spatial dims
    spatial = total // channels
    h = w = int(np.sqrt(spatial))
    assert h * w * channels == total, f"Cannot infer shape: {total} elements, channels={channels}"
    latent = raw.reshape(1, channels, h, w)
    print(f"  Shape: {latent.shape}, range: [{latent.min():.4f}, {latent.max():.4f}], mean: {latent.mean():.4f}")

    for c in range(channels):
        ch = latent[0, c]
        print(f"  ch{c}: min={ch.min():.4f} max={ch.max():.4f} mean={ch.mean():.4f}")

    # 2. Load VAE weights from safetensors
    print(f"\nLoading VAE weights from {checkpoint_path}...")
    vae_keys = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        all_keys = f.keys()
        # Find VAE keys - in Flux single-file checkpoints they may start with "vae." or just be the decoder keys
        for key in all_keys:
            if "first_stage_model" in key or key.startswith("vae."):
                clean_key = key.replace("first_stage_model.", "").replace("vae.", "")
                vae_keys[clean_key] = f.get_tensor(key)

    if not vae_keys:
        print("  No VAE keys found with 'first_stage_model' or 'vae.' prefix.")
        print("  Trying to find decoder keys directly...")
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "decoder" in key or "post_quant_conv" in key or "conv_out" in key or "conv_in" in key or "conv_norm_out" in key:
                    vae_keys[key] = f.get_tensor(key)

    print(f"  Found {len(vae_keys)} VAE keys")
    if len(vae_keys) > 0:
        # Show first few keys and their dtypes
        for i, (k, v) in enumerate(vae_keys.items()):
            if i < 5:
                print(f"    {k}: {v.shape} {v.dtype}")
            elif i == 5:
                print(f"    ... and {len(vae_keys) - 5} more")

    # 3. Build diffusers VAE and load weights
    print("\nBuilding diffusers AutoencoderKL...")
    from diffusers import AutoencoderKL

    # Flux VAE config (no quant_conv/post_quant_conv for Flux)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
        latent_channels=16,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )

    # Try to load state dict
    try:
        # Map our keys to diffusers VAE keys
        state_dict = {}
        for k, v in vae_keys.items():
            # Cast FP8/FP16 to FP32
            if v.dtype != torch.float32:
                v = v.float()
            state_dict[k] = v

        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: {len(state_dict) - len(missing)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"  First 5 missing: {missing[:5]}")
        if unexpected:
            print(f"  First 5 unexpected: {unexpected[:5]}")
    except Exception as e:
        print(f"  Direct load failed: {e}")
        print("  Trying to remap keys...")
        # Fallback: list what we have and what VAE expects
        expected = set(vae.state_dict().keys())
        have = set(vae_keys.keys())
        print(f"  Expected keys: {len(expected)}, Have: {len(have)}")
        overlap = expected & have
        print(f"  Overlap: {len(overlap)}")
        only_expected = expected - have
        only_have = have - expected
        if only_expected:
            print(f"  Only in expected (first 10): {sorted(list(only_expected))[:10]}")
        if only_have:
            print(f"  Only in have (first 10): {sorted(list(only_have))[:10]}")
        return

    vae.eval()

    # 4. Decode latent
    print(f"\nDecoding latent with diffusers VAE...")
    latent_tensor = torch.from_numpy(latent).float()

    with torch.no_grad():
        # Apply inverse scaling (same as Flux VAE)
        z = (latent_tensor / 0.3611) + 0.1159
        decoded = vae.decode(z).sample  # [1, 3, H*8, W*8]

    decoded_np = decoded.numpy()
    print(f"  Decoded shape: {decoded_np.shape}")

    # 5. Per-channel stats
    print("\n=== Python VAE output per-channel stats ===")
    for c in range(3):
        ch = decoded_np[0, c]
        print(f"  ch{c} ({'RGB'[c]}): min={ch.min():.4f} max={ch.max():.4f} mean={ch.mean():.4f}")

    # 6. Convert to RGB bytes (same as C#: (val+1)*0.5*255, clamp 0-255)
    rgb_float = (decoded_np[0].transpose(1, 2, 0) + 1.0) * 0.5  # [H, W, 3] in [0, 1]
    rgb_bytes = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)

    print("\n=== Python RGB byte per-channel stats ===")
    for c in range(3):
        ch = rgb_bytes[:, :, c]
        print(f"  {'RGB'[c]}: min={ch.min()} max={ch.max()} mean={ch.mean():.1f}")

    # 7. Save output image
    try:
        from PIL import Image
        img = Image.fromarray(rgb_bytes)
        out_path = os.path.join(output_dir, "python_vae_decoded.png")
        img.save(out_path)
        print(f"\nSaved Python decoded image to {out_path}")
    except ImportError:
        out_path = os.path.join(output_dir, "python_vae_decoded.npy")
        np.save(out_path, rgb_bytes)
        print(f"\nSaved as numpy: {out_path} (install PIL for PNG)")


if __name__ == "__main__":
    main()
