"""Generates the tiny LTX-2 VAE encoder fixtures from diffusers 0.37.0 on CPU.

Writes encoder weights (diffusers key names, F32) and one input/expected posterior-mode pair next to this script.
Run with the upstream venv; CUDA is hidden before torch loads so nothing touches a GPU:

    python generate_ltx2_vae_encoder_fixtures.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Three down blocks cover the three stride tuples the real checkpoint uses (spatial, temporal, spatiotemporal).
# Patch size 4 matches the real config, so the pixel-unshuffle channel order is exercised as it is in the checkpoint.
TINY = dict(
    in_channels=3,
    out_channels=3,
    latent_channels=8,
    block_out_channels=(8, 8, 16),
    down_block_types=("LTX2VideoDownBlock3D",) * 3,
    spatio_temporal_scaling=(True, True, True),
    downsample_type=("spatial", "temporal", "spatiotemporal"),
    layers_per_block=(1, 1, 1, 1),
    decoder_block_out_channels=(8, 8, 8),
    decoder_layers_per_block=(1, 1, 1, 1),
    decoder_spatio_temporal_scaling=(True, True, True),
    decoder_inject_noise=(False, False, False, False),
    upsample_residual=(True, True, True),
    upsample_factor=(2, 2, 2),
    patch_size=4,
    patch_size_t=1,
    resnet_norm_eps=1e-6,
    encoder_causal=True,
    decoder_causal=False,
    encoder_spatial_padding_mode="zeros",
)

# T = 1 + 4n (two temporal stride-2 stages), H and W multiples of 16 (patch 4 times three spatial stride-2 stages).
INPUT_SHAPE = (1, 3, 5, 32, 48)


def main() -> None:
    torch.manual_seed(20261109)
    torch.set_grad_enabled(False)
    vae = AutoencoderKLLTX2Video(**TINY).eval()

    generator = torch.Generator().manual_seed(7)
    video = torch.rand(INPUT_SHAPE, generator=generator) * 2.0 - 1.0

    # The encoder returns mean ‖ shared logvar; the posterior mode is the mean half.
    encoded = vae.encoder(video, causal=None)
    expected_mode = vae.encode(video).latent_dist.mode()
    assert expected_mode.shape == (1, 8, 2, 2, 3), expected_mode.shape
    assert torch.equal(encoded[:, :8], expected_mode), "encode().mode() is not the first latent_channels of encoder()"

    weights = {k: v.detach().contiguous() for k, v in vae.state_dict().items() if k.startswith("encoder.")}
    print(f"encoder weight tensors: {len(weights)}, elements: {sum(v.numel() for v in weights.values())}")
    save_file(weights, os.path.join(HERE, "ltx2_vae_encoder_tiny_weights.safetensors"), metadata={"format": "pt"})
    save_file(
        {"input": video.contiguous(), "expected_mode": expected_mode.detach().contiguous()},
        os.path.join(HERE, "ltx2_vae_encoder_tiny_io.safetensors"),
        metadata={"format": "pt"},
    )
    print("expected_mode shape:", tuple(expected_mode.shape), "input shape:", tuple(video.shape))


if __name__ == "__main__":
    main()
