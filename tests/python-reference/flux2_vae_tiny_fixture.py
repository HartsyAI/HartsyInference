"""
Writes a tiny random VAE in the FLUX.2 VAE's diffusers key layout (encoder/decoder down/up blocks, mid-block attention,
quant/post-quant convs, the patchified-latent BatchNorm stats), every stage 8 channels wide over a 4-channel latent, so
the packed tokens are 16 wide like the tiny MOSS-Image fixture's. MossImageEditPipelineTests drives the whole MOSS-Image
pipeline (VAE encode of the references, the DiT, VAE decode) with it. Random weights: it checks wiring, not images.

Usage: python flux2_vae_tiny_fixture.py <output .safetensors>
Requires: numpy, safetensors.
"""
import sys

import numpy as np
from safetensors.numpy import save_file

C, LATENT, STAGES, SEED = 8, 4, 4, 20261019


def main():
    rng = np.random.default_rng(SEED)
    shapes = {}

    def conv(name, out_ch, in_ch, k=3):
        shapes[f"{name}.weight"] = (out_ch, in_ch, k, k)
        shapes[f"{name}.bias"] = (out_ch,)

    def norm(name, ch):
        shapes[f"{name}.weight"] = (ch,)
        shapes[f"{name}.bias"] = (ch,)

    def resnet(prefix):
        norm(f"{prefix}.norm1", C)
        conv(f"{prefix}.conv1", C, C)
        norm(f"{prefix}.norm2", C)
        conv(f"{prefix}.conv2", C, C)

    def mid(prefix):
        resnet(f"{prefix}.mid_block.resnets.0")
        a = f"{prefix}.mid_block.attentions.0"
        norm(f"{a}.group_norm", C)
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            shapes[f"{a}.{proj}.weight"] = (C, C)
            shapes[f"{a}.{proj}.bias"] = (C,)
        resnet(f"{prefix}.mid_block.resnets.1")

    conv("encoder.conv_in", C, 3)
    for i in range(STAGES):
        for j in range(2):
            resnet(f"encoder.down_blocks.{i}.resnets.{j}")
        if i < STAGES - 1:
            conv(f"encoder.down_blocks.{i}.downsamplers.0.conv", C, C)
    mid("encoder")
    norm("encoder.conv_norm_out", C)
    conv("encoder.conv_out", 2 * LATENT, C)
    conv("quant_conv", 2 * LATENT, 2 * LATENT, 1)

    conv("post_quant_conv", LATENT, LATENT, 1)
    conv("decoder.conv_in", C, LATENT)
    mid("decoder")
    for i in range(STAGES):
        for j in range(3):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}")
        if i < STAGES - 1:
            conv(f"decoder.up_blocks.{i}.upsamplers.0.conv", C, C)
    norm("decoder.conv_norm_out", C)
    conv("decoder.conv_out", 3, C)

    tensors = {}
    for name, shape in shapes.items():
        if "norm" in name and name.endswith(".weight"):
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name.endswith(".bias"):
            value = 0.05 * rng.standard_normal(shape)
        else:
            fan_in = int(np.prod(shape[1:]))
            value = rng.standard_normal(shape) / np.sqrt(fan_in)
        tensors[name] = value.astype(np.float32)
    tensors["bn.running_mean"] = (0.1 * rng.standard_normal(4 * LATENT)).astype(np.float32)
    tensors["bn.running_var"] = (1.0 + 0.2 * rng.random(4 * LATENT)).astype(np.float32)
    save_file(tensors, sys.argv[1])
    print(len(tensors), "tensors,", sum(t.size for t in tensors.values()), "values")


if __name__ == "__main__":
    main()
