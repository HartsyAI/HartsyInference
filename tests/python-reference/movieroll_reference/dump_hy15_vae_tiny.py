"""Tiny-config reference for the HunyuanVideo-1.5 causal VAE (MovieRoll's causal_vae.py: 16x spatial / 4x temporal, 5 levels, RMS norm,
DCAE shuffle resamplers), non-streaming numerics.

Run inside the CPU reference image, never the Windows venv:

  bash .claude/run-outputs/refenv/run.sh -v "<worktree>:/wt" -v "<MovieRoll checkout at 0b5f73e>:/src:ro" -- \
      python -I /wt/tests/python-reference/movieroll_reference/dump_hy15_vae_tiny.py \
      --out /wt/tests/HartsyInference.Diffusion.Tests/Fixtures/hy15_vae_tiny.safetensors

The real Encoder and Decoder modules are imported from the MovieRoll checkout; only diffusers and the loader are stubbed so the module imports.
Weights are random (seeded) and stored under the reference's own state-dict names, prefixed encoder./decoder. as AutoencoderKLConv3D does.
Besides the end-to-end tensors, the DCAE resamplers (input x, conv output h, output) and the mid attention (input, output) are dumped so the
C# tests can localise a mismatch to one branch instead of to the whole VAE.
"""
import argparse
import sys
import types

sys.path.insert(0, "/src")


class _Base:
    """Stands in for diffusers' ConfigMixin/ModelMixin/BaseOutput: the classes are only subclassed at import time."""


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


_module("diffusers")
_module("diffusers.configuration_utils", ConfigMixin=type("ConfigMixin", (), {}), register_to_config=lambda fn: fn)
_module("diffusers.models")
_module("diffusers.models.modeling_outputs", AutoencoderKLOutput=type("AutoencoderKLOutput", (), {}))
_module("diffusers.models.modeling_utils", ModelMixin=type("ModelMixin", (), {}))
_module("diffusers.utils", BaseOutput=type("BaseOutput", (), {}))
_module("diffusers.utils.torch_utils", randn_tensor=None)
_module("movieroll.models.autoencoders.vae_loader", DiagonalGaussianDistribution=_Base)

import torch
from safetensors.torch import save_file

from movieroll.models.autoencoders.causal_vae import Decoder, Encoder

BLOCK_OUT = (4, 8, 16, 32, 32)
LATENT = 2
LAYERS = 2
SPATIAL = 16
TEMPORAL = 4


def randomise_norms(module):
    """RMS gammas default to ones; random values make the norm's per-channel scale visible in the comparison."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("gamma"):
                p.copy_(torch.rand_like(p) * 0.5 + 0.75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(20261124)
    torch.set_num_threads(1)
    encoder = Encoder(in_channels=3, z_channels=LATENT, block_out_channels=BLOCK_OUT, num_res_blocks=LAYERS,
                      ffactor_spatial=SPATIAL, ffactor_temporal=TEMPORAL).eval()
    decoder = Decoder(z_channels=LATENT, out_channels=3, block_out_channels=tuple(reversed(BLOCK_OUT)),
                      num_res_blocks=LAYERS, ffactor_spatial=SPATIAL, ffactor_temporal=TEMPORAL).eval()
    randomise_norms(encoder)
    randomise_norms(decoder)

    captured = {}

    def hook_dcae(name, module):
        def fn(mod, inputs, output):
            x = inputs[0].detach()
            captured[f"{name}.x"] = x.clone()
            captured[f"{name}.h"] = mod.conv(x).detach().clone()
            captured[f"{name}.out"] = output.detach().clone()
        module.register_forward_hook(fn)

    def hook_attn(name, module):
        def fn(mod, inputs, output):
            captured[f"{name}.in"] = inputs[0].detach().clone()
            captured[f"{name}.out"] = output.detach().clone()
        module.register_forward_hook(fn)

    for i, level in enumerate(encoder.down):
        if hasattr(level, "downsample"):
            hook_dcae(f"enc.down.{i}.downsample", level.downsample)
    hook_attn("enc.mid.attn_1", encoder.mid.attn_1)
    for i, level in enumerate(decoder.up):
        if hasattr(level, "upsample"):
            hook_dcae(f"dec.up.{i}.upsample", level.upsample)
    hook_attn("dec.mid.attn_1", decoder.mid.attn_1)

    tensors = {}
    with torch.no_grad():
        video = torch.randn(1, 3, 5, 16, 32)
        moments = encoder(video)
        mean = moments[:, :LATENT].clone()
        decoded = decoder(mean)

        image = torch.randn(1, 3, 1, 16, 32)
        image_moments = encoder(image)
        image_mean = image_moments[:, :LATENT].clone()
        image_decoded = decoder(image_mean)

    for key, value in encoder.state_dict().items():
        tensors[f"encoder.{key}"] = value.detach().contiguous()
    for key, value in decoder.state_dict().items():
        tensors[f"decoder.{key}"] = value.detach().contiguous()
    tensors["input"] = video.contiguous()
    tensors["moments"] = moments.contiguous()
    tensors["mean"] = mean
    tensors["decoded"] = decoded.contiguous()
    tensors["image_input"] = image.contiguous()
    tensors["image_moments"] = image_moments.contiguous()
    tensors["image_decoded"] = image_decoded.contiguous()
    for key, value in captured.items():
        tensors[f"stage.{key}"] = value.contiguous()

    save_file(tensors, args.out)
    print(f"wrote {len(tensors)} tensors to {args.out}")
    print("moments", tuple(moments.shape), "decoded", tuple(decoded.shape))


if __name__ == "__main__":
    main()
