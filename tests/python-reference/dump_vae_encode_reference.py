"""
Dumps VAE *encode*-direction references for the C# `VaeEncoder` parity gate.

Why this exists: the decode direction has had parity coverage for a long time, but the encode half
did not. MODEL_STATUS_IMAGE.md:98 records a stride-2 asymmetric-padding bug in `VaeEncoder` that
drove encode correlation against ComfyUI to 0.871 -- and notes that "img2img always masked it". Every
property-style img2img assertion (strength ordering, roundtrip drift, mask locality) passes with that
bug live, because they all compare img2img against itself. Only a reference comparison catches it.

The gate is per-VaeConfig, not per-family: one dump here covers every family sharing that VAE
(VaeConfig.Flux alone backs chroma / flux1 / lumina2 / kandinsky5 / f-lite / hidream / boogu /
omnigen2).

Run with a venv that has torch + diffusers, e.g.
    ~/venvs/seedvr2/bin/python tests/python-reference/dump_vae_encode_reference.py --family flux
    ~/venvs/seedvr2/bin/python tests/python-reference/dump_vae_encode_reference.py --family all

Output: tests/python-reference/vae_encode_reference/<family>/
  image.bin        [1, 3, H, W]  F32 in [-1, 1]   the encoder input
  mean.bin         [1, C, H/8, W/8] F32           raw posterior mean (pre scale/shift)
  latent.bin       [1, C, H/8, W/8] F32           (mean - shift) * scale, using the file's own config
  index.json       shapes + scaling_factor / shift_factor + stats

The C# side (VaeEncodeParityTests) asserts BOTH that our VaeConfig's scale/shift match the values
recorded here (convention drift) and that our encoder output matches latent.bin (network drift).
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO_ROOT, "Models", "VAE")
OUT_ROOT = os.path.join(REPO_ROOT, "tests", "python-reference", "vae_encode_reference")

# Deterministic, non-degenerate input: a smooth gradient plus structure, so a padding/stride bug on
# any edge shows up rather than being averaged away by a flat field.
SIZE = 256


def make_input(height: int, width: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    base = (ys + xs) / 2.0
    rings = torch.sin(6.0 * torch.pi * torch.sqrt(ys**2 + xs**2))
    r = base
    g = rings * 0.5
    b = (base * rings)
    img = torch.cat([r.expand(1, 1, height, width),
                     g.expand(1, 1, height, width),
                     b.expand(1, 1, height, width)], dim=1)
    # Hard edges at the border are what the asymmetric-padding bug corrupts.
    img[:, :, :4, :] = 1.0
    img[:, :, -4:, :] = -1.0
    img[:, :, :, :4] = -1.0
    img[:, :, :, -4:] = 1.0
    return img.clamp(-1.0, 1.0).contiguous()


# scaling / shift are NOT read from the checkpoint: these are bare safetensors with no config.json, so
# diffusers would silently fall back to its 0.18215 SD1.5 default. They are transcribed here from each
# model's official vae/config.json, which makes the C# assertion against VaeConfig a real cross-check
# (two independent transcriptions of the same published value) rather than a tautology.
#   flux  : black-forest-labs/FLUX.1-dev        vae/config.json
#   flux2 : Flux.2 normalizes via bn.running_mean/var at the pipeline boundary, not in the VAE
#   sd3   : stabilityai/stable-diffusion-3.5-medium vae/config.json
#   qwen  : per-channel latents_mean/latents_std, dumped separately below
FAMILIES = {
    # name: (weights path, diffusers class, latent channels, scaling_factor, shift_factor)
    "flux": (os.path.join(MODELS, "Flux", "ae.safetensors"), "AutoencoderKL", 16, 0.3611, 0.1159),
    "flux2": (os.path.join(MODELS, "Flux", "flux2-vae.safetensors"), "AutoencoderKL", 32, 1.0, 0.0),
    "sd3": (os.path.join(MODELS, "SD3", "sd3.5_medium_vae_extracted.safetensors"), "AutoencoderKL", 16, 1.5305, 0.0609),
    "qwen-image": (os.path.join(MODELS, "QwenImage", "qwen_image_vae.safetensors"), "AutoencoderKLQwenImage", 16, 1.0, 0.0),
}


def save(t: torch.Tensor, path: str) -> None:
    t.float().detach().cpu().contiguous().numpy().tofile(path)


def stats(t: torch.Tensor) -> dict:
    f = t.float().flatten()
    return {
        "shape": list(t.shape),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "min": float(f.min()),
        "max": float(f.max()),
        "first_8": [float(x) for x in f[:8]],
    }


def build_vae(cls_name: str, weights_path: str, latent_channels: int):
    """Builds the diffusers module from a bare state dict, inferring config from tensor shapes."""
    import diffusers

    state = load_file(weights_path)
    # Extracted-from-checkpoint VAEs (SD3.5) keep the LDM `first_stage_model.` prefix.
    for prefix in ("first_stage_model.", "vae."):
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            break
    if cls_name == "AutoencoderKLQwenImage":
        if "encoder.conv1.weight" not in state:
            raise SystemExit(f"{weights_path}: unrecognized Qwen VAE layout (no encoder.conv1.weight).")
        vae = diffusers.AutoencoderKLQwenImage()
        converted = translate_qwen_encoder(state)
        missing, _ = vae.load_state_dict(converted, strict=False)
        # Hard completeness check on the encode path. The sibling dumper leaves this to strict=False and notes
        # "we don't run encoder" -- that is exactly how the Flux/SD3 reference ended up with a randomly
        # initialized quant_conv and disagreed with correct code at corr~0.02.
        missing = [m for m in missing if m.startswith("encoder.") or m.startswith("quant_conv.")]
        if missing:
            raise SystemExit(f"{weights_path}: encode-path weights left uninitialized: {missing[:8]}")
        return vae.eval()

    # LDM-named files (Flux's ae.safetensors) need the diffusers converter; diffusers-named pass through.
    if not any(k.startswith("encoder.down_blocks") or k.startswith("encoder.down.") for k in state):
        raise SystemExit(f"{weights_path}: unrecognized VAE layout (no encoder.down* keys).")

    block_out = infer_block_out_channels(state)
    # LOAD-BEARING: diffusers' AutoencoderKL builds quant_conv/post_quant_conv unconditionally unless told
    # otherwise, but Flux.1 and SD3 VAEs ship without them (our VaeConfig mirrors that with
    # UseQuantConv=false). Constructing them anyway and loading strict=False leaves a *randomly initialized*
    # 1x1 conv in the encode path, which silently scrambles mu/logvar -- the reference then disagrees with a
    # perfectly correct implementation at corr~0.02. Mirror the file instead of trusting the default.
    has_quant = "quant_conv.weight" in state
    has_post_quant = "post_quant_conv.weight" in state
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=tuple("DownEncoderBlock2D" for _ in block_out),
        up_block_types=tuple("UpDecoderBlock2D" for _ in block_out),
        block_out_channels=tuple(block_out),
        layers_per_block=2,
        latent_channels=latent_channels,
        sample_size=1024,
        use_quant_conv=has_quant,
        use_post_quant_conv=has_post_quant,
    )
    converted = convert_keys(state, vae)
    missing, unexpected = vae.load_state_dict(converted, strict=False)
    # Nothing on the encode path may be left at its random init -- that is the failure mode above.
    missing = [m for m in missing if m.startswith("encoder.") or "quant_conv" in m]
    if missing:
        raise SystemExit(f"{weights_path}: encode-path weights left uninitialized: {missing[:8]}")
    return vae.eval()


def translate_qwen_encoder(state: dict) -> dict:
    """Raw Comfy/WAN naming -> diffusers `AutoencoderKLQwenImage` encoder naming.

    Derived from the two key sets rather than guessed: every pairing below was confirmed by matching tensor
    shapes, and the caller then asserts nothing on the encode path is left uninitialized. The flat
    `downsamples.N` index carries straight over to `down_blocks.N`; only the sub-key vocabulary differs,
    because the checkpoint stores each residual block as an nn.Sequential (`residual.0/2/3/6`) while
    diffusers names the members.
    """
    residual = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}
    out: dict = {}
    for key, tensor in state.items():
        # Top-level 1x1x1 convs: raw `conv1` is the quantiser, `conv2` its decode-side inverse.
        if key.startswith("conv1."):
            out["quant_conv." + key[len("conv1."):]] = tensor
            continue
        if key.startswith("conv2."):
            out["post_quant_conv." + key[len("conv2."):]] = tensor
            continue
        if not key.startswith("encoder."):
            continue
        rest = key[len("encoder."):]

        if rest.startswith("conv1."):
            out["encoder.conv_in." + rest[len("conv1."):]] = tensor
        elif rest.startswith("head.0."):
            out["encoder.norm_out." + rest[len("head.0."):]] = tensor
        elif rest.startswith("head.2."):
            out["encoder.conv_out." + rest[len("head.2."):]] = tensor
        elif rest.startswith("middle."):
            # middle.0 / middle.2 are resnets 0 / 1; middle.1 is the attention block.
            idx, sub = rest[len("middle."):].split(".", 1)
            if idx == "1":
                out[f"encoder.mid_block.attentions.0.{sub}"] = tensor
            else:
                resnet = "0" if idx == "0" else "1"
                out[f"encoder.mid_block.resnets.{resnet}.{map_block_sub(sub, residual)}"] = tensor
        elif rest.startswith("downsamples."):
            idx, sub = rest[len("downsamples."):].split(".", 1)
            out[f"encoder.down_blocks.{idx}.{map_block_sub(sub, residual)}"] = tensor
    return out


def map_block_sub(sub: str, residual: dict) -> str:
    """Maps one residual-block sub-key; `resample` and `time_conv` pass through unchanged."""
    if sub.startswith("residual."):
        member, tail = sub[len("residual."):].split(".", 1)
        if member not in residual:
            raise SystemExit(f"unmapped residual member '{member}' in '{sub}'")
        return f"{residual[member]}.{tail}"
    if sub.startswith("shortcut."):
        return "conv_shortcut." + sub[len("shortcut."):]
    return sub


def infer_block_out_channels(state: dict) -> list:
    channels = []
    i = 0
    while True:
        key = f"encoder.down_blocks.{i}.resnets.0.conv1.weight"
        alt = f"encoder.down.{i}.block.0.conv1.weight"
        if key in state:
            channels.append(state[key].shape[0])
        elif alt in state:
            channels.append(state[alt].shape[0])
        else:
            break
        i += 1
    if not channels:
        raise SystemExit("could not infer block_out_channels from the state dict")
    return channels


def convert_keys(state: dict, vae) -> dict:
    """LDM -> diffusers key remap, mirroring CheckpointConvertUtils.ConvertVaeKey on the C# side."""
    # `encoder.down_blocks.` is diffusers-only naming (LDM uses `encoder.down.`), so its presence is a
    # reliable signal that the file needs no conversion. Sampling the first N keys is not: the sample
    # can land on bn./quant_conv keys that are absent from the module's own state dict.
    if any(k.startswith("encoder.down_blocks.") for k in state):
        return state
    try:
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        return convert_ldm_vae_checkpoint(state, vae.config)
    except Exception as ex:  # noqa: BLE001 - the converter is version-sensitive; fail loudly
        raise SystemExit(f"LDM->diffusers VAE key conversion failed: {ex}")


def dump(family: str) -> None:
    weights_path, cls_name, latent_channels, scaling, shift = FAMILIES[family]
    if not os.path.exists(weights_path):
        print(f"SKIP {family}: {weights_path} not found")
        return

    out_dir = os.path.join(OUT_ROOT, family)
    os.makedirs(out_dir, exist_ok=True)

    vae = build_vae(cls_name, weights_path, latent_channels)

    image = make_input(SIZE, SIZE)
    with torch.no_grad():
        if cls_name == "AutoencoderKLQwenImage":
            posterior = vae.encode(image.unsqueeze(2)).latent_dist  # [B,C,T,H,W], T=1
            mean = posterior.mean.squeeze(2)
        else:
            mean = vae.encode(image).latent_dist.mean
    # Qwen-Image normalizes per channel instead of by a scalar pair, and our QwenImageVaeEncoder implements that
    # inverse directly -- so the reference has to apply the same transform or it compares against the wrong thing.
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        lm = torch.tensor([float(v) for v in latents_mean], dtype=torch.float32).view(1, -1, 1, 1)
        ls = torch.tensor([float(v) for v in latents_std], dtype=torch.float32).view(1, -1, 1, 1)
        latent = (mean - lm) / ls
    else:
        latent = (mean - shift) * scaling

    save(image, os.path.join(out_dir, "image.bin"))
    save(mean, os.path.join(out_dir, "mean.bin"))
    save(latent, os.path.join(out_dir, "latent.bin"))

    index = {
        "family": family,
        "weights": os.path.relpath(weights_path, REPO_ROOT),
        "diffusers_class": cls_name,
        "scaling_factor": scaling,
        "shift_factor": shift,
        "image": stats(image),
        "mean": stats(mean),
        "latent": stats(latent),
    }
    # Per-key checksums of the *converted* encoder weights. When the latent comparison fails, these say
    # whether the cause is the LDM->diffusers key mapping (a key holding the wrong tensor) or the network
    # math -- without them a mismatch is just "wrong somewhere".
    index["encoder_weight_sums"] = {
        name: float(tensor.float().sum())
        for name, tensor in vae.state_dict().items()
        if name.startswith("encoder.") or name in ("quant_conv.weight", "quant_conv.bias")
    }

    # Qwen-Image normalizes per channel instead of by a scalar pair; record the vectors so the C# side
    # can check the mechanism it actually implements.
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        index["latents_mean"] = [float(v) for v in latents_mean]
        index["latents_std"] = [float(v) for v in latents_std]
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(f"OK   {family}: latent {list(latent.shape)} scale={scaling} shift={shift} -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="all", choices=[*FAMILIES, "all"])
    args = parser.parse_args()
    targets = list(FAMILIES) if args.family == "all" else [args.family]
    failures = []
    for name in targets:
        try:
            dump(name)
        except SystemExit as ex:
            print(f"FAIL {name}: {ex}")
            failures.append(name)
    if failures:
        raise SystemExit(f"failed: {', '.join(failures)}")
