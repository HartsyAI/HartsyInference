"""Reference dump of the FastFrames rollout from Apple's own code (ml-fastframes), for the C# parity tests.

Python only produces saved tensors here; it never enters the runtime. Run it on the CPU reference image (never the Windows
venv), with the upstream code at the pinned commit mounted or cloned:

    git clone https://github.com/apple-aiml-research/ml-fastframes <dir>
    git -C <dir> checkout a65c59240e01cf1de86197fa771ca09b38cf592a

    # tiny random-weight config (fixture for tests/HartsyInference.Diffusion.Tests/Fixtures/fastframes_tiny):
    python dump_fastframes_reference.py tiny --upstream <dir> --out <out-dir>

    # released weights (an FP32 CPU run of S is slow; pass --frames to keep it short):
    python dump_fastframes_reference.py checkpoint --upstream <dir> --model ff-s --checkpoints <dir-with-weights> \\
        --prompt-embeds <safetensors holding "prompt_embeds" [L, 4096]> --out <out-dir> --frames 4 --precision fp32

Upstream hard-codes BF16 in several places (the cuDNN attention casts q/k/v to BF16, the KV cache defaults to BF16, the refiner
input is cast to BF16). ``--precision fp32`` replaces ``torch.bfloat16`` with ``torch.float32`` before the upstream modules are
imported and forces PyTorch's own attention kernel, which gives an FP32 run of upstream's exact code: the reference the engine's
F32 path is compared with. ``--precision bf16`` keeps upstream's BF16 and only swaps the attention kernel (the cuDNN backend
does not exist on CPU), which gives the BF16 rounding floor.

Noise is saved, not seeded: upstream draws it from a CUDA generator in BF16, which nothing else reproduces. The dump writes the
initial and transition noise it used, and the rollout is run on exactly those arrays.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


def _load_state(path: Path, key: str | None = None):
    """Loads a state dict to F32 on the host. Under ``--precision fp32`` torch.bfloat16 is aliased to float32, which makes
    safetensors' torch path read BF16 tensors at the wrong width. So a safetensors file is parsed here: BF16 and F16 widen to F32
    bit-exactly (BF16 is the high 16 bits of F32), and the upstream key prefixes are stripped as upstream does. Other formats use
    upstream's loader."""
    import json
    import struct

    import numpy as np
    import torch
    from fastframes.checkpoints import _strip, load_state_dict

    if not str(path).endswith(".safetensors"):
        return load_state_dict(path, key)
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        base = 8 + header_len
        out = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            fh.seek(base + start)
            raw = fh.read(end - start)
            shape = tuple(info["shape"])
            if info["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            elif info["dtype"] == "F16":
                arr = np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
            elif info["dtype"] == "BF16":
                bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                arr = bits.view(np.float32).reshape(shape).copy()
            else:
                raise ValueError(f"{path}: {name} has dtype {info['dtype']}, not a float format the dump reads")
            out[_strip(name)] = torch.from_numpy(arr)
    return out


def configure(upstream: Path, precision: str):
    import torch
    # A --precision fp32 run replaces torch.bfloat16 process-wide. safetensors must be imported first, so that its dtype table is
    # bound to the real dtype before the alias exists; a BF16 checkpoint then still loads as BF16.
    import safetensors.torch  # noqa: F401

    if precision == "fp32":
        torch.bfloat16 = torch.float32  # before any upstream module binds a dtype default or builds a tensor
    sys.path.insert(0, str(upstream))
    # The pipeline module imports the umT5 text encoder at import time; the dump feeds saved prompt embeddings instead.
    stub = types.ModuleType("fastframes.wan.text_encoder")
    stub.TextEncoder = object
    sys.modules["fastframes.wan.text_encoder"] = stub

    from fastframes import pipeline
    from fastframes.wan import attention as attention_module
    from fastframes.wan import causal_model, model

    attention_module._cross_attention_backend = "auto"
    original = attention_module.attention

    def attention(q, k, v, causal=False, dropout_p=0.0, backend="auto"):
        return original(q, k, v, causal=causal, dropout_p=dropout_p, backend="auto")

    attention_module.attention = attention
    causal_model.attention = attention
    model.attention = attention
    model.cross_attention = lambda q, k, v: attention(q, k, v)
    return torch, pipeline, causal_model, model


def randomize(module, torch, seed: int, scale: float = 0.5):
    """Random, well-conditioned parameters (upstream zero-initialises heads and gates; a zero network proves nothing)."""
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in module.named_parameters():
        shape = parameter.shape
        if name.endswith(("norm_q.weight", "norm_k.weight")) or name.endswith("norm3.weight"):
            value = 1 + 0.1 * torch.randn(shape, generator=generator)
        elif name.endswith("bias") or name == "v_logscale" or name.endswith("_b"):
            value = 0.05 * torch.randn(shape, generator=generator)
        elif parameter.dim() == 1 and name.endswith("weight"):
            value = 1 + 0.1 * torch.randn(shape, generator=generator)
        elif parameter.dim() >= 2 and not name.endswith("modulation"):
            fan_in = shape[-2] if "heads_grouped" in name and parameter.dim() == 3 else parameter[0].numel()
            value = scale / fan_in**0.5 * torch.randn(shape, generator=generator)
        else:
            value = 0.3 * torch.randn(shape, generator=generator)
        parameter.data.copy_(value)


def save(tensors: dict, path: Path):
    from safetensors.torch import save_file

    save_file({k: v.detach().contiguous().cpu().float() for k, v in tensors.items()}, str(path))


def run_rollout(torch, pipeline, causal_model, config, dit, refiner, decoder, prompt_embeds, noise, transition, frames):
    class Replay(pipeline.FastFrames):
        def _noise(self, seed):
            return noise, lambda step, frame: transition[step, :, frame]

    ff = Replay(config, dit, refiner, decoder, text_encoder=None)
    latents, features, modulation = ff.rollout([prompt_embeds], seed=0, context="refiner")
    pixels = ff.decode(features, modulation)
    return latents, features, modulation, pixels


def tiny(args):
    torch, pipeline, causal_model, model = configure(Path(args.upstream), args.precision)
    from fastframes.models.kv_refinement import KVRefinementNetwork
    from fastframes.models.pixel_decoder import BridgedPixelDecoder

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wan = model.WanConfig(dim=64, ffn_dim=32, num_heads=2, num_layers=2, text_len=8, text_dim=16)
    frames, height, width = args.frames, 8, 12   # non-square on purpose: the released grid is 30x52
    config = pipeline.FastFramesConfig(
        name="tiny", wan_model="tiny", first_frame_timesteps=(1000.0, 937.5, 833.3333129882812, 625.0),
        timesteps=(1000.0, 833.3333129882812), refine_timestep=833.3333129882812, noise="sequential",
        autocast=False, fast_ops=False, decoder_dtype=torch.float32, contiguous_stem=True, allow_tf32=False,
        num_frames=frames, latent_size=(height, width), num_pixel_frames=4 * frames - 3)

    dit = causal_model.CausalWanModel(wan)
    randomize(dit, torch, seed=1)
    dit.eval().requires_grad_(False)
    refiner = KVRefinementNetwork(dim=wan.dim, num_heads=wan.num_heads, head_dim=wan.dim // wan.num_heads,
                                  num_blocks=wan.num_layers, d_a=16, n_stem=2)
    randomize(refiner, torch, seed=2)
    refiner.eval().requires_grad_(False)
    decoder = BridgedPixelDecoder(in_dim=wan.dim, dim=16)
    randomize(decoder, torch, seed=3)
    decoder.eval().requires_grad_(False)

    generator = torch.Generator().manual_seed(4)
    prompt_embeds = torch.randn(5, wan.text_dim, generator=generator)
    noise = torch.randn(1, frames, 16, height, width, generator=generator)
    transition = torch.randn(3, 1, frames, 16, height, width, generator=generator)

    latents, features, modulation, pixels = run_rollout(
        torch, pipeline, causal_model, config, dit, refiner, decoder, prompt_embeds, noise, transition, frames)

    save(dict(dit.state_dict()), out / "generator.safetensors")
    save(dict(refiner.state_dict()), out / "refiner.safetensors")
    save(dict(decoder.state_dict()), out / "decoder.safetensors")
    save({
        "prompt_embeds": torch.cat([prompt_embeds, prompt_embeds.new_zeros(wan.text_len - 5, wan.text_dim)]),
        "noise": noise[0], "transition": transition[:, 0],
        "latents": latents[0], "features": features[0].reshape(frames, -1, wan.dim),
        "modulation": modulation[0, :, 0], "pixels": pixels[0],
    }, out / "io.safetensors")
    print(f"tiny reference ({args.precision}): latents {tuple(latents.shape)} pixels {tuple(pixels.shape)} -> {out}")


def checkpoint(args):
    torch, pipeline, causal_model, model = configure(Path(args.upstream), args.precision)
    from safetensors.torch import load_file

    spec = pipeline.MODELS[args.model]
    root = Path(args.checkpoints)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    from fastframes.checkpoints import build_wan, load_state_dict
    from fastframes.models.kv_refinement import KVRefinementNetwork
    from fastframes.models.pixel_decoder import BridgedPixelDecoder, PixelDecoder

    # --dit-file names a generator in safetensors form (the engine's converted copy, prefix kept; upstream's loader strips it), so the
    # pickle is never loaded. Without it the generator is root/spec.dit, the upstream pickle, which the runbook forbids opening.
    dit_path = Path(args.dit_file) if args.dit_file else root / spec.dit
    dit = build_wan(spec.config.wan_model, load_state_dict(dit_path, "generator_ema"), dtype=dtype, device="cpu")
    wan = model.WAN_CONFIGS[spec.config.wan_model]
    refiner_state = _load_state(root / spec.refiner, "adapter")
    refiner = KVRefinementNetwork(dim=wan.dim, num_heads=wan.num_heads, head_dim=wan.dim // wan.num_heads,
                                  num_blocks=wan.num_layers, d_a=refiner_state["stem_in.weight"].shape[0]).to(dtype)
    refiner.load_state_dict(refiner_state, strict=True)
    decoder_state = _load_state(root / spec.decoder)
    if any(k.startswith("bridge.") for k in decoder_state):
        decoder = BridgedPixelDecoder(in_dim=wan.dim)
    else:
        decoder = PixelDecoder(wan.dim)
        decoder_state = {k.removeprefix("pixel_head."): v for k, v in decoder_state.items()}
    decoder.load_state_dict(decoder_state, strict=True)
    decoder = decoder.to(dtype).eval().requires_grad_(False)
    refiner.eval().requires_grad_(False)

    frames = args.frames
    config = spec.config.__class__(**{**spec.config.__dict__, "num_frames": frames, "num_pixel_frames": 4 * frames - 3,
                                      "decoder_dtype": dtype, "autocast": False, "fast_ops": False, "allow_tf32": False})
    prompt_embeds = load_file(args.prompt_embeds)["prompt_embeds"].to(dtype)
    if prompt_embeds.shape[0] < wan.text_len:
        # The C# side takes the umT5 output already zero-padded to the generator's text length, as upstream does.
        pad = prompt_embeds.new_zeros(wan.text_len - prompt_embeds.shape[0], prompt_embeds.shape[1])
        prompt_embeds = torch.cat([prompt_embeds, pad])
    generator = torch.Generator().manual_seed(args.seed)
    height, width = config.latent_size
    noise = torch.randn(1, frames, 16, height, width, generator=generator).to(dtype)
    transition = torch.randn(3, 1, frames, 16, height, width, generator=generator).to(dtype)

    latents, features, modulation, pixels = run_rollout(
        torch, pipeline, causal_model, config, dit, refiner, decoder, prompt_embeds, noise, transition, frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save({
        "prompt_embeds": prompt_embeds, "noise": noise[0], "transition": transition[:, 0],
        "latents": latents[0], "features": features[0].reshape(frames, -1, wan.dim),
        "modulation": modulation[0, :, 0], "pixels": pixels[0],
    }, out / "io.safetensors")
    print(f"{args.model} reference ({args.precision}, {frames} frames): pixels {tuple(pixels.shape)} -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tiny")
    t.add_argument("--upstream", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--frames", type=int, default=3)
    t.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    t.set_defaults(func=tiny)
    c = sub.add_parser("checkpoint")
    c.add_argument("--upstream", required=True)
    c.add_argument("--model", choices=["ff-s", "ff-l"], required=True)
    c.add_argument("--checkpoints", required=True)
    c.add_argument("--prompt-embeds", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--frames", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    c.add_argument("--dit-file", default=None,
                   help="the generator as .safetensors (upstream's loader strips the prefix); default root/<spec.dit>, a pickle")
    c.set_defaults(func=checkpoint)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
