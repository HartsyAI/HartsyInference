"""
MOSS-Image references from OpenMOSS's own ComfyUI module (MOSS-Image comfyui/moss-image), in two modes.

verify   Re-runs the tiny fixture moss_image_numpy_reference.py wrote (its stored F16-rounded weights and inputs)
         through MOSSImageTransformer itself and checks the fixture's outputs against it: the numpy transcription the
         C# parity tests trust is then checked against the real module, not just against itself. Exits non-zero when
         a case drifts past the tolerance.

real     Dumps real-weight references from an OpenMOSS diffusers folder (transformer/, text_encoder/, tokenizer/,
         vae/) through the node's own HF loaders, for MossImageRealWeightTests:
           prompt.ids / negative.ids       the Qwen3.5 ids of PROMPT and the empty negative (chat template applied)
           prompt.hidden / negative.hidden the encoder's final normalized hidden states, [1, L, 4096]
           inputs.latent                   seeded noise as packed tokens [1, H*W, 128] (raster order)
           inputs.timesteps                the timesteps of the velocity cases (sigma * 1000)
           cond.velocity.N                 prompt conditioning, every block
           uncond.velocity.N               negative conditioning with block 22 skipped (the reference's two-pass SLG)
         The DiT runs in float32 by default (--dtype bfloat16 halves its memory); ids are stored as float32, exact
         below 2^24.

Usage:
  python -I dump_moss_image_reference.py verify <ComfyUI checkout> <MOSS-Image checkout> <moss_image_tiny.safetensors>
  python -I dump_moss_image_reference.py real <ComfyUI checkout> <MOSS-Image checkout> <MOSS-Image-Base folder> <output .safetensors>
         [--device cuda] [--dtype float32|bfloat16] [--size 512]
Requires: torch, safetensors, transformers, ComfyUI's requirements.
"""
import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROMPT = ("A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the "
          "refrigerator door is handwritten in Chinese: “记得买牛奶和鸡蛋！”")
SKIP_LAYER = 22
TIMESTEPS = (1000.0, 630.0, 24.0)
SEED = 20261016
TOLERANCE = 2e-4
FIXTURE_GRID = (3, 4)  # moss_image_numpy_reference.py GRID


def comfy_path(comfy_root):
    sys.path.insert(0, str(Path(comfy_root).resolve()))
    import comfy.cli_args

    return comfy.cli_args


def node_package(moss_root):
    """The node folder as a package, without running its __init__ (which registers the UI nodes)."""
    folder = Path(moss_root).resolve() / "comfyui" / "moss-image"
    package = types.ModuleType("moss_image_node")
    package.__path__ = [str(folder)]
    sys.modules["moss_image_node"] = package
    return folder


def skip_patch(layers):
    def passthrough(args, extra):
        return {"img": args["img"]}

    return {"patches_replace": {"dit": {("single_block", layer): passthrough for layer in layers}}}


def packed(spatial):
    return spatial.flatten(2).transpose(1, 2).contiguous()


def spatial(tokens, height, width):
    return tokens.transpose(1, 2).reshape(1, -1, height, width).contiguous()


def verify(args):
    comfy_path(args.comfy).args.cpu = True
    import comfy.ops

    node_package(args.moss)
    model_module = importlib.import_module("moss_image_node.model")
    fixture = load_file(args.fixture)
    weights = {k[len("weights."):]: v.float() for k, v in fixture.items() if k.startswith("weights.")}
    dim = weights["video_embedder.bias"].shape[0]
    head = weights["layers.0.q_norm.weight"].shape[0]
    layers = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("layers."))
    model = model_module.MOSSImageTransformer(
        dim=dim, ffn_dim=weights["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2, num_heads=dim // head,
        num_layers=layers, text_dim=weights["text_embedder.proj.weight"].shape[1],
        freq_dim=weights["time_embedding.0.weight"].shape[1], latent_channels=weights["final_linear_video.weight"].shape[0],
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    model.load_state_dict(weights, strict=True)
    model.eval()

    latent, context = fixture["inputs.latent"].float(), fixture["inputs.context"].float()
    height, width = FIXTURE_GRID
    x = spatial(latent, height, width)
    failures = 0
    with torch.no_grad():
        rope = model.rope(height, width, context.shape[1], "cpu")
        cases = {
            "full.velocity": packed(model(x, 500.0, context)),
            "skip.velocity": packed(model(x, 937.0, context, transformer_options=skip_patch([1]))),
            "text.refined": model.text_embedder(context, rope[:, height * width:]),
        }
    for name, value in cases.items():
        expected = fixture[name].float()
        worst = (value - expected).abs().max().item()
        bound = TOLERANCE * max(1.0, expected.abs().max().item())
        status = "ok" if worst <= bound else "DRIFT"
        failures += status != "ok"
        print(f"{name}: max |delta| = {worst:.3e} (bound {bound:.3e}) {status}")
    return 1 if failures else 0


def real(args):
    cli = comfy_path(args.comfy).args
    if args.device == "cpu":
        cli.cpu = True
    node_package(args.moss)
    hf = importlib.import_module("moss_image_node.hf")
    root = str(Path(args.folder).resolve())
    tensors = {}

    clip = hf.load_hf_clip(root, "cpu" if args.device == "cpu" else "default", disable_dynamic=True)
    hidden = {}
    for label, text in (("prompt", PROMPT), ("negative", "")):
        tokens = clip.tokenize(text)
        ids = [token[0] for token in tokens["qwen35_9b"][0]]
        tensors[f"{label}.ids"] = torch.tensor(ids, dtype=torch.float32)
        hidden[label] = clip.encode_from_tokens(tokens).float().cpu()
        tensors[f"{label}.hidden"] = hidden[label].contiguous()
        print(f"{label}: {len(ids)} ids, hidden {tuple(hidden[label].shape)}")
    del clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dtype = getattr(torch, args.dtype)
    patcher = hf.load_hf_model(root, "default", disable_dynamic=True)
    dit = patcher.model.diffusion_model.to(device=args.device, dtype=dtype).eval()
    grid = args.size // 16
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(1, dit.latent_channels, grid, grid, generator=generator)
    tensors["inputs.latent"] = packed(x)
    tensors["inputs.timesteps"] = torch.tensor(TIMESTEPS, dtype=torch.float32)
    device_x = x.to(args.device, dtype)
    cond = hidden["prompt"].to(args.device, dtype)
    uncond = hidden["negative"].to(args.device, dtype)
    with torch.no_grad():
        for index, timestep in enumerate(TIMESTEPS):
            tensors[f"cond.velocity.{index}"] = packed(dit(device_x, timestep, cond).float().cpu())
            tensors[f"uncond.velocity.{index}"] = packed(
                dit(device_x, timestep, uncond, transformer_options=skip_patch([SKIP_LAYER])).float().cpu())
    save_file({k: v.contiguous() for k, v in tensors.items()}, args.output, metadata={"prompt": PROMPT})
    for key in sorted(tensors):
        t = tensors[key].double()
        print(f"{key}: {tuple(t.shape)} mean={t.mean():.6f} std={t.std():.6f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    check = modes.add_parser("verify")
    check.add_argument("comfy")
    check.add_argument("moss")
    check.add_argument("fixture")
    dump = modes.add_parser("real")
    dump.add_argument("comfy")
    dump.add_argument("moss")
    dump.add_argument("folder")
    dump.add_argument("output")
    dump.add_argument("--device", default="cuda")
    dump.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    dump.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    sys.exit(verify(args) if args.mode == "verify" else real(args))


if __name__ == "__main__":
    main()
