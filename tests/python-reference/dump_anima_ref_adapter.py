"""
Dumps the Anima reference adapter's reference path from ComfyUI, for the engine's reference parity tests.

The reference is ComfyUI-Anima-Ref-Adapter's own code, called directly: its `_encode_ref_asym`, `_asym_forward`,
`_frame_positions`, `_rope_from_positions` and `_correct`. The network is ComfyUI's Anima (MiniTrainDIT) loaded from the base
checkpoint through `comfy.sd.load_diffusion_model_state_dict`, and the adapter's LoRA is installed with the node's own `_Adapter`,
so the dump includes the LoRA the way the node applies it: live during the reference prepass and the reference-conditioned call.

Everything runs in F32 with autograd off. Inputs are random but seeded, and every input the engine consumes is written to the file,
so both sides see identical tensors: the target latent, the reference latent (both already in the model's latent space, so no VAE
is involved), the post-adapter text context (512 tokens, as the pipeline pads it), and the timestep.

Outputs (safetensors keys):
  in.latent      [1, 16, H, W]     target latent
  in.reference   [1, 16, Hr, Wr]   reference latent
  in.context     [1, 512, 1024]    text context fed to the blocks
  in.sigma       [1]               timestep
  ref.k.{i}      [S_ref, heads, D] reference keys of block i, after QK-norm and RoPE
  ref.v.{i}      [S_ref, heads, D] reference values of block i
  rope.ref.cos   [S_ref, D/2]      RoPE cos per pair, the reference's (t=1, h+row, w) positions
  rope.ref.sin   [S_ref, D/2]
  rope.target.cos [S, D/2]         the target's (t=0, h, w) positions
  rope.target.sin [S, D/2]
  out.velocity   [1, 16, H, W]     target output with the reference joined
  corr.guided / corr.pos           inputs for the correction fixtures
  corr.apg / corr.rescale / corr.none   _correct outputs, with rescale_phi 0.5

Usage (run with the ComfyUI checkout's python environment, from anywhere):
  python -I dump_anima_ref_adapter.py \\
      --comfy <ComfyUI checkout at b5a4321> \\
      --node  <ComfyUI-Anima-Ref-Adapter/__init__.py at fb587d2> \\
      --checkpoint <anima-base-v1.0.safetensors> \\
      --adapter <anima_ref_adapter_v2_r64.safetensors> \\
      --out <dump.safetensors>

Requirements: torch, safetensors, numpy, pillow, huggingface_hub, einops, and ComfyUI's requirements. Peak memory is the full
model in F32, about 8 GB of weights plus activations. The script always runs on the CPU (it sets `cpu=True`), so it needs that much RAM and no GPU.

Status: written against the sources pinned above; not yet executed. Run it before trusting any parity result built on it.
"""
import argparse
import importlib.util
import sys

import torch
from safetensors.torch import save_file

ADAPTER_SLOT_OFFSET_H = 1.0  # the released adapter's "character" slot: offset_h 1.0, offset_w 0 (metadata "slots")
ADAPTER_SLOT_OFFSET_W = 0
OFFSET_RATIO = 1.0           # the node's ref_offset_h default
TEXT_TOKENS = 512            # the pipeline pads the adapter output to 512 (AnimaRecipe.T5MaxTokens)
TEXT_WIDTH = 1024
CHANNELS = 16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy", required=True, help="ComfyUI checkout (b5a4321), put on sys.path for comfy.* imports")
    parser.add_argument("--node", required=True, help="ComfyUI-Anima-Ref-Adapter/__init__.py (fb587d2)")
    parser.add_argument("--checkpoint", required=True, help="anima-base-v1.0.safetensors, the single-file checkpoint")
    parser.add_argument("--adapter", required=True, help="anima_ref_adapter_v2_r64.safetensors")
    parser.add_argument("--out", required=True, help="output .safetensors")
    parser.add_argument("--latent-h", type=int, default=32, help="target latent height (divisible by 2)")
    parser.add_argument("--latent-w", type=int, default=40, help="target latent width (divisible by 2)")
    parser.add_argument("--ref-h", type=int, default=24, help="reference latent height (divisible by 2)")
    parser.add_argument("--ref-w", type=int, default=32, help="reference latent width (divisible by 2)")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--sigma", type=float, default=0.625)
    return parser.parse_args()


def load_comfy(comfy_root):
    sys.path.insert(0, comfy_root)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True  # no device placement: the dump is F32 on whatever device torch picks below
    import comfy.model_management  # noqa: F401  (ComfyUI's device bookkeeping must be initialized before loading)


def load_node(path):
    spec = importlib.util.spec_from_file_location("anima_ref_adapter_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_network(checkpoint, device):
    import comfy.sd
    import comfy.utils

    state_dict = comfy.utils.load_torch_file(checkpoint, safe_load=True)
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options={"dtype": torch.float32, "load_device": device})
    if patcher is None:
        raise RuntimeError(f"ComfyUI did not recognize {checkpoint} as a diffusion model")
    network = patcher.model.diffusion_model.to(device=device, dtype=torch.float32).eval()
    if type(network).__name__ != "Anima":
        raise RuntimeError(f"expected ComfyUI's Anima network, got {type(network).__name__}")
    return network


def main():
    args = parse_args()
    load_comfy(args.comfy)
    node = load_node(args.node)
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    generator = torch.Generator().manual_seed(args.seed)

    network = load_network(args.checkpoint, device)
    adapter = node._Adapter.get(args.adapter, 1.0)  # the node's own loader: weights, metadata, and the arch/slots it reads
    adapter.install(network)
    if not adapter.asym or not adapter.ref_lora:
        raise RuntimeError("this dump follows the asym reference path with the reference LoRA; the adapter's metadata says otherwise")

    latent = torch.randn(1, CHANNELS, args.latent_h, args.latent_w, generator=generator, device=device)
    reference = torch.randn(1, CHANNELS, args.ref_h, args.ref_w, generator=generator, device=device)
    context = torch.randn(1, TEXT_TOKENS, TEXT_WIDTH, generator=generator, device=device)
    sigma = torch.tensor([args.sigma], device=device)

    # The node's shapes: the target and reference are 5D [B, C, T=1, H, W]; the grid height is the patched one.
    x5 = latent.unsqueeze(2)
    hp = -(-args.latent_h // network.patch_spatial)
    wp = -(-args.latent_w // network.patch_spatial)
    ref_offset = (round(ADAPTER_SLOT_OFFSET_H * OFFSET_RATIO * hp), ADAPTER_SLOT_OFFSET_W)

    # The reference prepass runs with the LoRA live, as the node's _asym_ref_kv does.
    node._active["key"] = adapter.key
    try:
        ref_kv = node._encode_ref_asym(network, reference.unsqueeze(2), ref_offset)
    finally:
        node._active["key"] = None

    # The target forward with the reference joined, as the node's _make_wrapper does.
    node._active["key"] = adapter.key
    try:
        out = node._asym_forward(network, x5, sigma, context, ref_kv, {})
    finally:
        node._active["key"] = None

    tensors = {
        "in.latent": latent.contiguous(),
        "in.reference": reference.contiguous(),
        "in.context": context.contiguous(),
        "in.sigma": sigma.contiguous(),
        "out.velocity": out[:, :, 0].contiguous(),
    }
    for i, (keys, values) in enumerate(ref_kv):
        # The node returns [B, S, heads, D] per block; the engine's cache holds [S, heads, D] for batch 1.
        tensors[f"ref.k.{i}"] = keys.reshape(keys.shape[-3], keys.shape[-2], keys.shape[-1]).contiguous()
        tensors[f"ref.v.{i}"] = values.reshape(values.shape[-3], values.shape[-2], values.shape[-1]).contiguous()

    ref_grid_h = -(-args.ref_h // network.patch_spatial)
    ref_grid_w = -(-args.ref_w // network.patch_spatial)
    t, h, w = node._frame_positions(1, ref_grid_h, ref_grid_w, ref_offset[0], ref_offset[1], device)
    tensors.update(rope_tables("rope.ref", node, network, t, h, w))
    t, h, w = node._frame_positions(0, hp, wp, 0, 0, device)
    tensors.update(rope_tables("rope.target", node, network, t, h, w))

    guided = torch.randn(1, CHANNELS, args.latent_h, args.latent_w, generator=generator, device=device)
    pos = torch.randn(1, CHANNELS, args.latent_h, args.latent_w, generator=generator, device=device)
    tensors["corr.guided"] = guided.contiguous()
    tensors["corr.pos"] = pos.contiguous()
    for mode in ("apg", "rescale", "none"):
        tensors[f"corr.{mode}"] = node._correct(guided, pos, mode, 0.5).contiguous()

    metadata = {
        "source": "ComfyUI-Anima-Ref-Adapter (node) on ComfyUI Anima",
        "ref_offset": f"{ref_offset[0]},{ref_offset[1]}",
        "seed": str(args.seed),
        "sigma": str(args.sigma),
        "latent": f"{args.latent_h}x{args.latent_w}",
        "reference": f"{args.ref_h}x{args.ref_w}",
    }
    save_file(tensors, args.out, metadata=metadata)
    print(f"wrote {len(tensors)} tensors to {args.out}")


def rope_tables(prefix, node, network, t, h, w):
    """The node's rotation matrices per position, reduced to cos and sin per pair: [S, D/2], in the (t, h, w) pair order the engine uses."""
    emb = node._rope_from_positions(network.pos_embedder, t, h, w)  # [S, pairs, 2, 2]
    cos = emb[..., 0, 0].contiguous()
    sin = emb[..., 1, 0].contiguous()
    return {f"{prefix}.cos": cos, f"{prefix}.sin": sin}


if __name__ == "__main__":
    main()
