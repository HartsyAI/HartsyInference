"""Reference values for the X-Part DiT, produced by upstream's own modules on the CPU, block by block.

Runs `PartFormerDitBlock`, `LocalGlobalProcessor`, `TimestepEmbedder` and `FinalLayer` from `XPart/partgen/models/partformer_dit.py` (and `MoEBlock` from `moe_layers.py`) of a
checkout of tencent/Hunyuan3D-Part (pinned in docs/Research/HUNYUAN3D_PART_ARCHITECTURE.md) with the released `model/model.safetensors`. The whole DiT is 6.6 GB of BF16 (13 GB as
float32 on both sides), so the dump covers the block kinds one at a time instead of one forward pass: layers 0 (local self-attention, dense MLP), 1 (global self-attention), 11 (U-skip,
global), 15 (MoE, skip, global) and 16 (MoE, skip, local), plus the input embedding (`x_embedder`, the time token, `part_embed`) and the final layer. Everything that differs from a normal run
keeps the dump safe and dependency-free: `einops` is replaced by the one pattern upstream uses, and the package relative import is replaced by the executed `moe_layers.py`.

    python dump_xpart_dit.py --upstream <checkout>/XPart/partgen/models --weights <model.safetensors> --out <fixture.safetensors> --subset <subset.safetensors>

The fixture holds the inputs and outputs (`block.<L>.x`, `.ctx1`, `.ctx2`, `.skip`, `.out`; `embed.latents`, `.sigma`, `.ids`, `.out`; `final.x`, `.out`), the subset file the BF16 tensors of
those layers and of the embedding and final layer (the C# test loads it instead of the whole model). Upstream draws the part ids at random (`torch.randperm`); the dump pins them.
"""
import argparse
import logging
import os
import time
import types

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

K, TOKENS, CTX1, CTX2 = 3, 16, 24, 20  # parts, latent tokens (plus the time token), the two context lengths
WIDTH, HEADS, DEPTH = 2048, 16, 21
LAYERS = (0, 1, 11, 15, 16)


def load_upstream(folder):
    def _rearrange(t, pattern, h=None):
        assert pattern == "b n h d -> b h n d", pattern
        return t.permute(0, 2, 1, 3)

    moe = types.ModuleType("moe_layers")
    with open(os.path.join(folder, "moe_layers.py"), encoding="utf-8") as handle:
        exec(compile(handle.read(), "moe_layers.py", "exec"), moe.__dict__)
    with open(os.path.join(folder, "partformer_dit.py"), encoding="utf-8") as handle:
        source = handle.read()
    source = source.replace("from einops import rearrange", "rearrange = _rearrange")
    source = source.replace("from .moe_layers import MoEBlock", "MoEBlock = _moe.MoEBlock")
    module = types.ModuleType("partformer_dit")
    module.__dict__.update({"_rearrange": _rearrange, "_moe": moe})
    exec(compile(source, "partformer_dit.py", "exec"), module.__dict__)
    return module


def read(handle, prefix):
    return {k[len(prefix):]: handle.get_tensor(k).float() for k in handle.keys() if k.startswith(prefix)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subset", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    logging.disable(logging.CRITICAL)
    dit = load_upstream(args.upstream)
    out, subset = {}, {}
    handle = safe_open(args.weights, framework="pt")

    for layer in LAYERS:
        start = time.time()
        block = dit.PartFormerDitBlock(
            WIDTH, HEADS, use_self_attention=True, use_cross_attention=True, use_cross_attention_2=True, encoder_hidden_dim=1024, encoder_hidden2_dim=1024,
            qkv_bias=False, qk_norm=True, norm_layer=nn.LayerNorm, qk_norm_layer=nn.RMSNorm, skip_connection=layer > DEPTH // 2, use_moe=DEPTH - layer <= 6, num_experts=8, moe_top_k=2,
        )
        if (layer + 1) % 2 == 0:
            block.attn1.processor = dit.LocalGlobalProcessor(use_global=True)
        prefix = f"blocks.{layer}."
        block.load_state_dict(read(handle, prefix))
        block.eval()
        for k in handle.keys():
            if k.startswith(prefix):
                subset[k] = handle.get_tensor(k)
        x = torch.randn(K, TOKENS + 1, WIDTH)
        ctx1, ctx2 = torch.randn(K, CTX1, 1024), torch.randn(K, CTX2, 1024)
        skip = torch.randn(K, TOKENS + 1, WIDTH) if block.skip_linear is not None else None
        with torch.no_grad():
            y = block(hidden_states=x, encoder_hidden_states=ctx1, encoder_hidden_states_2=ctx2, temb=None, skip_value=skip)
        out[f"block.{layer}.x"], out[f"block.{layer}.ctx1"], out[f"block.{layer}.ctx2"], out[f"block.{layer}.out"] = x, ctx1, ctx2, y
        if skip is not None:
            out[f"block.{layer}.skip"] = skip
        print(f"block {layer}: global={(layer + 1) % 2 == 0} moe={block.use_moe} skip={skip is not None} in {time.time() - start:.1f} s, out std {y.std():.3f}")

    # Input embedding and final layer: a one-block model, whose other tensors stay unused.
    model = dit.PartFormerDITPlain(
        input_size=TOKENS, in_channels=64, hidden_size=WIDTH, depth=1, num_heads=HEADS, qk_norm=True, qkv_bias=False, qk_norm_type="rms", use_part_embed=True, valid_num=50,
        num_moe_layers=0, use_self_attention=True, use_cross_attention=True, use_cross_attention_2=True,
    )
    edge = {k: v for k, v in read(handle, "").items() if not k.startswith("blocks.")}
    for k in edge:
        subset[k] = handle.get_tensor(k)
    print(model.load_state_dict(edge, strict=False))
    model.eval()
    latents, sigma, ids = torch.randn(K, TOKENS, 64), 0.37, torch.tensor([7, 0, 31])
    with torch.no_grad():
        t = model.t_embedder(torch.full((K,), sigma), condition=None)
        x = model.x_embedder(latents) + model.part_embed[ids].unsqueeze(1)
        embedded = torch.cat([t, x], dim=1)
        hidden = torch.randn(K, TOKENS + 1, WIDTH)
        final = model.final_layer(hidden)
    out["embed.latents"], out["embed.sigma"], out["embed.ids"], out["embed.out"] = latents, torch.tensor([sigma]), ids.float(), embedded
    out["final.x"], out["final.out"] = hidden, final
    print("embed", tuple(embedded.shape), "final", tuple(final.shape))
    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    save_file({k: v.contiguous() for k, v in subset.items()}, args.subset)
    print("wrote", args.out, args.subset)


if __name__ == "__main__":
    main()
