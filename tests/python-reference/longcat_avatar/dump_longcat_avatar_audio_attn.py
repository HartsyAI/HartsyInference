"""Reference dump for the avatar audio cross-attention (SingleStreamAttention, modules/avatar/attention.py @ 58eb336).

Runs the pinned class on a tiny geometry with seeded weights and inputs, for a text-to-video call (no conditioning frames) and an
image-to-video call (one conditioning frame, which the reference returns as zeros). Writes the weights, the inputs and both outputs as raw
F32 files plus meta.json (the layout LongCatReferenceDump.Load reads).

Run in the hartsy-ref:cpu image only (never the Windows host). Assemble the pinned source (longcat_video/) in a scratch folder, then:

    python dump_longcat_avatar_audio_attn.py --repo /src --out /out/longcat_avatar_attn_reference_tensors

Stubbed, so NOT used by the reference path under test:
- flash_attn.flash_attn_func: replaced by torch SDPA with the same layout and softmax scale (the reference's own attention call site is kept;
  without one of its attention flags the class has no attention path at all). Mirrors the B1 and B4 dumps.
- audio_process.torch_utils (librosa, pyloudnorm, torchvision) and block_sparse_attention.bsa_interface (triton): only used by the
  multi-speaker reference map and by the block-sparse path, neither of which this dump enables.
"""

import argparse
import importlib
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F


def install_stubs(repo):
    flash = types.ModuleType("flash_attn")

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **_):
        # q, k, v: [B, S, H, D] (flash layout). Non-causal softmax over the keys.
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale
        )
        return out.transpose(1, 2)

    flash.flash_attn_func = flash_attn_func
    sys.modules["flash_attn"] = flash

    for name in ("longcat_video.audio_process.torch_utils", "longcat_video.block_sparse_attention.bsa_interface"):
        stub = types.ModuleType(name)
        stub.get_attn_map_with_target = None
        stub.flash_attn_bsa_3d = None
        sys.modules[name] = stub


def write_tensor(out, meta, name, array):
    array = np.ascontiguousarray(array.detach().cpu().numpy().astype(np.float32))
    array.tofile(os.path.join(out, name + ".f32"))
    meta["tensors"][name] = list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    install_stubs(args.repo)
    attention = importlib.import_module("longcat_video.modules.avatar.attention")

    # Tiny geometry: dim 16, 2 heads of 8, encoder width 12, 5 audio tokens per frame, 4 latent frames of 6 patch tokens.
    dim, heads, enc, ctx, frames, tokens = 16, 2, 12, 5, 4, 6
    torch.manual_seed(args.seed)
    layer = attention.SingleStreamAttention(
        dim=dim, encoder_hidden_states_dim=enc, num_heads=heads, qkv_bias=True, qk_norm=True,
        class_range=24, class_interval=4, cp_split_hw=[1, 1], enable_flashattn2=True,
    ).eval()
    with torch.no_grad():
        for name, p in layer.named_parameters():
            p.copy_(torch.randn_like(p) * 0.2 + (1.0 if "norm" in name else 0.0))

    x = torch.randn(1, frames * tokens, dim)
    cond = torch.randn(frames, ctx, enc)  # [(B N_t), ctx, enc] with B = 1
    shape = (frames, 2, 3)  # (N_t, H, W) in latent tokens; only N_t is read here

    os.makedirs(args.out, exist_ok=True)
    meta = {"dims": dict(dim=dim, heads=heads, enc=enc, ctx=ctx, frames=frames, tokens=tokens), "tensors": {}}
    for key, value in layer.state_dict().items():
        write_tensor(args.out, meta, "attn." + key, value)
    write_tensor(args.out, meta, "inputs.x", x.reshape(frames * tokens, dim))
    write_tensor(args.out, meta, "inputs.audio", cond.reshape(frames * ctx, enc))

    with torch.no_grad():
        _, t2v = layer(x, cond, shape=shape, num_cond_latents=0)
        cond_out, i2v = layer(x, cond, shape=shape, num_cond_latents=1)
    write_tensor(args.out, meta, "output.t2v", t2v.reshape(frames * tokens, dim))
    write_tensor(args.out, meta, "output.i2v", i2v.reshape((frames - 1) * tokens, dim))
    write_tensor(args.out, meta, "output.i2v_cond", cond_out.reshape(tokens, dim))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps({"t2v": list(t2v.shape), "i2v": list(i2v.shape), "i2v_cond": list(cond_out.shape)}))


if __name__ == "__main__":
    main()
