"""Per-stage reference tensors of the LongCat-Video DiT on a tiny random checkpoint, from the pinned reference modules, on CPU in F32.

Runs the reference's own LongCatVideoTransformer3DModel (and, for the avatar reference/skip cases, its avatar Attention swapped into the
same blocks) with flash-attn replaced by torch SDPA, which is the only change. Writes a tiny sharded checkpoint under the reference key
names (the C# loader reads it as it reads the real one), the saved inputs of every scenario, and every stage the C# test compares.

Usage:
    python dump_longcat_dit.py --repo /path/to/LongCat-Video [--out DIR] [--seed 7]
Needs torch >= 2.6 (CPU is enough), einops, safetensors, numpy, and a checkout of meituan-longcat/LongCat-Video at
58eb336edc5c8c63389838727e40b8c18434e54e. Does not need CUDA, triton, flash-attn, transformers or any real weight.

Stubs, none of which touches the numerics: flash_attn (SDPA), diffusers (only its config base classes), the triton block-sparse module (never reached: enable_bsa is False),
audio_process.torch_utils (multi-speaker only), and ulysses_wrapper (identity: no context parallel). cp_split_hw is [1, 1] because the
shipped null crashes the RoPE module's size check.
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

LONGCAT_COMMIT = "58eb336edc5c8c63389838727e40b8c18434e54e"
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "longcat_dit_reference_tensors")

CONFIG = dict(
    in_channels=4, out_channels=4, hidden_size=128, depth=2, num_heads=4, caption_channels=48, mlp_ratio=4,
    adaln_tembed_dim=64, frequency_embedding_size=64, patch_size=(1, 2, 2), enable_flashattn2=True, cp_split_hw=[1, 1],
    text_tokens_zero_pad=True,
)
TEXT_LEN = 7
VALID_TEXT = 5

# frames/h/w are latent sizes; timesteps are per latent frame (conditioning frames 0, as the pipelines set them).
SCENARIOS = [
    dict(name="t2v", frames=5, h=6, w=4, num_cond=0, num_ref=0, ref_index=0, mask_range=0, timesteps=[900, 780, 640, 510, 380]),
    dict(name="i2v", frames=5, h=6, w=4, num_cond=1, num_ref=0, ref_index=0, mask_range=0, timesteps=[0, 850, 700, 560, 420]),
    dict(name="vc", frames=7, h=4, w=6, num_cond=2, num_ref=0, ref_index=0, mask_range=0, timesteps=[0, 0, 820, 700, 590, 480, 350]),
    dict(name="avatar_ref_skip", frames=9, h=4, w=4, num_cond=3, num_ref=1, ref_index=4, mask_range=1,
         timesteps=[0, 0, 0, 900, 820, 740, 660, 580, 500]),
    dict(name="avatar_ref_fallback", frames=9, h=4, w=4, num_cond=3, num_ref=1, ref_index=10, mask_range=1,
         timesteps=[0, 0, 0, 900, 820, 740, 660, 580, 500]),
    dict(name="avatar_ref_i2v", frames=5, h=6, w=4, num_cond=1, num_ref=1, ref_index=4, mask_range=0, timesteps=[0, 850, 700, 560, 420]),
]


def install_stubs(repo):
    sys.path.insert(0, repo)

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **_):  # [B, S, H, D]
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
        return out.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **_):  # packed [S, H, D], one segment
        assert len(cu_seqlens_q) == 2 and len(cu_seqlens_k) == 2, "the dump runs batch 1"
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        return out[0].transpose(0, 1).contiguous()

    flash = types.ModuleType("flash_attn")
    flash.flash_attn_func = flash_attn_func
    flash.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = flash

    bsa = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")

    def refuse(*_a, **_k):
        raise RuntimeError("block-sparse attention is not part of this dump")

    bsa.flash_attn_bsa_3d = refuse
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = bsa

    torch_utils = types.ModuleType("longcat_video.audio_process.torch_utils")
    torch_utils.get_attn_map_with_target = refuse
    sys.modules["longcat_video.audio_process.torch_utils"] = torch_utils

    # diffusers is only the config/model base class here: register_to_config just calls __init__, ModelMixin is nn.Module.
    config_utils = types.ModuleType("diffusers.configuration_utils")
    config_utils.ConfigMixin = type("ConfigMixin", (), {})
    config_utils.register_to_config = lambda init: init
    modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
    modeling_utils.ModelMixin = torch.nn.Module
    models = types.ModuleType("diffusers.models")
    diffusers = types.ModuleType("diffusers")
    diffusers.ConfigMixin, diffusers.ModelMixin = config_utils.ConfigMixin, modeling_utils.ModelMixin
    for name, module in (("diffusers", diffusers), ("diffusers.configuration_utils", config_utils), ("diffusers.models", models),
                         ("diffusers.models.modeling_utils", modeling_utils)):
        sys.modules.setdefault(name, module)

    ulysses = types.ModuleType("longcat_video.context_parallel.ulysses_wrapper")
    ulysses.ulysses_wrapper = lambda func: func
    sys.modules["longcat_video.context_parallel.ulysses_wrapper"] = ulysses


def randomize(model, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=gen)
            if name.endswith("norm.weight") or name.endswith("pre_crs_attn_norm.weight"):
                p.copy_(1.0 + 0.2 * noise)
            else:
                p.copy_(0.07 * noise)


def save_checkpoint(model, out_dir):
    from safetensors.torch import save_file

    ckpt = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt, exist_ok=True)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    names = sorted(state)
    shard_files = ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"]
    weight_map, total = {}, 0
    for s, file in enumerate(shard_files):
        part = {k: state[k] for i, k in enumerate(names) if i % 2 == s}
        save_file(part, os.path.join(ckpt, file))
        for k, v in part.items():
            weight_map[k] = file
            total += v.numel() * v.element_size()
    with open(os.path.join(ckpt, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()}
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    return state


class Recorder:
    def __init__(self):
        self.tensors = {}

    def put(self, name, tensor):
        self.tensors[name] = tensor.detach().float().cpu().numpy()


class AvatarAttnAdapter(torch.nn.Module):
    """The avatar model's Attention inside a base block: same weights, the reference-frame RoPE index and the skip window."""

    def __init__(self, base_attn, num_ref, ref_index, mask_range):
        super().__init__()
        from longcat_video.modules.avatar.attention import Attention as AvatarAttention

        self.attn = AvatarAttention(dim=base_attn.dim, num_heads=base_attn.num_heads, enable_flashattn2=True, cp_split_hw=[1, 1])
        self.attn.load_state_dict(base_attn.state_dict())
        self.num_ref, self.ref_index, self.mask_range = num_ref, ref_index, mask_range

    def forward(self, x, shape=None, num_cond_latents=None, return_kv=False):
        out, _ = self.attn(x, shape=shape, num_cond_latents=num_cond_latents, return_kv=False, num_ref_latents=self.num_ref,
                           ref_img_index=self.ref_index, mask_frame_range=self.mask_range, ref_target_masks=None)
        return out


def build_model(state, scenario):
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

    model = LongCatVideoTransformer3DModel(**CONFIG).float().eval()
    model.load_state_dict(state, strict=True)
    if scenario["num_ref"] > 0:
        for block in model.blocks:
            block.attn = AvatarAttnAdapter(block.attn, scenario["num_ref"], scenario["ref_index"], scenario["mask_range"]).eval()
    return model


def attach_hooks(model, rec, scenario):
    model.x_embedder.register_forward_hook(lambda m, a, o: rec.put("x_embedder", o[0]))
    model.t_embedder.register_forward_hook(lambda m, a, o: rec.put("t_embedder", o))
    model.y_embedder.register_forward_hook(lambda m, a, o: rec.put("y_embedder_raw", o[0, 0]))
    model.final_layer.register_forward_hook(lambda m, a, o: rec.put("final_layer", o[0]))
    for i, block in enumerate(model.blocks):
        p = f"block{i}"
        block.adaLN_modulation.register_forward_hook(lambda m, a, o, p=p: rec.put(f"{p}.modulation", o[0]))
        block.attn.register_forward_pre_hook(lambda m, a, p=p: rec.put(f"{p}.attn_in", a[0][0]))
        block.attn.register_forward_hook(lambda m, a, o, p=p: rec.put(f"{p}.self_attn", o[0]))
        block.pre_crs_attn_norm.register_forward_pre_hook(lambda m, a, p=p: rec.put(f"{p}.after_self", a[0][0]))
        block.cross_attn.register_forward_hook(lambda m, a, o, p=p: rec.put(f"{p}.cross_attn_full", o[0]))
        block.mod_norm_ffn.register_forward_pre_hook(lambda m, a, p=p: rec.put(f"{p}.after_cross", a[0].reshape(-1, a[0].shape[-1])))
        block.ffn.register_forward_hook(lambda m, a, o, p=p: rec.put(f"{p}.ffn", o[0]))
        block.register_forward_hook(lambda m, a, o, p=p: rec.put(f"{p}.out", o[0]))

    # Block 0's q and k after the norm and the RoPE, and v, as token-major [S, heads, headDim].
    holder = model.blocks[0].attn
    holder = holder.attn if hasattr(holder, "attn") else holder
    def record_rope(module, args, output):  # must return None: a hook's return value would replace the module output
        rec.put("block0.q", output[0][0].permute(1, 0, 2))
        rec.put("block0.k", output[1][0].permute(1, 0, 2))

    holder.rope_3d.register_forward_hook(record_rope)
    heads, head_dim = holder.num_heads, holder.head_dim
    holder.qkv.register_forward_hook(lambda m, a, o: rec.put("block0.v", o[0].view(o.shape[1], 3, heads, head_dim)[:, 2]))
    return holder


def rope_tables(holder, scenario):
    grid = (scenario["frames"], scenario["h"] // 2, scenario["w"] // 2)
    if scenario["num_ref"] > 0:
        freqs = holder.rope_3d.precompute_freqs_cis_3d(grid, scenario["ref_index"], scenario["num_ref"])
    else:
        freqs = holder.rope_3d.precompute_freqs_cis_3d(grid)
    freqs = freqs.float()
    return freqs.cos(), freqs.sin()


def write_f32(out_dir, meta, name, array):
    array = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    array.tofile(os.path.join(out_dir, name + ".f32"))
    meta["tensors"][name] = list(array.shape)


@torch.no_grad()
def run_scenario(state, scenario, out_dir, meta, seed):
    gen = torch.Generator().manual_seed(seed + sum(map(ord, scenario["name"])))
    latent = torch.randn((1, CONFIG["in_channels"], scenario["frames"], scenario["h"], scenario["w"]), generator=gen)
    encoder = torch.randn((1, 1, TEXT_LEN, CONFIG["caption_channels"]), generator=gen)
    mask = torch.zeros((1, TEXT_LEN))
    mask[:, :VALID_TEXT] = 1.0
    timesteps = torch.tensor([scenario["timesteps"]], dtype=torch.float32)

    model = build_model(state, scenario)
    rec = Recorder()
    holder = attach_hooks(model, rec, scenario)
    velocity = model(hidden_states=latent, timestep=timesteps, encoder_hidden_states=encoder, encoder_attention_mask=mask,
                     num_cond_latents=scenario["num_cond"])

    name = scenario["name"]
    cos, sin = rope_tables(holder, scenario)
    rec.put("t_freq", model.t_embedder.timestep_embedding(timesteps.flatten().float(), CONFIG["frequency_embedding_size"]))
    rec.put("rope_cos", cos)
    rec.put("rope_sin", sin)
    rec.put("y_embedder", torch.from_numpy(rec.tensors["y_embedder_raw"]) * mask[0, :, None])
    rec.put("velocity", velocity)
    write_f32(out_dir, meta, f"{name}.latent", latent)
    write_f32(out_dir, meta, f"{name}.encoder", encoder[0, 0])
    for stage, array in rec.tensors.items():
        write_f32(out_dir, meta, f"{name}.{stage}", array)
    meta["scenarios"][name] = dict(scenario, valid_text=VALID_TEXT, text_len=TEXT_LEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout of meituan-longcat/LongCat-Video at " + LONGCAT_COMMIT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    install_stubs(os.path.abspath(args.repo))

    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

    torch.manual_seed(args.seed)
    base = LongCatVideoTransformer3DModel(**CONFIG).float().eval()
    randomize(base, args.seed)
    state = save_checkpoint(base, out_dir)

    meta = dict(commit=LONGCAT_COMMIT, torch=torch.__version__, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in CONFIG.items()},
                tensors={}, scenarios={})
    for scenario in SCENARIOS:
        run_scenario(state, scenario, out_dir, meta, args.seed)
        print("dumped", scenario["name"], flush=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
