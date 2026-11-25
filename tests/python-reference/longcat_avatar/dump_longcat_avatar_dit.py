"""Reference dump for the LongCat-Video-Avatar DiT (LongCatVideoAvatarTransformer3DModel, modules/avatar/longcat_video_dit_avatar.py @ 58eb336).

Builds the pinned model on a tiny config with seeded weights and runs three calls on the reference's own forward: an at2v call (no
conditioning frames), an ai2v call (one conditioning latent, no reference latent) and an avc call (two conditioning latents, one reference
latent at ref_img_index). Writes the weights, the inputs, the audio projection output and the three velocities as raw F32 files plus
meta.json (the layout LongCatReferenceDump.Load reads).

Run in the hartsy-ref:cpu image only (never the Windows host). Assemble the pinned source (longcat_video/) in a scratch folder, then:

    python dump_longcat_avatar_dit.py --repo /src --out /out/longcat_avatar_dit_reference_tensors

Stubbed, so NOT used by the path under test:
- diffusers: ConfigMixin, ModelMixin (nn.Module) and register_to_config (identity). The model needs nothing else from it.
- safetensors.torch.load_file: only used by load_lora, which this dump does not call.
- flash_attn.flash_attn_func and flash_attn_varlen_func: replaced by torch SDPA with the same layouts and softmax scale (as in the B1,
  B4 and audio-attention dumps). The varlen stub handles one segment per cumulative-length pair, non-causal.
- block_sparse_attention.bsa_interface and audio_process.torch_utils: multi-speaker and block-sparse paths only; not enabled here.
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


def install_stubs():
    diffusers = types.ModuleType("diffusers")

    class ConfigMixin:
        pass

    class ModelMixin(torch.nn.Module):
        pass

    def register_to_config(fn):
        return fn

    models = types.ModuleType("diffusers.models")
    modeling = types.ModuleType("diffusers.models.modeling_utils")
    modeling.ModelMixin = ModelMixin
    configuration = types.ModuleType("diffusers.configuration_utils")
    configuration.ConfigMixin = ConfigMixin
    configuration.register_to_config = register_to_config
    diffusers.ConfigMixin = ConfigMixin
    diffusers.ModelMixin = ModelMixin
    diffusers.models = models
    for name, module in (("diffusers", diffusers), ("diffusers.models", models), ("diffusers.models.modeling_utils", modeling),
                         ("diffusers.configuration_utils", configuration)):
        sys.modules[name] = module

    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = None
    safetensors.torch = safetensors_torch
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors_torch

    flash = types.ModuleType("flash_attn")

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, **_):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
        return out.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, softmax_scale=None,
                               causal=False, **_):
        # Packed [total, H, D] inputs, split at the cumulative lengths; one SDPA per segment.
        assert not causal, "the reference never asks for causal attention here"
        outs = []
        for i in range(len(cu_seqlens_q) - 1):
            qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]].transpose(0, 1).unsqueeze(0)
            ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1).unsqueeze(0)
            vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1).unsqueeze(0)
            outs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)[0].transpose(0, 1))
        return torch.cat(outs, dim=0)

    flash.flash_attn_func = flash_attn_func
    flash.flash_attn_varlen_func = flash_attn_varlen_func
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
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    install_stubs()
    dit_module = importlib.import_module("longcat_video.modules.avatar.longcat_video_dit_avatar")

    config = dict(
        in_channels=4, out_channels=4, hidden_size=64, depth=2, num_heads=4, caption_channels=32, mlp_ratio=4,
        adaln_tembed_dim=32, frequency_embedding_size=32, patch_size=(1, 2, 2), enable_flashattn2=True,
        text_tokens_zero_pad=True, audio_window=5, audio_block=3, audio_channel=6, intermediate_dim=16, output_dim=16,
        context_tokens=4, vae_scale=4, class_range=24, class_interval=4, cp_split_hw=[1, 1],
    )
    torch.manual_seed(args.seed)
    model = dit_module.LongCatVideoAvatarTransformer3DModel(**config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)

    frames, height, width, text_len = 4, 4, 4, 6
    latent = torch.randn(1, config["in_channels"], frames, height, width)
    encoder = torch.randn(1, 1, text_len, config["caption_channels"])
    audio_len = 1 + config["vae_scale"] * (frames - 1)
    audio = torch.randn(1, audio_len, config["audio_window"], config["audio_block"], config["audio_channel"])
    timestep = torch.tensor([500.0])

    os.makedirs(args.out, exist_ok=True)
    meta = {"config": config, "frames": frames, "h": height, "w": width, "text_len": text_len, "tensors": {}}
    for key, value in model.state_dict().items():
        write_tensor(args.out, meta, "dit." + key, value)
    write_tensor(args.out, meta, "inputs.latent", latent.reshape(-1))
    write_tensor(args.out, meta, "inputs.encoder", encoder.reshape(text_len, -1))
    write_tensor(args.out, meta, "inputs.audio", audio.reshape(audio_len, -1))

    captured = {}
    hook = model.audio_proj.register_forward_hook(lambda _m, _i, out: captured.__setitem__("audio_proj", out))
    cases = {
        "at2v": dict(num_cond_latents=0),
        "ai2v": dict(num_cond_latents=1),
        "avc": dict(num_cond_latents=2, num_ref_latents=1, ref_img_index=10),
    }
    with torch.no_grad():
        for name, kwargs in cases.items():
            velocity = model(latent, timestep, encoder, audio_embs=audio, **kwargs)
            write_tensor(args.out, meta, f"velocity.{name}", velocity.reshape(-1))
            if name == "at2v":
                write_tensor(args.out, meta, "audio_proj", captured["audio_proj"].reshape(-1, config["output_dim"]))
    hook.remove()

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps({"tensors": len(meta["tensors"]), "velocity_shape": list(velocity.shape)}))


if __name__ == "__main__":
    main()
