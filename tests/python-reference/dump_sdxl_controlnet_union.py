"""
Dumps diffusers ControlNetUnionModel references from the REAL xinsir union SDXL checkpoint (CPU, float32)
for HartsyInference union ControlNet parity (SdxlControlNetUnionTests).

Two cases (all inputs seeded, saved alongside the outputs so both engines see identical tensors):
  - canny: control_type_idx 3 (thin line), batch 1  — the production call shape (cond-branch-only CN pass).
  - depth: control_type_idx 1, batch 2             — exercises the fusion transformer's real attention path
            (the reference feeds [B, tokens, C] into nn.MultiheadAttention with batch_first=False, so
            attention runs over the batch dim; at B=1 it degenerates to the V/out_proj path).

Outputs: io.down.{0..8} (zero-conv'd skip residuals) + io.mid, all after conditioning_scale.

Usage: python dump_sdxl_controlnet_union.py <checkpoint.safetensors> [output_dir]
Requires: torch (CPU), diffusers>=0.36 (ControlNetUnionModel), safetensors.
"""
import sys

import torch
from safetensors.torch import load_file, save_file

from diffusers import ControlNetUnionModel

TIMESTEP = 500.0
COND_SCALE = 0.8
LATENT = 64  # 512x512 pixel-space control image
NUM_CONTROL_TYPE = 8  # ProMax revision

SDXL_CONFIG = dict(
    in_channels=4,
    conditioning_channels=3,
    flip_sin_to_cos=True,
    freq_shift=0,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    block_out_channels=[320, 640, 1280],
    layers_per_block=2,
    cross_attention_dim=2048,
    transformer_layers_per_block=[1, 2, 10],
    num_attention_heads=[5, 10, 20],
    use_linear_projection=True,
    addition_embed_type="text_time",
    addition_time_embed_dim=256,
    projection_class_embeddings_input_dim=2816,
    conditioning_embedding_out_channels=[16, 32, 96, 256],
    num_control_type=NUM_CONTROL_TYPE,
)


def run_case(model, name, batch, control_idx, seed, out_dir):
    g = torch.Generator().manual_seed(seed)
    sample = torch.randn(batch, 4, LATENT, LATENT, generator=g)
    encoder_hidden_states = torch.randn(batch, 77, 2048, generator=g) * 0.5
    text_embeds = torch.randn(batch, 1280, generator=g) * 0.5
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]] * batch)
    cond_image = torch.rand(batch, 3, LATENT * 8, LATENT * 8, generator=g)
    control_type = torch.zeros(batch, NUM_CONTROL_TYPE)
    control_type[:, control_idx] = 1.0

    with torch.no_grad():
        down, mid = model(
            sample,
            TIMESTEP,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=[cond_image],
            control_type=control_type,
            control_type_idx=[control_idx],
            conditioning_scale=COND_SCALE,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )

    io = {
        "io.sample": sample,
        "io.timestep": torch.tensor([TIMESTEP]),
        "io.encoder_hidden_states": encoder_hidden_states,
        "io.text_embeds": text_embeds,
        "io.time_ids": time_ids,
        "io.controlnet_cond": cond_image,
        "io.control_type_idx": torch.tensor([float(control_idx)]),
        "io.conditioning_scale": torch.tensor([COND_SCALE]),
        "io.mid": mid.contiguous(),
    }
    for i, d in enumerate(down):
        io[f"io.down.{i}"] = d.contiguous()
    path = f"{out_dir}/sdxl_cn_union_ref_{name}.safetensors"
    save_file(io, path)
    stats = ", ".join(f"down.{i} std={d.std():.4f}" for i, d in enumerate(down[:3]))
    print(f"{name}: wrote {path} ({len(down)} down + mid; {stats}, mid std={mid.std():.4f})")


def main():
    ckpt = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    state = load_file(ckpt)
    state = {k: v.float() for k, v in state.items()}
    model = ControlNetUnionModel(**SDXL_CONFIG)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:10]}"
    assert not missing, f"missing keys: {missing[:10]}"
    model.eval()

    run_case(model, "canny", batch=1, control_idx=3, seed=1001, out_dir=out_dir)
    run_case(model, "depth_b2", batch=2, control_idx=1, seed=1002, out_dir=out_dir)


if __name__ == "__main__":
    main()
