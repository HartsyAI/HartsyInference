"""
Dumps diffusers FluxControlNetModel references on tiny synthetic weights (CPU) for
HartsyInference FluxControlNet parity (FluxControlNetParityTests).

Two variants:
  - union:  num_layers=2, num_single_layers=2, num_mode=7 (InstantX Union shape, incl. mode embedder
            + controlnet_single_blocks) — exercises the mode-token prepend and the single-block residual path.
  - plain:  num_layers=2, num_single_layers=0, no mode embedder (Shakker Union-Pro-2.0 shape).

All parameters (including the zero-init controlnet blocks / controlnet_x_embedder) are re-randomized with a
fixed seed so the residuals are non-trivial. Weights + inputs + outputs land in one safetensors per variant.

Usage: python dump_flux_controlnet.py [output_dir]
Requires: torch (CPU), diffusers, safetensors.
"""
import sys
import types
import importlib.machinery

# hfvenv's torchaudio native lib is broken; transformers only probes for it — stub it out.
_ta = types.ModuleType("torchaudio")
_ta.__version__ = "0.0.0"
_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _ta

import torch
from safetensors.torch import save_file

from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

HEADS = 2
HEAD_DIM = 8
HIDDEN = HEADS * HEAD_DIM  # 16
JOINT_DIM = 32
POOLED_DIM = 24
IN_CHANNELS = 8
AXES = [2, 4, 2]  # must each be even (RoPE pairs) and sum to HEAD_DIM
HP = WP = 4
IMG_SEQ = HP * WP
TXT_SEQ = 6
SIGMA = 0.7
GUIDANCE = 3.5
COND_SCALE = 0.8


def build(num_layers, num_single_layers, num_mode, seed):
    torch.manual_seed(seed)
    model = FluxControlNetModel(
        patch_size=1,
        in_channels=IN_CHANNELS,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        joint_attention_dim=JOINT_DIM,
        pooled_projection_dim=POOLED_DIM,
        guidance_embeds=True,
        axes_dims_rope=AXES,
        num_mode=num_mode,
    ).eval()
    # Re-randomize EVERYTHING (zero_module leaves controlnet blocks all-zero -> trivially zero residuals).
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    return model


def dump(name, num_layers, num_single_layers, num_mode, seed, out_dir):
    model = build(num_layers, num_single_layers, num_mode, seed)
    gen = torch.Generator().manual_seed(seed + 2)

    hidden_states = torch.randn(1, IMG_SEQ, IN_CHANNELS, generator=gen)
    controlnet_cond = torch.randn(1, IMG_SEQ, IN_CHANNELS, generator=gen)
    encoder_hidden_states = torch.randn(1, TXT_SEQ, JOINT_DIM, generator=gen)
    pooled = torch.randn(1, POOLED_DIM, generator=gen)
    timestep = torch.tensor([SIGMA])
    guidance = torch.tensor([GUIDANCE])
    txt_ids = torch.zeros(TXT_SEQ, 3)
    img_ids = torch.zeros(IMG_SEQ, 3)
    for r in range(HP):
        for c in range(WP):
            img_ids[r * WP + c, 1] = r
            img_ids[r * WP + c, 2] = c
    mode = torch.tensor([[2]], dtype=torch.long) if num_mode is not None else None  # union: Depth

    with torch.no_grad():
        block_samples, single_block_samples = model(
            hidden_states=hidden_states,
            controlnet_cond=controlnet_cond,
            controlnet_mode=mode,
            conditioning_scale=COND_SCALE,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )

    tensors = {f"weights.{k}": v.contiguous() for k, v in model.state_dict().items()}
    tensors["io.hidden_states"] = hidden_states
    tensors["io.controlnet_cond"] = controlnet_cond
    tensors["io.encoder_hidden_states"] = encoder_hidden_states
    tensors["io.pooled_projections"] = pooled
    tensors["io.timestep"] = timestep
    tensors["io.guidance"] = guidance
    for i, s in enumerate(block_samples or []):
        tensors[f"io.block_sample.{i}"] = s.contiguous()
    for i, s in enumerate(single_block_samples or []):
        tensors[f"io.single_block_sample.{i}"] = s.contiguous()

    path = f"{out_dir}/flux_controlnet_{name}_ref.safetensors"
    save_file(tensors, path)
    n_single = len(single_block_samples) if single_block_samples is not None else 0
    print(f"{name}: wrote {path} ({len(block_samples)} double residuals, {n_single} single residuals)")
    for i, s in enumerate(block_samples):
        print(f"  block_sample.{i}: absmean={s.abs().mean():.6f} max={s.abs().max():.6f}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    dump("union", num_layers=2, num_single_layers=2, num_mode=7, seed=1234, out_dir=out_dir)
    dump("plain", num_layers=2, num_single_layers=0, num_mode=None, seed=5678, out_dir=out_dir)


if __name__ == "__main__":
    main()
