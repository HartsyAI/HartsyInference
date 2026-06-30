"""
Hunyuan3D-2 DINOv2-giant conditioner parity oracle.

The conditioner (DINOv2-giant, SwiGLU FFN) is bundled in the shape DiT checkpoint under
`conditioner.main_image_encoder.model.*`. This extracts those weights, runs the upstream HF
`Dinov2Model` (giant config from the DiT config.yaml), and dumps:
  1. dino_giant.safetensors    the extracted conditioner state dict (F32; C# SafeTensorsLoader reads it)
  2. dino_giant_ref_io.safetensors:
       pixels        [1,3,518,518]   raw image in [0,1] (PRE imagenet-norm)
       image_tokens  [1,1370,1536]   last_hidden_state (the DiT cross-attn context)

Usage:
  python3 tests/python-reference/hunyuan3d_reference/dump_dino_giant_reference.py
Env:
  HUNYUAN3D_DIT   path to hunyuan3d-dit-v2-0/model.safetensors (default /tmp/hunyuan3d2/...)
"""
import os
import sys
import torch
from safetensors.torch import load_file, save_file
from transformers import Dinov2Model, Dinov2Config

DIT = os.environ.get("HUNYUAN3D_DIT", "/tmp/hunyuan3d2/hunyuan3d-dit-v2-0/model.safetensors")
OUT = os.path.dirname(os.path.abspath(__file__))
PREFIX = "conditioner.main_image_encoder.model."

# From hunyuan3d-dit-v2-0/config.yaml -> conditioner.params.main_image_encoder.kwargs.config
CFG = dict(
    hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, mlp_ratio=4,
    image_size=518, patch_size=14, num_channels=3, layer_norm_eps=1.0e-6,
    use_swiglu_ffn=True, layerscale_value=1.0, hidden_act="gelu",
    attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0, drop_path_rate=0.0,
    initializer_range=0.02,
)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def main():
    full = load_file(DIT)
    cond = {k[len(PREFIX):]: v.float().contiguous() for k, v in full.items() if k.startswith(PREFIX)}
    print(f"extracted {len(cond)} conditioner tensors")
    save_file(cond, os.path.join(OUT, "dino_giant.safetensors"))

    model = Dinov2Model(Dinov2Config(**CFG)).eval()
    missing, unexpected = model.load_state_dict(cond, strict=False)
    # mask_token is the only expected-but-unused param; report anything else.
    miss = [m for m in missing if "mask_token" not in m]
    print("missing (non-mask):", miss)
    print("unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    torch.manual_seed(0)
    pixels = torch.rand(1, 3, 518, 518)
    norm = (pixels - MEAN) / STD
    out = model(norm).last_hidden_state  # [1,1370,1536]
    print("image_tokens", tuple(out.shape))

    save_file({"pixels": pixels.contiguous(), "image_tokens": out.float().contiguous()},
              os.path.join(OUT, "dino_giant_ref_io.safetensors"))
    print("wrote dino_giant.safetensors + dino_giant_ref_io.safetensors ->", OUT)


if __name__ == "__main__":
    main()
