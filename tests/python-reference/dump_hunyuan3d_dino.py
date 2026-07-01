"""Hunyuan3D-2 conditioner (DINOv2-giant) parity oracle. Extracts the giant weights from the DiT checkpoint
(conditioner.main_image_encoder.model.*) into a standalone F32 safetensors the C# loader reads, and dumps
last_hidden_state for a fixed normalized input via HF Dinov2Model. Env HY3D_DIT (default /tmp/hy3d/...)."""
import os, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Dinov2Model, Dinov2Config

DIT = os.environ.get("HY3D_DIT", "/tmp/hy3d/hunyuan3d-dit-v2-0/model.fp16.safetensors")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hunyuan3d_reference_tensors")
PRE = "conditioner.main_image_encoder.model."

# 1. Extract giant weights (F32), stripping the conditioner prefix → HF Dinov2Model key layout.
w = {}
with safe_open(DIT, "pt") as f:
    for k in f.keys():
        if k.startswith(PRE):
            w[k[len(PRE):]] = f.get_tensor(k).float().contiguous()
save_file(w, os.path.join(OUT, "dino_giant.safetensors"))
print("extracted giant weights:", len(w))

# 2. Build HF giant model (SwiGLU FFN, layerscale), load, eval.
cfg = Dinov2Config(hidden_size=1536, num_hidden_layers=40, num_attention_heads=24, mlp_ratio=4,
                   image_size=518, patch_size=14, layerscale_value=1.0, use_swiglu_ffn=True,
                   hidden_act="gelu", layer_norm_eps=1e-6, num_register_tokens=0)
model = Dinov2Model(cfg).eval()
missing, unexpected = model.load_state_dict(w, strict=False)
print("missing", len(missing), "unexpected", len(unexpected), missing[:4], unexpected[:4])

# 3. Fixed normalized input [1,3,518,518]; dump pixels + last_hidden_state.
torch.manual_seed(42)
pixels = torch.randn(1, 3, 518, 518)
with torch.no_grad():
    out = model(pixel_values=pixels).last_hidden_state   # [1,1370,1536]
save_file({"pixels": pixels.contiguous(), "last_hidden_state": out.float().contiguous()},
          os.path.join(OUT, "dino_ref_io.safetensors"))
print("last_hidden_state", tuple(out.shape), "std", out.std().item(), "mean", out.mean().item())
