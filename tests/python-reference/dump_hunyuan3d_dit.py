"""Hunyuan3D-2 DiT (Flux, no-RoPE) parity oracle. Instantiates the real hy3dgen Hunyuan3DDiT, loads the
`model.*` weights from the checkpoint, and dumps velocity for a fixed (latent, cond, t). Also extracts the
DiT weights (F32, `model.` stripped) for the C# loader. Env HY3D_DIT, HY3D_REPO."""
import os, sys, numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

DIT = os.environ.get("HY3D_DIT", "/tmp/hy3d/hunyuan3d-dit-v2-0/model.fp16.safetensors")
REPO = os.environ.get("HY3D_REPO", "/tmp/Hunyuan3D-2")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hunyuan3d_reference_tensors")
import importlib.util as _u
_spec=_u.spec_from_file_location("hy3ddit", os.path.join(REPO,"hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py"))
_m=_u.module_from_spec(_spec); _spec.loader.exec_module(_m); Hunyuan3DDiT=_m.Hunyuan3DDiT

# 1. Extract + save DiT weights (F32, strip 'model.').
w = {}
with safe_open(DIT, "pt") as f:
    for k in f.keys():
        if k.startswith("model."):
            w[k[len("model."):]] = f.get_tensor(k).float().contiguous()
save_file(w, os.path.join(OUT, "dit_weights.safetensors"))
print("extracted DiT weights:", len(w))

# 2. Build the real model, load, eval (fp32 on CPU).
model = Hunyuan3DDiT(in_channels=64, context_in_dim=1536, hidden_size=1024, mlp_ratio=4.0, num_heads=16,
                     depth=16, depth_single_blocks=32, axes_dim=[64], theta=10000, qkv_bias=True,
                     time_factor=1000, guidance_embed=False).eval().float()
missing, unexpected = model.load_state_dict(w, strict=False)
print("missing", len(missing), "unexpected", len(unexpected), missing[:3], unexpected[:3])

# 3. Fixed input: latent [1,3072,64], cond [1,1370,1536], t=[0.5]; dump velocity + per-block (double_0, single_0).
torch.manual_seed(123)
latent = torch.randn(1, 3072, 64)
cond = torch.randn(1, 1370, 1536)
t = torch.tensor([0.5])
with torch.no_grad():
    vel = model(latent, t, {"main": cond})
save_file({"latent": latent.contiguous(), "cond": cond.contiguous(), "timestep": t.contiguous(),
           "velocity": vel.float().contiguous()}, os.path.join(OUT, "dit_ref_io.safetensors"))
print("velocity", tuple(vel.shape), "std", vel.std().item(), "mean", vel.mean().item())
