"""SeedVR2 NaDiT tiny-config parity dump (Parts A5/A6 gate).

Builds ByteDance's real NaDiT (models/dit_v2) at a tiny config with seeded random weights, runs one
forward on CPU/f32, and dumps per-block vid/txt states + final v-prediction as raw f32 .bin files plus
weights.safetensors (verbatim state_dict names — the C# side loads them directly; rope freq buffers are
dropped, C# recomputes them). flash_attn is absent on this machine, so a math-identical SDPA shim is
injected BEFORE import (varlen == per-sequence SDPA at scale 1/sqrt(d) — no numerics caveat beyond
kernel-order float noise).

Usage:  <seedvr2-venv-python> seedvr2_transformer_parity_dump.py <SeedVR-checkout> $SEEDVR2_PARITY_DIR
Then:   SEEDVR2_PARITY_DIR=... dotnet test --filter SeedVr2DitParityTests

Tiny dims must match SeedVr2DitParityTests.TinyConfig exactly.
"""
import os
import sys
import types

import torch

# ---- flash_attn shim (must precede models.dit_v2 import) ----
def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                 dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        s, e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        qi = q[s:e].transpose(0, 1).unsqueeze(0).float()  # (1,h,s,d)
        ki = k[s:e].transpose(0, 1).unsqueeze(0).float()
        vi = v[s:e].transpose(0, 1).unsqueeze(0).float()
        oi = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)
        outs.append(oi.squeeze(0).transpose(0, 1).to(q.dtype))
    return torch.cat(outs)

import importlib.machinery  # noqa: E402

_fa = types.ModuleType("flash_attn")
_fa.flash_attn_varlen_func = _sdpa_varlen
_fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
sys.modules["flash_attn"] = _fa

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from models.dit_v2 import na  # noqa: E402
from models.dit_v2.nadit import NaDiT  # noqa: E402

OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# Must match SeedVr2DitParityTests.TinyConfig exactly.
VID_DIM, TXT_IN_DIM, HEADS, HEAD_DIM = 128, 32, 1, 128
LAYERS, MM_LAYERS = 4, 2
T, H, W = 5, 90, 160          # pre-patchify latent grid (patch 1x2x2 -> tokens 5x45x80)
TXT_LEN = 7
TIMESTEP = 937.0              # non-round to exercise the sinusoid

torch.manual_seed(1234)
model = NaDiT(
    vid_in_channels=33, vid_out_channels=16, vid_dim=VID_DIM,
    txt_in_dim=TXT_IN_DIM, txt_dim=VID_DIM, emb_dim=6 * VID_DIM,
    heads=HEADS, head_dim=HEAD_DIM, expand_ratio=4,
    norm="rms", norm_eps=1e-5, ada="single", qk_bias=False, qk_norm="rms",
    patch_size=(1, 2, 2), num_layers=LAYERS,
    block_type="mmdit_sr", mm_layers=MM_LAYERS, mlp_type="swiglu",
    window=[(4, 3, 3)] * LAYERS,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * (LAYERS // 2),
    rope_type="mmrope3d", rope_dim=HEAD_DIM,
    vid_out_norm="rms",
).float().eval()

with torch.no_grad():
    for _, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * 0.1)

sd = {k: v.clone() for k, v in model.state_dict().items() if not k.endswith("rope.rope.freqs")}
save_file(sd, os.path.join(OUT, "weights.safetensors"))


def dump(name, tensor):
    np.ascontiguousarray(tensor.detach().float().cpu().numpy(), dtype=np.float32).tofile(
        os.path.join(OUT, name + ".bin"))


gen = torch.Generator().manual_seed(42)
latent = torch.randn((T, H, W, 33), generator=gen).float()
txt = torch.randn((TXT_LEN, TXT_IN_DIM), generator=gen).float()
dump("input_latent", latent)
dump("input_txt", txt)

vid_flat, vid_shape = na.flatten([latent])
txt_flat, txt_shape = na.flatten([txt])

captures = {}
for i, block in enumerate(model.blocks):
    def hook(mod, args, output, idx=i):
        captures[f"block{idx}_vid"] = output[0].detach().clone()
        captures[f"block{idx}_txt"] = output[1].detach().clone()
    block.register_forward_hook(hook)

with torch.no_grad():
    out = model(vid=vid_flat, txt=txt_flat, vid_shape=vid_shape, txt_shape=txt_shape,
                timestep=torch.tensor([TIMESTEP]))

for k, v in captures.items():
    dump(k, v)
dump("output", out.vid_sample)
print("dumped", len(captures) + 3, "tensors ->", OUT)
print("output shape", tuple(out.vid_sample.shape))
