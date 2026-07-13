"""Matrix-Game 2.0 Wan-backbone parity oracle (Foundation config: no action module, full block-causal).
Runs the Skywork CausalWanModel._forward_train on the real distilled checkpoint and dumps per-stage taps +
the velocity for MatrixGame2ParityTests. flex_attention is monkeypatched to a dense-additive-mask SDPA
(the block mask for a single 3-frame block = full attention among real tokens, excluding the 64 pad tokens),
so no triton/flexattention is needed.

Env: MG2_SRC (/tmp/mg2_src/Matrix-Game-2), MG2_CKPT (base_distill.safetensors). Out: mg2_ref_io.safetensors.
"""
import os, sys, math, types, torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

# Stub heavy optional deps unused by the Foundation (no-action) path so the wan package imports cleanly.
import importlib.machinery
for _m in ["flash_attn", "flash_attn_interface", "xfuser", "xfuser.core", "xfuser.core.distributed"]:
    if _m not in sys.modules:
        mod = types.ModuleType(_m)
        mod.__spec__ = importlib.machinery.ModuleSpec(_m, None)
        mod.__version__ = "2.9.9"
        sys.modules[_m] = mod
sys.modules["flash_attn"].flash_attn_func = lambda *a, **k: None
sys.modules["flash_attn"].flash_attn_varlen_func = lambda *a, **k: None

SRC = os.environ.get("MG2_SRC", "/tmp/mg2_src/Matrix-Game-2")
CKPT = os.environ.get("MG2_CKPT", "/tmp/mg2_ckpt/base_distilled_model/base_distill.safetensors")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

# ── monkeypatch flex_attention → dense-mask SDPA, BEFORE importing causal_model ──
_DENSE_MASK = {"m": None}
def flex_dense(query, key, value, block_mask=None):
    # query/key/value: [B, heads, S_pad, headdim]. Additive mask [S_pad, S_pad] broadcasts over B, heads.
    return F.scaled_dot_product_attention(query, key, value, attn_mask=_DENSE_MASK["m"])

# Register wan / wan.modules as bare namespace packages (skip their heavy __init__.py that pulls t5/clip/vae).
for _pkg, _path in [("wan", f"{SRC}/wan"), ("wan.modules", f"{SRC}/wan/modules")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [_path]
    _m.__spec__ = importlib.machinery.ModuleSpec(_pkg, None, is_package=True)
    sys.modules[_pkg] = _m

import wan.modules.causal_model as CM
CM.flex_attention = flex_dense
# Cross-attention uses wan's flash_attention wrapper ([B,L,N,C]); replace with plain SDPA (no mask, full).
import wan.modules.model as WM
def flash_dense(q, k, v, q_lens=None, k_lens=None, softmax_scale=None, causal=False, **kw):
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [B, N, L, C]
    o = F.scaled_dot_product_attention(qh, kh, vh, scale=softmax_scale, is_causal=causal)
    return o.transpose(1, 2).contiguous()                                 # [B, L, N, C]
WM.flash_attention = flash_dense
from wan.modules.causal_model import CausalWanModel

@torch.no_grad()
def main():
    model = CausalWanModel(model_type='i2v', patch_size=(1, 2, 2), in_dim=36, dim=1536, ffn_dim=8960,
                           freq_dim=256, text_dim=4096, out_dim=16, num_heads=12, num_layers=30,
                           local_attn_size=-1, qk_norm=True, cross_attn_norm=True, action_config={}, eps=1e-6).eval()
    model.num_frame_per_block = 3

    sd = load_file(CKPT)
    sd = {(k[len('model.'):] if k.startswith('model.') else k): v.float() for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("load: missing", len(miss), miss[:6], "| unexpected", len(unexp), unexp[:6])
    model.float()

    torch.manual_seed(0)
    x36 = torch.randn(1, 36, 3, 16, 16)          # [B, 36, F, H, W]
    clip = torch.randn(1, 257, 1280)             # CLIP-ViT-H context
    t = torch.tensor([[500., 500., 500.]])       # [B, F] per-frame timesteps
    x_noisy = x36[:, :16]
    cond = x36[:, 16:]                           # cat(x_noisy, cond) == x36

    # Build the dense additive self-attn mask: total 192 real tokens, padded to 256 (ceil(192/128)*128).
    # ends[q]=192 for real q (<192) else 0; mask[q,k] = (k<ends[q]) | (q==k).  (single 3-frame block → full real↔real)
    total, pad = 192, math.ceil(192 / 128) * 128 - 192
    S = total + pad
    ends = torch.zeros(S, dtype=torch.long); ends[:total] = total
    q_idx = torch.arange(S).view(S, 1); k_idx = torch.arange(S).view(1, S)
    allow = (k_idx < ends[q_idx]) | (q_idx == k_idx)
    add = torch.zeros(S, S); add[~allow] = float('-inf')
    _DENSE_MASK["m"] = add

    taps = {}
    def hook(name, post=None):
        def f(mod, inp, out):
            taps[name] = (post(out) if post else out).detach().float()
        return f
    model.blocks[0].register_forward_hook(hook('tap_block0', lambda o: o[0]))
    model.blocks[-1].register_forward_hook(hook('tap_blockLast', lambda o: o[0]))
    model.img_emb.register_forward_hook(hook('tap_ctx', lambda o: o[0]))
    model.patch_embedding.register_forward_hook(hook('tap_patch', lambda o: o.flatten(2).transpose(1, 2)[0]))

    v = model._forward_train(x_noisy, t, clip, cond)   # [B, C_out, F, H, W]
    v_fchw = v[0]                                       # [C, F, H, W]  (natural Wan layout)
    v_tchw = v[0].permute(1, 0, 2, 3).contiguous()      # [F, C, H, W]  (C# [T,C,H,W] layout)

    refs = {"x36": x36, "clip": clip, "v": v_fchw, "v_tchw": v_tchw,   # C# unpatchify is [C,F,H,W] (natural Wan)
            "tap_patch": taps['tap_patch'], "tap_ctx": taps['tap_ctx'],
            "tap_block0": taps['tap_block0'], "tap_blockLast": taps['tap_blockLast']}
    refs = {k: val.float().contiguous() for k, val in refs.items()}
    save_file(refs, os.path.join(OUT, "mg2_ref_io.safetensors"))
    for k, val in refs.items():
        print(f"  {k:14s} {tuple(val.shape)}")
    print("wrote mg2_ref_io.safetensors ->", OUT)


if __name__ == "__main__":
    main()
