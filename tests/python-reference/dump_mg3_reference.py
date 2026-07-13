"""Matrix-Game 3.0 parity oracle — runs the Skywork `WanModel.forward` (wan/modules/model.py) on the real
base_model checkpoint and dumps the velocity + per-stage taps for MatrixGame3ParityTests.

Staged (env MG3_STAGE):
  A  core Wan2.2 backbone: use_memory=True, action_config={} (no ActionModule), plucker_emb=None, memory_length=0.
     Isolates patchify / per-frame timestep modulation / memory-mode RoPE (rope_apply_with_indices, sigma_theta=0.8)
     / the destructive cross-attn norm3 residual / FFN / head / unpatchify.
  B  + ActionModule (mouse self-attn + keyboard cross-attn) on the config's action blocks, with live action streams.

flash_attention (model.py) is monkeypatched to a dense SDPA (no flash_attn/triton needed); wan / wan.modules /
wan.distributed are registered as bare namespace packages so their heavy __init__ (t5/clip/vae) is skipped. Runs on
CPU float32 for determinism.

Env: MG3_SRC (/tmp/mg2_src/Matrix-Game-3), MG3_CKPT (base_model safetensors). Out: mg3_ref_io_<stage>.safetensors.
"""
import os, sys, types, importlib.machinery, torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

SRC = os.environ.get("MG3_SRC", "/tmp/mg2_src/Matrix-Game-3")
CKPT = os.environ.get("MG3_CKPT", "/tmp/mg3_ckpt/base_model/diffusion_pytorch_model.safetensors")
STAGE = os.environ.get("MG3_STAGE", "A").upper()
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

# Register wan / wan.modules / wan.distributed as bare namespace packages (skip the heavy __init__.py).
for _pkg, _path in [("wan", f"{SRC}/wan"), ("wan.modules", f"{SRC}/wan/modules"),
                    ("wan.distributed", f"{SRC}/wan/distributed")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [_path]
    _m.__spec__ = importlib.machinery.ModuleSpec(_pkg, None, is_package=True)
    sys.modules[_pkg] = _m

import wan.modules.model as WM

# flash_attention(q,k,v,...) with q/k/v [B, L, N, C] -> dense SDPA (no k_lens masking: our inputs have no padding).
def flash_dense(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [B, N, L, C]
    o = F.scaled_dot_product_attention(qh, kh, vh, scale=softmax_scale, is_causal=causal)
    return o.transpose(1, 2).contiguous()                                 # [B, L, N, C]
WM.flash_attention = flash_dense
from wan.modules.model import WanModel, WanCrossAttention


# WanCrossAttention.forward hardcodes x/context .to(bfloat16) (production runs bf16); keep it F32 for tight parity.
def _cross_forward_f32(self, x, context, context_lens, fa_version=None):
    b, n, d = x.size(0), self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)
    x = WM.flash_attention(q, k, v, k_lens=context_lens, version=fa_version)
    return self.o(x.flatten(2))
WanCrossAttention.forward = _cross_forward_f32

# Config: base_model/config.json (dim 3072 / 24 heads / 30 layers / ffn 14336, sigma_theta 0.8).
ACTION_CONFIG = dict(
    blocks=list(range(15)), enable_keyboard=True, enable_mouse=True, heads_num=16, hidden_size=128,
    img_hidden_size=3072, keyboard_dim_in=6, keyboard_hidden_dim=1024, mouse_dim_in=2, mouse_hidden_dim=1024,
    mouse_qk_dim_list=[8, 28, 28], patch_size=[1, 2, 2], qk_norm=True, qkv_bias=False,
    rope_dim_list=[8, 28, 28], rope_theta=256, vae_time_compression_ratio=4, windows_size=3,
)


@torch.no_grad()
def main():
    action_config = {} if STAGE == "A" else ACTION_CONFIG
    model = WanModel(model_type='ti2v', patch_size=(1, 2, 2), text_len=512, in_dim=48, dim=3072, ffn_dim=14336,
                     freq_dim=256, text_dim=4096, out_dim=48, num_heads=24, num_layers=30, window_size=(-1, -1),
                     qk_norm=True, cross_attn_norm=True, eps=1e-6, action_config=action_config, use_memory=True,
                     sigma_theta=0.8, use_text_crossattn=True).eval().float()

    sd = load_file(CKPT)   # bf16; load_state_dict copy_ casts to the model's f32 params (avoids a 26 GB f32 dict)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"load: missing {len(miss)} {miss[:6]} | unexpected {len(unexp)} {unexp[:6]}")
    del sd

    torch.manual_seed(0)
    F_lat, H, W = 3, 8, 8
    gt, gh, gw = F_lat, H // 2, W // 2            # 3, 4, 4
    s = gt * gh * gw                              # 48
    x = [torch.randn(48, F_lat, H, W)]
    context = [torch.randn(20, 4096)]            # umT5 rows (padded to text_len=512 inside forward)
    frame_t = torch.tensor([1000., 700., 400.])  # per-latent-frame timesteps
    t = frame_t.repeat_interleave(gh * gw).unsqueeze(0)   # [1, s] per-token (frame-major token order)

    mouse = keyboard = None
    extra = {}
    if STAGE != "A":
        # RGB-rate action rows covering the F_lat latent frames: N_rgb = (F_lat-1)*ratio + 1 = 9.
        n_rgb = (F_lat - 1) * 4 + 1
        mouse = torch.randn(1, n_rgb, 2)
        keyboard = torch.randn(1, n_rgb, 6)
        extra = dict(mouse_cond=mouse, keyboard_cond=keyboard)

    taps = {}

    def hook(name, post):
        def f(mod, inp, out):
            taps[name] = post(out).detach().float().contiguous()
        return f
    model.patch_embedding.register_forward_hook(hook('tap_patch', lambda o: o.flatten(2).transpose(1, 2)[0]))
    model.text_embedding.register_forward_hook(hook('tap_ctx', lambda o: o[0]))
    for _i, _blk in enumerate(model.blocks):
        _blk.register_forward_hook(hook(f'tap_b{_i}', lambda o: o[0]))

    # Block-0 sub-taps for intra-block layer-diff (raw F32 bins alongside the C# WAN_DEBUG_DIR dumps).
    dbgdir = os.environ.get("WAN_DEBUG_DIR")
    if dbgdir:
        import numpy as np
        ld = os.path.join(dbgdir, "layers"); os.makedirs(ld, exist_ok=True)
        def save_bin(nm, tt): tt.detach().float().cpu().numpy().tofile(os.path.join(ld, nm + ".bin"))
        b0 = model.blocks[0]
        def _mod_hook(m, args, kwargs):
            e0 = kwargs['e'] if 'e' in kwargs else args[1]          # [1, seq, 6, dim]
            e = (m.modulation.unsqueeze(0) + e0).chunk(6, dim=2)    # e[0..5], each [1, seq, 1, dim]
            save_bin("b0ref_shiftMsa", e[0].squeeze(2)[0]); save_bin("b0ref_scaleMsa", e[1].squeeze(2)[0])
            save_bin("b0ref_gateMsa", e[2].squeeze(2)[0])
        b0.register_forward_pre_hook(_mod_hook, with_kwargs=True)
        b0.self_attn.register_forward_pre_hook(lambda m, a: save_bin("b0ref_n1", a[0][0]))
        b0.self_attn.register_forward_hook(lambda m, i, o: save_bin("b0ref_attn1", o[0]))
        b0.cross_attn.register_forward_hook(lambda m, i, o: save_bin("b0ref_attn2", o[0]))

    v = model(x, t, context, s, plucker_emb=None, **extra)   # [1, 48, F, H, W] = [B, C, F, H, W]
    v_cfhw = v[0]                                             # [C, F, H, W]

    refs = {
        "x": x[0], "context": context[0], "frame_t": frame_t,
        "v_cfhw": v_cfhw,
        "tap_patch": taps['tap_patch'], "tap_ctx": taps['tap_ctx'],
        **{k: v for k, v in taps.items() if k.startswith('tap_b')},
    }
    if mouse is not None:
        refs["mouse"] = mouse[0]
        refs["keyboard"] = keyboard[0]
    refs = {k: val.float().contiguous() for k, val in refs.items()}
    out = os.path.join(OUT, f"mg3_ref_io_{STAGE}.safetensors")
    save_file(refs, out)
    for k, val in refs.items():
        print(f"  {k:14s} {tuple(val.shape)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
