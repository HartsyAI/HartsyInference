"""
Matrix-Game 2.0 parity oracle (Stage A: Wan backbone, action module DISABLED).

Runs the self-contained Skywork reference (`wan/modules/model.py` WanModel) on the real Skywork base
checkpoint, with flash-attn / flex-attn stubbed to plain SDPA so it runs on CPU. Dumps the DiT forward +
per-stage taps for the C# parity test (action skipped: mouse/keyboard = None, C# passes keyboard=null).

  mg2_ref_io.safetensors:
    x36          [1,36,F,H,W]   concatenated latent (16 noisy + 20 cond) fed to both sides
    clip         [1,257,1280]   CLIP visual context
    t            [1]            single timestep
    v            [1,16,F,H,W]   DiT velocity output
    tap_patch    [1,S,1536]     post patch-embed tokens
    tap_ctx      [1,257,1536]   img_emb(visual_context)
    tap_block0   [1,S,1536]
    tap_blockLast[1,S,1536]

Usage: python3 tests/python-reference/mg2_reference/dump_mg2_reference.py
Env: MG2_DIT (Skywork base safetensors), MG2_SRC (/tmp/mg2ref/Matrix-Game-2)
"""
import os
import sys
import types
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

DIT = os.environ.get("MG2_DIT", "/tmp/mg2/base_model/diffusion_pytorch_model.safetensors")
SRC = os.environ.get("MG2_SRC", "/tmp/mg2ref/Matrix-Game-2")
OUT = os.path.dirname(os.path.abspath(__file__))
F_FRAMES, H, W = 3, 16, 16  # latent grid: patch (1,2,2) -> 3 x 8 x 8 = 192 tokens (one NumFramePerBlock)

# --- stub flash_attn so the reference modules import on CPU ---
def _sdpa(q, k, v, q_lens=None, k_lens=None, window_size=(-1, -1), causal=False, dropout_p=0.0, **kw):
    # q,k,v: [B, L, N, D] -> [B, N, L, D] -> sdpa -> back
    q2, k2, v2 = (t.transpose(1, 2).float() for t in (q, k, v))
    o = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    return o.transpose(1, 2).to(q.dtype)


import importlib.machinery  # noqa: E402
_stub_fn = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("action disabled"))
def _pkg(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    m.__path__ = []
    m.__version__ = "2.8.0"
    def _ga(n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _stub_fn
    m.__getattr__ = _ga  # probed non-dunder attr -> stub callable
    return m
fa = _pkg("flash_attn")
fa.flash_attn_func = _stub_fn
fa.flash_attn_varlen_func = _stub_fn
fai = _pkg("flash_attn.flash_attn_interface")
fai.flash_attn_func = _stub_fn
fa.flash_attn_interface = fai
sys.modules["flash_attn"] = fa
sys.modules["flash_attn.flash_attn_interface"] = fai
fi = _pkg("flash_attn_interface")
fi.flash_attn_func = _stub_fn
sys.modules["flash_attn_interface"] = fi

sys.path.insert(0, SRC)
# Bypass wan/__init__ + wan/modules/__init__ (they import t5/tokenizers/configs → heavy deps we don't need).
# Register minimal package stubs pointing at the real dirs so ONLY model.py + its siblings import.
for pkgname, subdir in [("wan", "wan"), ("wan.modules", "wan/modules")]:
    p = types.ModuleType(pkgname)
    p.__path__ = [os.path.join(SRC, subdir)]
    p.__spec__ = importlib.machinery.ModuleSpec(pkgname, None, is_package=True)
    sys.modules[pkgname] = p
import importlib  # noqa: E402
wanmodel = importlib.import_module("wan.modules.model")
wanattn = importlib.import_module("wan.modules.attention")

wanattn.FLASH_ATTN_2_AVAILABLE = False
wanattn.FLASH_ATTN_3_AVAILABLE = False
wanmodel.flash_attention = _sdpa  # self/cross attn -> SDPA (full attention)


def load_sd(fn):
    sd = load_file(fn)
    return {k.replace("module.", "").replace("model.", "") if k.startswith(("module.", "model.")) else k: v
            for k, v in sd.items()}


@torch.no_grad()
def main():
    torch.manual_seed(0)
    # Build WITH action modules (the base variant asserts use_action_module) then NULL them: this validates
    # the Wan backbone alone (Stage A). Action keys become "unexpected" on load; the C# passes keyboard=null.
    action_config = dict(mouse_dim_in=2, keyboard_dim_in=6, hidden_size=128, img_hidden_size=1536,
                         keyboard_hidden_dim=1024, mouse_hidden_dim=1024, heads_num=16, patch_size=[1, 2, 2],
                         qk_norm=True, qkv_bias=False, rope_dim_list=[8, 28, 28], rope_theta=256,
                         mouse_qk_dim_list=[8, 28, 28], windows_size=3, vae_time_compression_ratio=4,
                         enable_mouse=True, enable_keyboard=True, local_attn_size=6, blocks=list(range(30)))
    m = wanmodel.WanModel(model_type="i2v", patch_size=(1, 2, 2), in_dim=36, dim=1536, ffn_dim=8960,
                          freq_dim=256, out_dim=16, num_heads=12, num_layers=30,
                          qk_norm=True, cross_attn_norm=True, action_config=action_config).eval()
    for blk in m.blocks:
        blk.action_model = None
    miss, unexp = m.load_state_dict(load_sd(DIT), strict=False)
    print("missing (non-action):", [x for x in miss if "action" not in x][:8])
    print("unexpected (non-action):", [x for x in unexp if "action" not in x][:8])

    x16 = torch.randn(1, 16, F_FRAMES, H, W)
    cond = torch.randn(1, 20, F_FRAMES, H, W)
    x36 = torch.cat([x16, cond], dim=1)
    clip = torch.randn(1, 257, 1280)
    t = torch.tensor([500.0])

    taps = {}
    m.blocks[0].register_forward_hook(lambda mod, i, o: taps.__setitem__("tap_block0", o.detach().clone()))
    m.blocks[-1].register_forward_hook(lambda mod, i, o: taps.__setitem__("tap_blockLast", o.detach().clone()))
    m.patch_embedding.register_forward_hook(
        lambda mod, i, o: taps.__setitem__("tap_patch", o.detach().flatten(2).transpose(1, 2).clone()))
    m.img_emb.register_forward_hook(lambda mod, i, o: taps.__setitem__("tap_ctx", o.detach().clone()))

    v = m._forward(x16, t, clip, cond, mouse_cond=None, keyboard_cond=None)  # [1,16,F,H,W]

    refs = {"x36": x36, "clip": clip, "t": t, "v": v,
            "tap_patch": taps["tap_patch"], "tap_ctx": taps["tap_ctx"],
            "tap_block0": taps["tap_block0"], "tap_blockLast": taps["tap_blockLast"]}
    refs = {k: val.float().contiguous() for k, val in refs.items()}
    os.makedirs(OUT, exist_ok=True)
    save_file(refs, os.path.join(OUT, "mg2_ref_io.safetensors"))
    for k, val in refs.items():
        print(f"  {k:14s} {tuple(val.shape)}")
    print("wrote mg2_ref_io.safetensors ->", OUT)


if __name__ == "__main__":
    main()
