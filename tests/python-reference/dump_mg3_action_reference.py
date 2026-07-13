"""Matrix-Game 3.0 ActionModule parity oracle — runs the Skywork `ActionModule.forward`
(wan/modules/action_module.py) in ISOLATION on block-0's real action weights, and dumps the input hidden state +
mouse/keyboard streams + output residual for MatrixGame3ActionParityTests. Isolated (single module, no 30-block
accumulation) so it's a clean, fast check of the novel dual-attention surface (mouse temporal self-attn + keyboard
cross-attn, with the [8,28,28] θ=256 action RoPE). memory_length=0 (single segment).

flash_attn is absent → the module's own SDPA fallback runs; only `wan`/`wan.modules` are registered as namespace
packages. Runs on CPU float32. Env: MG3_SRC, MG3_CKPT. Out: mg3_action_ref_io.safetensors.
"""
import os, sys, types, importlib.machinery, torch
from safetensors.torch import save_file, load_file

SRC = os.environ.get("MG3_SRC", "/tmp/mg2_src/Matrix-Game-3")
CKPT = os.environ.get("MG3_CKPT", "/tmp/mg3_ckpt/base_model/diffusion_pytorch_model.safetensors")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

for _pkg, _path in [("wan", f"{SRC}/wan"), ("wan.modules", f"{SRC}/wan/modules")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [_path]
    _m.__spec__ = importlib.machinery.ModuleSpec(_pkg, None, is_package=True)
    sys.modules[_pkg] = _m

from wan.modules.action_module import ActionModule

ACTION_CONFIG = dict(
    mouse_dim_in=2, keyboard_dim_in=6, hidden_size=128, img_hidden_size=3072, keyboard_hidden_dim=1024,
    mouse_hidden_dim=1024, vae_time_compression_ratio=4, windows_size=3, heads_num=16, patch_size=[1, 2, 2],
    qk_norm=True, qkv_bias=False, rope_dim_list=[8, 28, 28], rope_theta=256, mouse_qk_dim_list=[8, 28, 28],
    enable_mouse=True, enable_keyboard=True,
)


@torch.no_grad()
def main():
    sd = load_file(CKPT)
    prefix = "blocks.0.action_model."
    msd = {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    del sd

    def build(**over):
        m = ActionModule(**{**ACTION_CONFIG, **over}).eval().float()
        miss, unexp = m.load_state_dict(msd, strict=False)
        return m

    module = build()
    miss, unexp = module.load_state_dict(msd, strict=False)
    print(f"load: missing {len(miss)} {miss[:8]} | unexpected {len(unexp)} {unexp[:8]}")

    torch.manual_seed(1)
    tt, th, tw, C = 3, 4, 4, 3072
    n_rgb = (tt - 1) * 4 + 1            # 9 RGB-rate action rows
    x = torch.randn(1, tt * th * tw, C)
    mouse = torch.randn(1, n_rgb, 2)
    keyboard = torch.randn(1, n_rgb, 6)

    out = module(x, tt, th, tw, mouse_condition=mouse, keyboard_condition=keyboard)   # [1, 48, 3072]
    out_m = build(enable_keyboard=False)(x, tt, th, tw, mouse_condition=mouse, keyboard_condition=keyboard)  # mouse only
    # (keyboard-only can't run in isolation: upstream references memory_length, set only inside the mouse block.)

    refs = {"x": x[0], "mouse": mouse[0], "keyboard": keyboard[0], "out": out[0], "out_m": out_m[0]}
    refs = {k: v.float().contiguous() for k, v in refs.items()}
    p = os.path.join(OUT, "mg3_action_ref_io.safetensors")
    save_file(refs, p)
    for k, v in refs.items():
        print(f"  {k:10s} {tuple(v.shape)}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
