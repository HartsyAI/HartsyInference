# Regenerates the sana_wm_s4_glumb_temp_ffn_*.safetensors fixtures from the upstream SANA-WM GLUMBConvTemp (CPU, float32).
# Run: CUDA_VISIBLE_DEVICES= <venv>/bin/python generate_sana_wm_s4_fixtures.py <Fixtures dir> <Sana src dir>
import math
import importlib
import os
import sys
import types

import torch
from safetensors.torch import save_file

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be set empty"
OUT_DIR, SANA_SRC = sys.argv[1], sys.argv[2]
CHANNELS, MLP_RATIO, BATCH, HEIGHT, WIDTH = 8, 3, 2, 4, 5
HIDDEN = CHANNELS * MLP_RATIO
CASES = {"t3": 3, "t1": 1}


def load_basic_modules(src):
    """Imports upstream basic_modules.py without running diffusion/__init__.py or nets/__init__.py (which pull in
    triton-backed GDN blocks). Only the FFN's own dependencies are loaded."""
    packages = {
        "diffusion": "diffusion",
        "diffusion.model": "diffusion/model",
        "diffusion.model.nets": "diffusion/model/nets",
        "diffusion.distributed": "diffusion/distributed",
        "diffusion.distributed.context_parallel": "diffusion/distributed/context_parallel",
    }
    for name, rel in packages.items():
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(src, rel)]
        sys.modules[name] = pkg
    sys.path.insert(0, src)
    return importlib.import_module("diffusion.model.nets.basic_modules")


def build_ffn(basic_modules, frames):
    torch.manual_seed(20261109 + frames)
    mlp = basic_modules.GLUMBConvTemp(
        in_features=CHANNELS,
        hidden_features=HIDDEN,
        use_bias=(True, True, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
        t_kernel_size=3,
    ).float().eval()
    with torch.no_grad():
        # The upstream constructor zero-initialises t_conv; a zero temporal weight would hide the temporal path.
        mlp.t_conv.weight.copy_(torch.randn_like(mlp.t_conv.weight) / math.sqrt(3 * CHANNELS))
    return mlp


def main():
    basic_modules = load_basic_modules(SANA_SRC)
    for tag, frames in CASES.items():
        mlp = build_ffn(basic_modules, frames)
        x = torch.randn(BATCH, frames * HEIGHT * WIDTH, CHANNELS)
        with torch.no_grad():
            expected = mlp(x, HW=(frames, HEIGHT, WIDTH))
            # Same module with t_conv zeroed: the difference is the temporal residual alone.
            saved = mlp.t_conv.weight.clone()
            mlp.t_conv.weight.zero_()
            without_temporal = mlp(x, HW=(frames, HEIGHT, WIDTH))
            mlp.t_conv.weight.copy_(saved)
        tensors = {
            "input": x.contiguous(),
            "expected_output": expected.contiguous(),
            "mlp.inverted_conv.conv.weight": mlp.inverted_conv.conv.weight.detach().contiguous(),
            "mlp.inverted_conv.conv.bias": mlp.inverted_conv.conv.bias.detach().contiguous(),
            "mlp.depth_conv.conv.weight": mlp.depth_conv.conv.weight.detach().contiguous(),
            "mlp.depth_conv.conv.bias": mlp.depth_conv.conv.bias.detach().contiguous(),
            "mlp.point_conv.conv.weight": mlp.point_conv.conv.weight.detach().contiguous(),
            "mlp.t_conv.weight": mlp.t_conv.weight.detach().contiguous(),
        }
        metadata = {
            "B": str(BATCH), "T": str(frames), "H": str(HEIGHT), "W": str(WIDTH),
            "channels": str(CHANNELS), "hidden": str(HIDDEN),
            "source": "Sana GLUMBConvTemp (diffusion/model/nets/basic_modules.py), CPU float32",
            "torch": torch.__version__,
        }
        path = os.path.join(OUT_DIR, f"sana_wm_s4_glumb_temp_ffn_{tag}.safetensors")
        save_file(tensors, path, metadata=metadata)
        delta = (expected - without_temporal).abs().max().item()
        print(f"{path} frames={frames} output={tuple(expected.shape)} temporal_residual_max_abs={delta:.6f}")


if __name__ == "__main__":
    main()
