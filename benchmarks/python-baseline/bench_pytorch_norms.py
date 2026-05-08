"""PyTorch norm baselines (GroupNorm, LayerNorm, RmsNorm). Mirrors `NormGpuBenchmarks.cs`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import MICROBENCH_COLUMNS, CsvWriter, RunInfo, time_kernel

GROUPNORM_SHAPES: list[tuple[int, int, int, int, int]] = [
    (1, 320, 128, 128, 32),
    (1, 640, 64, 64, 32),
    (1, 1280, 32, 32, 32),
    (1, 128, 256, 256, 32),
    (1, 256, 512, 512, 32),
]

LAYERNORM_SHAPES: list[tuple[int, int, int]] = [
    (1, 4096, 1280),
    (1, 1024, 1536),
    (1, 1024, 3072),
    (1, 1024, 3840),
    (1, 1024, 2304),
]

RMSNORM_SHAPES: list[tuple[int, int, int]] = [
    (1, 1024, 1536),
    (1, 1024, 3072),
    (1, 1024, 3840),
    (1, 1024, 2304),
    (1, 1024, 4096),
]


def run_groupnorm(N, C, H, W, groups, dtype, *, n_warmup, n_trials):
    x = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
    w = torch.randn(C, device="cuda", dtype=dtype)
    b = torch.randn(C, device="cuda", dtype=dtype)
    fn = lambda: F.group_norm(x, groups, weight=w, bias=b, eps=1e-6)
    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def run_layernorm(B, S, H, dtype, *, n_warmup, n_trials):
    x = torch.randn(B, S, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype)
    b = torch.randn(H, device="cuda", dtype=dtype)
    fn = lambda: F.layer_norm(x, (H,), weight=w, bias=b, eps=1e-6)
    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def run_rmsnorm(B, S, H, dtype, *, n_warmup, n_trials):
    """RMSNorm via PyTorch 2.4+ functional. Earlier versions fall back to manual."""
    x = torch.randn(B, S, H, device="cuda", dtype=dtype)
    w = torch.randn(H, device="cuda", dtype=dtype)
    if hasattr(F, "rms_norm"):
        fn = lambda: F.rms_norm(x, (H,), weight=w, eps=1e-6)
    else:
        # Manual fallback — exposes whether torch.compile / fused-Linear helps.
        def fn():  # type: ignore[no-redef]
            v = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(v + 1e-6) * w

    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    info = RunInfo.current()
    writer = CsvWriter(Path(args.output), MICROBENCH_COLUMNS)

    for spec in GROUPNORM_SHAPES:
        N, C, H, W, groups = spec
        shape_str = f"{N}x{C}x{H}x{W}-g{groups}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            print(f"[groupnorm] shape={shape_str} dtype={dtype_str}")
            try:
                trials = run_groupnorm(N, C, H, W, groups, dtype, n_warmup=args.warmup, n_trials=args.trials)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped: {e}")
                continue
            for trial_i, lat in enumerate(trials):
                writer.write(info.base_row(backend="pytorch") | {
                    "op": "groupnorm",
                    "shape": shape_str,
                    "dtype": dtype_str,
                    "trial": trial_i,
                    "latency_us": f"{lat:.3f}",
                })

    for spec in LAYERNORM_SHAPES:
        B, S, H = spec
        shape_str = f"{B}x{S}x{H}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            print(f"[layernorm] shape={shape_str} dtype={dtype_str}")
            try:
                trials = run_layernorm(B, S, H, dtype, n_warmup=args.warmup, n_trials=args.trials)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped: {e}")
                continue
            for trial_i, lat in enumerate(trials):
                writer.write(info.base_row(backend="pytorch") | {
                    "op": "layernorm",
                    "shape": shape_str,
                    "dtype": dtype_str,
                    "trial": trial_i,
                    "latency_us": f"{lat:.3f}",
                })

    for spec in RMSNORM_SHAPES:
        B, S, H = spec
        shape_str = f"{B}x{S}x{H}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            print(f"[rmsnorm] shape={shape_str} dtype={dtype_str}")
            try:
                trials = run_rmsnorm(B, S, H, dtype, n_warmup=args.warmup, n_trials=args.trials)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped: {e}")
                continue
            for trial_i, lat in enumerate(trials):
                writer.write(info.base_row(backend="pytorch") | {
                    "op": "rmsnorm",
                    "shape": shape_str,
                    "dtype": dtype_str,
                    "trial": trial_i,
                    "latency_us": f"{lat:.3f}",
                })


if __name__ == "__main__":
    main()
