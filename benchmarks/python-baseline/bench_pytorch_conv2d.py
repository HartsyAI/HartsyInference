"""PyTorch Conv2D baseline. Mirrors `Conv2DGpuBenchmarks.cs`. Both `cudnn.benchmark=True` (best
case, autotunes algorithm) and `False` (more directly comparable to a fixed-algorithm im2col path)
variants are captured so the paper can present both."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import MICROBENCH_COLUMNS, CsvWriter, RunInfo, time_kernel

# (N, Cin, Cout, H, W, K, stride, pad)
SHAPES: list[tuple[int, int, int, int, int, int, int, int]] = [
    (1, 320, 320, 128, 128, 3, 1, 1),
    (1, 640, 640, 64, 64, 3, 1, 1),
    (1, 1280, 1280, 32, 32, 3, 1, 1),
    (1, 320, 320, 128, 128, 3, 2, 1),
    (1, 320, 640, 128, 128, 1, 1, 0),
    (1, 128, 128, 256, 256, 3, 1, 1),
    (1, 256, 256, 512, 512, 3, 1, 1),
    (1, 128, 3, 1024, 1024, 3, 1, 1),
]


def run_one(spec, dtype, cudnn_benchmark, *, n_warmup, n_trials):
    N, Cin, Cout, H, W, K, stride, pad = spec
    torch.backends.cudnn.benchmark = cudnn_benchmark
    x = torch.randn(N, Cin, H, W, device="cuda", dtype=dtype)
    w = torch.randn(Cout, Cin, K, K, device="cuda", dtype=dtype)
    b = torch.randn(Cout, device="cuda", dtype=dtype)

    def fn() -> None:
        F.conv2d(x, w, b, stride=stride, padding=pad)

    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    info = RunInfo.current()
    writer = CsvWriter(Path(args.output), MICROBENCH_COLUMNS)

    for spec in SHAPES:
        N, Cin, Cout, H, W, K, stride, pad = spec
        shape_str = f"{N}x{Cin}x{H}x{W}-{Cout}x{K}x{K}-s{stride}p{pad}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            for cudnn_benchmark in (False, True):
                tag = "f-cudnn" if not cudnn_benchmark else "f-cudnn_benchmark"
                backend = f"pytorch-{tag}"
                print(f"[conv2d] shape={shape_str} dtype={dtype_str} cudnn_benchmark={cudnn_benchmark}")
                try:
                    trials = run_one(spec, dtype, cudnn_benchmark, n_warmup=args.warmup, n_trials=args.trials)
                except Exception as e:  # noqa: BLE001
                    print(f"  skipped: {e}")
                    continue
                for trial_i, lat in enumerate(trials):
                    writer.write(info.base_row(backend=backend) | {
                        "op": "conv2d",
                        "shape": shape_str,
                        "dtype": dtype_str,
                        "trial": trial_i,
                        "latency_us": f"{lat:.3f}",
                        "throughput_gflops": "",
                        "memory_mb": "",
                        "workspace_mb": "",
                    })


if __name__ == "__main__":
    main()
