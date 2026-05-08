"""PyTorch GEMM baseline. Mirrors the shape grid in `MatMulGpuBenchmarks.cs`.

Usage:
    python3 bench_pytorch_matmul.py --output results/run_X/microbench.csv --trials 5

The script appends to the CSV (does not overwrite), so multiple ops can target the same file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import (
    MICROBENCH_COLUMNS,
    CsvWriter,
    RunInfo,
    gflops_for_matmul,
    time_kernel,
)

# (M, K, N) — same grid as MatMulGpuBenchmarks.cs (one entry per row, comments match)
SHAPES: list[tuple[int, int, int]] = [
    (4096, 1280, 1280),
    (4096, 1280, 1280),
    (4096, 1280, 10240),
    (4096, 5120, 1280),
    (1024, 3072, 9216),
    (1024, 3072, 12288),
    (1024, 1536, 4608),
    (1024, 3840, 11520),
    (1024, 2304, 6912),
    (1024, 3072, 9216),
]


def run_one(M: int, K: int, N: int, dtype: torch.dtype, *, n_warmup: int, n_trials: int) -> list[float]:
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    def fn() -> None:
        torch.matmul(a, b)

    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="CSV path (created or appended to)")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    info = RunInfo.current()
    csv_path = Path(args.output)
    writer = CsvWriter(csv_path, MICROBENCH_COLUMNS)

    for shape_idx, (M, K, N) in enumerate(SHAPES):
        shape_str = f"{M}x{K}-{K}x{N}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            print(f"[matmul] shape={shape_str} dtype={dtype_str}")
            try:
                trials = run_one(M, K, N, dtype, n_warmup=args.warmup, n_trials=args.trials)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped: {e}")
                continue
            for trial_i, latency_us in enumerate(trials):
                row = info.base_row(backend="pytorch") | {
                    "op": "matmul",
                    "shape": shape_str,
                    "dtype": dtype_str,
                    "trial": trial_i,
                    "latency_us": f"{latency_us:.3f}",
                    "throughput_gflops": f"{gflops_for_matmul(M, N, K, latency_us):.3f}",
                    "memory_mb": "",
                    "workspace_mb": "",
                }
                writer.write(row)


if __name__ == "__main__":
    main()
