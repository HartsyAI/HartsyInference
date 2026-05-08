"""PyTorch elementwise (Silu / Gelu / BroadcastAdd) baseline. Mirrors `ElementwiseGpuBenchmarks.cs`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import MICROBENCH_COLUMNS, CsvWriter, RunInfo, time_kernel

SIZES: list[tuple[int, int, int, int]] = [
    (1, 4096, 1, 1280),
    (1, 1280, 32, 32),
    (1, 320, 128, 128),
    (1, 64, 1, 1),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    info = RunInfo.current()
    writer = CsvWriter(Path(args.output), MICROBENCH_COLUMNS)

    for B, C, H, W in SIZES:
        shape_str = f"{B}x{C}x{H}x{W}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            x = torch.randn(B, C, H, W, device="cuda", dtype=dtype)
            bias = torch.randn(C, device="cuda", dtype=dtype)

            for op_name, fn in [
                ("silu", lambda: F.silu(x)),
                ("gelu", lambda: F.gelu(x)),
                ("broadcast_add", lambda: x + bias.view(1, C, 1, 1)),
            ]:
                print(f"[elementwise] op={op_name} shape={shape_str} dtype={dtype_str}")
                try:
                    trials = time_kernel(fn, n_warmup=args.warmup, n_trials=args.trials).trials_us
                except Exception as e:  # noqa: BLE001
                    print(f"  skipped: {e}")
                    continue
                for trial_i, lat in enumerate(trials):
                    writer.write(info.base_row(backend="pytorch") | {
                        "op": op_name,
                        "shape": shape_str,
                        "dtype": dtype_str,
                        "trial": trial_i,
                        "latency_us": f"{lat:.3f}",
                    })


if __name__ == "__main__":
    main()
