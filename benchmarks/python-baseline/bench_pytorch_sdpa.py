"""PyTorch scaled-dot-product attention baseline. Mirrors `SdpaGpuBenchmarks.cs`. We collect three
backend variants:

  * `pytorch-sdpa` — `F.scaled_dot_product_attention` with default backend selection (FlashAttention
    when available on the GPU + dtype combo).
  * `pytorch-sdpa-math` — same call but with `sdp_kernel(enable_flash=False, enable_mem_efficient=False,
    enable_math=True)` — naive math implementation, used as the high-water-mark of bad SDPA. Closest
    apples-to-apples comparison vs our materialize-S baseline.
  * `pytorch-xformers` — xFormers' `memory_efficient_attention` for an alternate tiled implementation.

Speedups in the paper compare HartsyInference's FA2 to `pytorch-sdpa` (the default fast path)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from _common import MICROBENCH_COLUMNS, CsvWriter, RunInfo, time_kernel

# Same shape grid as SdpaGpuBenchmarks.cs.
SHAPES: list[tuple[int, int, int, int, int]] = [
    (1, 16, 1024, 1024, 80),
    (1, 16, 4096, 4096, 80),
    (1, 16, 4096, 77, 80),
    (1, 24, 1024 + 333, 1024 + 333, 64),
    (1, 24, 1024 + 256, 1024 + 256, 128),
    (1, 30, 1024 + 64, 1024 + 64, 128),
    (1, 24, 1024, 1024, 96),
    (1, 16, 4096, 32, 80),
]


def run_sdpa(B, H, Sq, Skv, D, dtype, backends, *, n_warmup, n_trials):
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, Skv, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, Skv, D, device="cuda", dtype=dtype)
    scale = 1.0 / (D ** 0.5)

    def fn() -> None:
        with sdpa_kernel(backends):
            F.scaled_dot_product_attention(q, k, v, scale=scale)

    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def run_xformers(B, H, Sq, Skv, D, dtype, *, n_warmup, n_trials):
    try:
        from xformers.ops import memory_efficient_attention
    except ImportError:
        return None
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=dtype)  # xFormers expects [B,S,H,D]
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, Skv, H, D, device="cuda", dtype=dtype)
    scale = 1.0 / (D ** 0.5)
    fn = lambda: memory_efficient_attention(q, k, v, scale=scale)
    return time_kernel(fn, n_warmup=n_warmup, n_trials=n_trials).trials_us


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    info = RunInfo.current()
    writer = CsvWriter(Path(args.output), MICROBENCH_COLUMNS)

    for B, H, Sq, Skv, D in SHAPES:
        shape_str = f"{B}x{H}x{Sq}x{Skv}x{D}"
        for dtype, dtype_str in [(torch.float32, "f32"), (torch.float16, "f16")]:
            for backends, backend_tag in [
                ([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], "pytorch-sdpa"),
                ([SDPBackend.MATH], "pytorch-sdpa-math"),
            ]:
                print(f"[sdpa] shape={shape_str} dtype={dtype_str} backend={backend_tag}")
                try:
                    trials = run_sdpa(B, H, Sq, Skv, D, dtype, backends, n_warmup=args.warmup, n_trials=args.trials)
                except Exception as e:  # noqa: BLE001
                    print(f"  skipped: {e}")
                    continue
                for trial_i, lat in enumerate(trials):
                    writer.write(info.base_row(backend=backend_tag) | {
                        "op": "sdpa",
                        "shape": shape_str,
                        "dtype": dtype_str,
                        "trial": trial_i,
                        "latency_us": f"{lat:.3f}",
                    })

            print(f"[sdpa] shape={shape_str} dtype={dtype_str} backend=pytorch-xformers")
            try:
                xtrials = run_xformers(B, H, Sq, Skv, D, dtype, n_warmup=args.warmup, n_trials=args.trials)
            except Exception as e:  # noqa: BLE001
                print(f"  skipped: {e}")
                xtrials = None
            if xtrials is None:
                continue
            for trial_i, lat in enumerate(xtrials):
                writer.write(info.base_row(backend="pytorch-xformers") | {
                    "op": "sdpa",
                    "shape": shape_str,
                    "dtype": dtype_str,
                    "trial": trial_i,
                    "latency_us": f"{lat:.3f}",
                })


if __name__ == "__main__":
    main()
