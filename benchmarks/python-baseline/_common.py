"""Shared utilities for the SharpInference PyTorch baseline benchmarks.

This module is intentionally small and dependency-light. It provides:

  * `iso8601_utc()`             — UTC timestamp string for run-id slugs.
  * `gpu_slug()`                — short device slug (e.g. "nvidia-rtx-3060").
  * `write_hardware_fingerprint(out_dir)` — `hardware.txt` matching the C# side.
  * `write_software_fingerprint(out_dir)` — `software.txt` matching the C# side.
  * `time_kernel(fn, n_warmup, n_trials)` — CUDA-event-based timing returning per-trial latencies.
  * `CsvWriter`                 — append-only CSV writer with the canonical schema.

All scripts in this directory import from `_common`. The CSV schemas are documented in
`docs/Research/PROFILING_METHODOLOGY.md` § 14.
"""

from __future__ import annotations

import csv
import os
import platform
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Run identification
# ─────────────────────────────────────────────────────────────────────────────


def iso8601_utc() -> str:
    """ISO-8601 UTC timestamp safe for filesystem paths (no `:`)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def gpu_slug(device: str = "cuda:0") -> str:
    """Short slug derived from `nvidia-smi` device name. Falls back to `unknown-gpu`."""
    if not torch.cuda.is_available():
        return "no-cuda"
    try:
        name = torch.cuda.get_device_name(device)
    except Exception:  # noqa: BLE001
        return "unknown-gpu"
    return (
        name.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def device_compute_cap(device: str = "cuda:0") -> str:
    """Compute capability string like `8.6`."""
    if not torch.cuda.is_available():
        return "n/a"
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}"


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprints
# ─────────────────────────────────────────────────────────────────────────────


def write_hardware_fingerprint(out_dir: Path) -> None:
    """Mirror of the bash hardware.txt block in PROFILING_METHODOLOGY.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    def _capture(label: str, cmd: list[str]) -> None:
        parts.append(f"## {label}")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            parts.append(r.stdout)
            if r.stderr:
                parts.append(f"# stderr:\n{r.stderr}")
        except FileNotFoundError:
            parts.append(f"# {cmd[0]} not on PATH; skipped")
        except Exception as e:  # noqa: BLE001
            parts.append(f"# error: {e}")
        parts.append("")

    parts.append(f"## hostname\n{platform.node()}\n")
    parts.append(f"## uname\n{platform.platform()}\n{platform.uname()}\n")

    _capture("nvidia-smi -q (full)", ["nvidia-smi", "-q"])
    _capture(
        "nvidia-smi --query-gpu (machine-readable)",
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,vbios_version,compute_cap,memory.total,power.limit,clocks.max.sm,clocks.max.mem,persistence_mode,ecc.mode.current",
            "--format=csv",
        ],
    )
    _capture("nvidia-smi topo --matrix", ["nvidia-smi", "topo", "--matrix"])
    _capture("lscpu (selected)", ["lscpu"])
    _capture("free -h", ["free", "-h"])

    governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(governor_path):
        with open(governor_path) as f:
            parts.append(f"## CPU governor\n{f.read().strip()}\n")
    else:
        parts.append("## CPU governor\nn/a (sysfs path missing)\n")

    (out_dir / "hardware.txt").write_text("\n".join(parts))


def write_software_fingerprint(out_dir: Path) -> None:
    """Mirror of the software.txt block in PROFILING_METHODOLOGY.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    parts.append(f"## python\n{platform.python_version()}\n{shutil.which('python3') or 'n/a'}\n")
    parts.append(f"## torch\n{torch.__version__}\n")
    parts.append(f"## torch.version.cuda\n{torch.version.cuda}\n")
    parts.append(f"## torch.backends.cudnn version / enabled\n{torch.backends.cudnn.version()} / {torch.backends.cudnn.enabled}\n")

    def _run(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return r.stdout + ("\n# stderr:\n" + r.stderr if r.stderr else "")
        except Exception as e:  # noqa: BLE001
            return f"# error: {e}"

    parts.append(f"## pip freeze (filtered)\n{_run(['pip', 'freeze'])}\n")
    parts.append(f"## nvcc --version\n{_run(['nvcc', '--version'])}\n")
    parts.append(f"## git rev-parse HEAD\n{_run(['git', 'rev-parse', 'HEAD'])}\n")
    parts.append(f"## git status --porcelain\n{_run(['git', 'status', '--porcelain'])}\n")

    (out_dir / "software.txt").write_text("\n".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimingResult:
    """Per-trial latencies in microseconds plus simple stats (mean, min, max)."""

    trials_us: list[float]

    @property
    def mean_us(self) -> float:
        return sum(self.trials_us) / len(self.trials_us)

    @property
    def min_us(self) -> float:
        return min(self.trials_us)

    @property
    def max_us(self) -> float:
        return max(self.trials_us)


def time_kernel(
    fn: Callable[[], Any],
    *,
    n_warmup: int = 1,
    n_trials: int = 5,
    sync_before: bool = True,
) -> TimingResult:
    """Times a single-call closure using CUDA events.

    `fn` must perform exactly one logical work unit per call (one MatMul, one Conv2D, etc.). The
    function returns ``n_trials`` per-trial latencies in microseconds. Trial 0 is the first
    measurement after warmup; warmup invocations are NOT included in the results.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("time_kernel requires CUDA")

    if sync_before:
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_trials)]

    for i in range(n_trials):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()

    trials_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return TimingResult(trials_us=trials_us)


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer (matches schema in PROFILING_METHODOLOGY.md § 14)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical microbench CSV columns. Schema is shared with the C# side.
MICROBENCH_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "gpu_name",
    "gpu_compute_cap",
    "driver_version",
    "cuda_version",
    "backend",
    "op",
    "shape",
    "dtype",
    "trial",
    "latency_us",
    "throughput_gflops",
    "memory_mb",
    "workspace_mb",
)


class CsvWriter:
    """Append-only CSV writer that creates the file with a header on first write."""

    def __init__(self, path: Path, columns: Iterable[str]):
        self._path = path
        self._columns = list(columns)
        self._wrote_header = path.exists() and path.stat().st_size > 0

    def write(self, row: dict[str, Any]) -> None:
        new_file = not self._wrote_header
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._columns)
            if new_file:
                w.writeheader()
                self._wrote_header = True
            # Fill missing columns with empty string for stability
            row = {col: row.get(col, "") for col in self._columns}
            w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Run-info helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunInfo:
    """Bundle of per-run identifiers attached to every CSV row."""

    run_id: str
    timestamp_utc: str
    gpu_name: str
    gpu_compute_cap: str
    driver_version: str
    cuda_version: str

    @staticmethod
    def current(device: str = "cuda:0") -> "RunInfo":
        timestamp = iso8601_utc()
        run_id = f"{timestamp}_{gpu_slug(device)}"
        gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "no-cuda"
        cap = device_compute_cap(device)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            driver = r.stdout.strip()
        except Exception:  # noqa: BLE001
            driver = "unknown"
        cuda_ver = torch.version.cuda or "unknown"
        return RunInfo(
            run_id=run_id,
            timestamp_utc=timestamp,
            gpu_name=gpu_name,
            gpu_compute_cap=cap,
            driver_version=driver,
            cuda_version=cuda_ver,
        )

    def base_row(self, *, backend: str = "pytorch") -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "gpu_name": self.gpu_name,
            "gpu_compute_cap": self.gpu_compute_cap,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "backend": backend,
        }


@contextmanager
def deterministic_seeds(seed: int = 42) -> Iterator[None]:
    """Resets NumPy + PyTorch RNGs to fixed seeds for the duration of the block."""
    import numpy as np

    py_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        np.random.set_state(py_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def gflops_for_matmul(M: int, N: int, K: int, latency_us: float) -> float:
    """Throughput in GFLOPS for a `[M,K] @ [K,N]` GEMM. 2*M*N*K FLOPs per call."""
    return (2.0 * M * N * K) / (latency_us * 1e3)
