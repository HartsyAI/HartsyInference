"""Joins SharpInference C# microbenchmark results with the PyTorch baselines and emits a
side-by-side comparison report. Implements the statistical analysis described in
docs/Research/PROFILING_METHODOLOGY.md § 11.

Usage:
    python3 analyze.py --csharp <path> --pytorch <path> --output-dir <dir>

Outputs `comparison.csv` and `comparison.md` in the output directory. Designed to be best-effort:
if either CSV is missing, the script writes whatever it can and exits 0 (so the parent harness
script doesn't fail just because the analysis is incomplete).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scipy import stats  # type: ignore
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_stddev(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (math.nan, math.nan)
    if n == 1:
        return (values[0], 0.0)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return (m, math.sqrt(var))


def ci95_halfwidth(values: list[float]) -> float:
    """Half-width of the 95% confidence interval for the mean (Student-t, df=n-1).

    Approximate t critical values for df = 1..10 → values from a stats table. For larger n we use
    the limit (1.96).
    """
    n = len(values)
    if n < 2:
        return math.nan
    _, sd = mean_stddev(values)
    sem = sd / math.sqrt(n)
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    t = t_table.get(n - 1, 1.96)
    return t * sem


def welch_ttest(a: list[float], b: list[float]) -> float | None:
    """Returns the p-value of Welch's two-sided t-test, or None if scipy unavailable / n too small."""
    if len(a) < 2 or len(b) < 2:
        return None
    if HAVE_SCIPY:
        result = stats.ttest_ind(a, b, equal_var=False)
        return float(result.pvalue)
    # Manual Welch's t-test fallback
    ma, sa = mean_stddev(a)
    mb, sb = mean_stddev(b)
    na, nb = len(a), len(b)
    if sa == 0 and sb == 0:
        return 0.0 if ma != mb else 1.0
    se = math.sqrt(sa * sa / na + sb * sb / nb)
    if se == 0:
        return None
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    if (sa * sa / na + sb * sb / nb) > 0:
        df_num = (sa * sa / na + sb * sb / nb) ** 2
        df_den = (sa ** 4) / ((na ** 2) * (na - 1)) + (sb ** 4) / ((nb ** 2) * (nb - 1))
        df = df_num / df_den if df_den > 0 else min(na - 1, nb - 1)
    else:
        df = min(na - 1, nb - 1)
    # Approximate two-sided p via survival function of t
    # Without scipy, use a simple normal approximation (df > 30): p ~= 2 * (1 - Phi(|t|))
    if df > 30:
        # erf-based normal CDF
        z = abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))
        return p
    return None  # not accurate enough without scipy at small df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[float]]:
    """Groups CSV rows into a {(backend, op, shape, dtype) → [latency_us]} map."""
    out: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        backend = r.get("backend") or r.get("Backend") or "unknown"
        op = r.get("op") or "?"
        shape = r.get("shape") or "?"
        dtype = r.get("dtype") or "?"
        lat = r.get("latency_us") or r.get("Mean") or ""
        try:
            v = float(lat)
        except (TypeError, ValueError):
            continue
        # BDN's "Mean" is in nanoseconds — convert to microseconds when source looks like a BDN row.
        if "Mean" in r:
            v = v / 1000.0
        out[(backend, op, shape, dtype)].append(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csharp", required=True, help="microbench_csharp.csv path (BDN raw or canonical)")
    p.add_argument("--pytorch", required=True, help="microbench_pytorch.csv path (canonical)")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csharp_rows = load_csv(Path(args.csharp))
    pytorch_rows = load_csv(Path(args.pytorch))

    if not csharp_rows and not pytorch_rows:
        (out_dir / "comparison.md").write_text("# Comparison\n\nNo input CSVs found. Re-run the harness.\n")
        return

    csharp_agg = aggregate(csharp_rows)
    pytorch_agg = aggregate(pytorch_rows)

    # Find every (op, shape, dtype) tuple present in either side
    all_keys: set[tuple[str, str, str]] = set()
    for backend, op, shape, dtype in csharp_agg:
        all_keys.add((op, shape, dtype))
    for backend, op, shape, dtype in pytorch_agg:
        all_keys.add((op, shape, dtype))

    # Pick the best backend per side: SharpInference for C# (only one), pytorch (default fast path)
    # for Python. Other Python variants (math, xformers) end up in the appendix.
    sharpinf_keys = {(op, shape, dtype): trials
                     for (b, op, shape, dtype), trials in csharp_agg.items()
                     if "SharpInference" in b or b == "sharpinference_cuda" or b.startswith("Job")}
    pytorch_main = {(op, shape, dtype): trials
                    for (b, op, shape, dtype), trials in pytorch_agg.items()
                    if b == "pytorch" or b == "pytorch-sdpa"}

    out_csv = out_dir / "comparison.csv"
    out_md = out_dir / "comparison.md"

    fieldnames = [
        "op", "shape", "dtype",
        "csharp_n", "csharp_mean_us", "csharp_ci95_us",
        "pytorch_n", "pytorch_mean_us", "pytorch_ci95_us",
        "speedup_x_csharp_over_pytorch", "p_value", "significant",
    ]
    rows_out: list[dict[str, Any]] = []

    for op, shape, dtype in sorted(all_keys):
        cs = sharpinf_keys.get((op, shape, dtype), [])
        py = pytorch_main.get((op, shape, dtype), [])
        cs_m, _ = mean_stddev(cs) if cs else (math.nan, math.nan)
        py_m, _ = mean_stddev(py) if py else (math.nan, math.nan)
        cs_ci = ci95_halfwidth(cs) if cs else math.nan
        py_ci = ci95_halfwidth(py) if py else math.nan
        speedup = (py_m / cs_m) if cs_m and not math.isnan(py_m) else math.nan
        p = welch_ttest(cs, py) if (cs and py) else None
        significant = (p is not None and p < 0.01)
        rows_out.append({
            "op": op, "shape": shape, "dtype": dtype,
            "csharp_n": len(cs),
            "csharp_mean_us": f"{cs_m:.3f}" if not math.isnan(cs_m) else "",
            "csharp_ci95_us": f"{cs_ci:.3f}" if not math.isnan(cs_ci) else "",
            "pytorch_n": len(py),
            "pytorch_mean_us": f"{py_m:.3f}" if not math.isnan(py_m) else "",
            "pytorch_ci95_us": f"{py_ci:.3f}" if not math.isnan(py_ci) else "",
            "speedup_x_csharp_over_pytorch": f"{speedup:.3f}" if not math.isnan(speedup) else "",
            "p_value": f"{p:.4g}" if p is not None else "",
            "significant": "yes" if significant else "no" if p is not None else "",
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Markdown
    md = ["# SharpInference vs PyTorch — Comparison", ""]
    md.append(f"- C# rows considered: {len(csharp_rows)}")
    md.append(f"- PyTorch rows considered: {len(pytorch_rows)}")
    md.append(f"- Joined op-shape-dtype tuples: {len(rows_out)}")
    md.append(f"- scipy available: {HAVE_SCIPY}")
    md.append("")
    md.append("Speedup = PyTorch_mean / SharpInference_mean. >1 = C# faster, <1 = PyTorch faster.")
    md.append("")
    md.append("| op | shape | dtype | C# (µs) | PyTorch (µs) | Speedup × | p | sig |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows_out, key=lambda r: float(r["speedup_x_csharp_over_pytorch"] or 0), reverse=True):
        md.append(
            f"| {r['op']} | {r['shape']} | {r['dtype']} | "
            f"{r['csharp_mean_us']}±{r['csharp_ci95_us']} (n={r['csharp_n']}) | "
            f"{r['pytorch_mean_us']}±{r['pytorch_ci95_us']} (n={r['pytorch_n']}) | "
            f"{r['speedup_x_csharp_over_pytorch']} | {r['p_value']} | {r['significant']} |"
        )
    out_md.write_text("\n".join(md) + "\n")

    print(f"[analyze] wrote {out_csv}")
    print(f"[analyze] wrote {out_md}")


if __name__ == "__main__":
    main()
