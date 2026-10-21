"""Turns profile_attention_share.sh output into the attention share of a denoise step per case.

Per case it reads prof{S}.profile.txt and prof{2S}.profile.txt (the synced op profiler's per-label call counts and
milliseconds) and the step ticks of plain{2S}.log, prof{S}.log and prof{2S}.log. Per-step figures are the 2S-minus-S
difference divided by S, which cancels load, encoding, decode and the first step. Attention is every SDPA, SDPA-TM and
FlashAttention label, minus an SDPA that a token-major call made on its permute fallback (same geometry, same calls per
step). Attention-adjacent is the work the shared layer removes: permutes, head repeats and the standalone RoPE ops.
A model with a fixed schedule runs the same steps in both runs; its per-step figures are the longer run's totals over
its step count, counting only labels called a whole number of times per step (the denoiser's, not the encoders').
Amdahl: a step-time reduction of f*(1 - 1/r) and an overall speedup of 1/((1 - f) + f/r) for attention speedup r.
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

ADJACENT = re.compile(r"^(Permute0213|RepeatKvHeads|ApplyRope|RopeInterleaved|Ltx2SplitRope|Ltx2QkNormRope|Cast)")
TICK = re.compile(r"\[(\d+)/(\d+)\] (\d+) ms")


def table(path):
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            calls, total = int(parts[-3]), float(parts[-2])
        except ValueError:
            continue
        rows[" ".join(parts[:-3])] = (calls, total)
    return rows


def step_ms(log, first=2):
    if not log.exists():
        return None
    # The mean, not the median: an unsynced pipeline enqueues steps ahead of the GPU, so single ticks read 0 ms and a
    # later one absorbs the backlog; only their sum over steps 2..N is the steps' real time.
    ticks = [int(m.group(3)) for m in TICK.finditer(log.read_text(errors="replace")) if int(m.group(1)) >= first]
    return round(statistics.fmean(ticks), 1) if ticks else None


def run_steps(log):
    """The step count a run actually took, from its [i/N] ticks."""
    if not log.exists():
        return None
    ns = [int(m.group(2)) for m in re.finditer(r"\[(\d+)/(\d+)\]", log.read_text(errors="replace"))]
    return max(ns) if ns else None


def geometry(label):
    """(sq, skv, heads, d) of an attention label, to pair a token-major call with its head-major fallback."""
    m = re.match(r"SDPA-TM (\d+)x(\d+)x(\d+)x(\d+)", label)
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.match(r"SDPA \S+ q\[(\d+), (\d+), (\d+), (\d+)\] k\[\d+, \d+, (\d+), \d+\]", label)
    if m:
        b, h, sq, d, skv = (int(x) for x in m.groups())
        return (sq, skv, h, d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--r", default="1.5,1.9,2.5", help="attention speedups to project")
    a = ap.parse_args()
    s = a.steps
    rs = [float(x) for x in a.r.split(",")]
    root = Path(a.out)
    print(f"| case | step ms (plain) | step ms (synced) | attention ms/step | share | adjacent ms/step | adj share | "
          + " | ".join(f"Δstep @ r={r}" for r in rs) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(rs))
    shapes = []
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        t1, t2 = table(case / f"prof{s}.profile.txt"), table(case / f"prof{2 * s}.profile.txt")
        if not t1 or not t2:
            print(f"| {case.name} | missing profile | | | | | |" + " |" * len(rs))
            continue
        per, flagged = {}, set()
        n1, n2 = run_steps(case / f"prof{s}.log"), run_steps(case / f"prof{2 * s}.log")
        fixed = n2 if n1 is not None and n1 == n2 else None
        for label, (c2, m2) in t2.items():
            if fixed:
                if c2 and c2 % fixed == 0:
                    per[label] = (c2 / fixed, m2 / fixed)
                    flagged.add(label)
                continue
            c1, m1 = t1.get(label, (0, 0.0))
            calls = (c2 - c1) / s
            # Per call: the run-to-run slope, unless a one-time cost (a plan build, a JIT, a cold weight upload) landed in
            # one run and not the other; then the longer run's average, which dilutes that cost, and the label is flagged.
            avg = m2 / c2 if c2 else 0.0
            slope = (m2 - m1) / (c2 - c1) if c2 > c1 else 0.0
            per_call = slope if 0 < slope <= avg else avg
            if per_call != slope:
                flagged.add(label)
            per[label] = (calls, calls * per_call)
        attn = {k: v for k, v in per.items() if k.startswith(("SDPA", "FlashAttention")) and v[0] > 0}
        tm = {geometry(k): v[0] for k, v in attn.items() if k.startswith("SDPA-TM")}
        nested = [k for k, v in attn.items() if k.startswith("SDPA ") and geometry(k) in tm and abs(tm[geometry(k)] - v[0]) < 1e-6]
        attn_ms = sum(v[1] for k, v in attn.items() if k not in nested)
        adj_ms = sum(v[1] for k, v in per.items() if ADJACENT.match(k) and v[0] > 0)
        # The synced run's ticks are per-step accurate, so its denominator skips step 2 (first-use uploads and plan builds,
        # which the per-call estimate above also excludes); the plain run is async and only its sum over 2..N is real.
        plain, synced = step_ms(case / f"plain{2 * s}.log"), step_ms(case / f"prof{2 * s}.log", first=3)
        denom = synced or sum(v[1] for v in per.values())
        f = attn_ms / denom if denom else float("nan")
        cells = [f"{f * (1 - 1 / r) * 100:.1f}% (×{1 / ((1 - f) + f / r):.2f})" for r in rs]
        name = f"{case.name} (fixed {fixed} steps)" if fixed else case.name
        print(f"| {name} | {plain or 'NA'} | {synced or 'NA'} | {attn_ms:.1f} | {f * 100:.1f}% | {adj_ms:.1f} | "
              f"{adj_ms / denom * 100 if denom else float('nan'):.1f}% | " + " | ".join(cells) + " |")
        for k, (calls, ms) in sorted(attn.items(), key=lambda kv: -kv[1][1]):
            shapes.append((case.name, k, calls, ms, k in nested, k in flagged, fixed))
    print("\nAttention calls per step (from the shape-split labels; the shape table is frozen from these):\n")
    print("| case | label | calls/step | ms/step | note |")
    print("|---|---|---|---|---|")
    for case, k, calls, ms, nest, flag, fixed in shapes:
        notes = [n for n, on in (("nested in SDPA-TM", nest), (f"run total over its fixed {fixed} steps", fixed),
                                 ("per-call cost from the longer run's average", flag and not fixed)) if on]
        print(f"| {case} | `{k}` | {calls:g} | {ms:.2f} | {'; '.join(notes)} |")


if __name__ == "__main__":
    sys.exit(main())
