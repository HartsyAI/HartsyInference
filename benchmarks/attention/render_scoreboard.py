"""Renders the attention scoreboard rows from table runs: the engine's table.tsv (run_attention_table.sh) and the
external one (bench_attention.py), for one card. Per shape: our fastest arm that ran ok, our production-default arm,
torch's fastest backend, Comfy Kitchen INT8, and the ratios. Only rows with status ok count; error is the oracle relL2.

    python3 render_scoreboard.py --engine 3060/table/table.tsv --external 3060/python/attention.tsv --card "RTX 3060"
"""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    if not path or not Path(path).exists():
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r.get("status") == "ok" and r.get("median_ms") not in (None, "NA")]


def ms(r):
    return float(r["median_ms"])


def cell(r, with_kernel=True):
    if r is None:
        return "—"
    kernel = f"{r['kernel']} " if with_kernel else ""
    return f"{kernel}{ms(r):.2f} ms ({float(r['rel_l2']):.1e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--external")
    ap.add_argument("--card", required=True)
    ap.add_argument("--shapes", default=str(Path(__file__).resolve().parents[1] / "python-baseline" / "attention_shapes.json"))
    a = ap.parse_args()
    ours, ext = load(a.engine), load(a.external)
    order = [s["id"] for s in json.load(open(a.shapes))["shapes"]]
    print(f"### {a.card}\n")
    print("| shape | ours: fastest arm | ours: production default | torch: fastest | Comfy Kitchen INT8 | ours ÷ torch | ours ÷ CK |")
    print("|---|---|---|---|---|---|---|")
    for sid in order:
        o = [r for r in ours if r["shape"] == sid]
        if not o:
            continue
        best = min(o, key=ms)
        default = next((r for r in o if r["arm"] in ("def-f16", "fa-def") and r["call"] == ("path" if any(x["call"] == "path" for x in o) else "kernel")), None)
        t = [r for r in ext if r["shape"] == sid and r["arm"].startswith("torch-")]
        tb = min(t, key=ms) if t else None
        ck = [r for r in ext if r["shape"] == sid and r["arm"].startswith("ck-int8")]
        cb = min(ck, key=ms) if ck else None
        tname = f"{tb['arm'].removeprefix('torch-')} {ms(tb):.2f} ms ({float(tb['rel_l2']):.1e})" if tb else "—"
        r1 = f"{ms(best) / ms(tb):.2f}×" if tb else "—"
        r2 = f"{ms(best) / ms(cb):.2f}×" if cb else "—"
        print(f"| {sid} | {best['arm']} {best['call']}: {cell(best)} | {cell(default)} | {tname} | {cell(cb, False)} | {r1} | {r2} |")


if __name__ == "__main__":
    main()
