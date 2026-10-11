#!/usr/bin/env python3
"""Turns the bench_vs_llama.py summary tables of one interleaved run (llama.cpp vs HartsyInference) into the markdown of the results document.
usage: make_report.py <dir with <model>.summary.md files> <out.md>"""
import sys, re, pathlib

MODELS = [("granite-3b", "Granite-3.0-3B-A800M instruct", "granite-moe, top-8 of 40"),
          ("olmoe", "OLMoE-1B-7B-0924 instruct", "olmoe, top-8 of 64"),
          ("qwen3-30b", "Qwen3-30B-A3B", "qwen3moe, top-8 of 128"),
          ("mixtral", "Mixtral-8x7B-Instruct v0.1", "llama + experts, top-2 of 8")]

def parse(path):
    rows = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.startswith("|") or line.startswith("|---") or "config" in line.split("|")[1]:
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        rows[(c[0], c[1], c[2])] = dict(n=int(c[3]), pp=float(c[4]), tg=float(c[5]), ttft=float(c[6]), vram=c[8], rss=c[9])
    return rows

def main(d, out):
    d = pathlib.Path(d)
    md = []
    for key, title, arch in MODELS:
        f = d / f"{key}.summary.md"
        if not f.exists():
            continue
        r = parse(f)
        cells = sorted({(k[1], k[2]) for k in r}, key=lambda x: (0 if x[0] == "4k" else 1, x[1]))
        md.append(f"### {title} ({arch})\n")
        md.append("| ctx | temp | metric | llama.cpp | hartsy | ratio | hartsy + F16 KV | ratio |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|")
        for ctx, temp in cells:
            L = r.get(("llama-cpp-gpu", ctx, temp)); H = r.get(("hartsy-cuda", ctx, temp)); K = r.get(("hartsy-cuda-kvf16", ctx, temp))
            if not L:
                continue
            for metric, label, fmt in (("pp", "prefill tok/s", "{:,.0f}"), ("tg", "decode tok/s", "{:,.1f}"), ("ttft", "TTFT s", "{:,.2f}")):
                def cell(x):
                    return (fmt.format(x[metric]) if x else "-")
                def ratio(x):
                    if not x:
                        return "-"
                    q = x[metric] / L[metric]
                    return f"{q:.2f}x" if metric != "ttft" else f"{L[metric]/x[metric]:.2f}x"
                md.append(f"| {ctx} | {temp} | {label} | {fmt.format(L[metric])} | {cell(H)} | {ratio(H)} | {cell(K)} | {ratio(K)} |")
        md.append("")
    pathlib.Path(out).write_text("\n".join(md))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
