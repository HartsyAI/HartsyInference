"""PyTorch and Comfy Kitchen attention at the shapes in attention_shapes.json — the external half of the attention
shape table; the engine half is `HartsyInference.GpuBenchmarks attention-table`. Rows use the same columns, the same
uniform [-1,1) inputs rounded to the arm's dtype, and the same oracle: float64 attention on a sample of (batch, head,
query row) triples (heads 0, H/2, H-1; up to 64 rows including the first and last).

Arms: torch's default SDPA ordering and each SDPA backend forced (FLASH, CUDNN, EFFICIENT, MATH when the score
matrix fits), in fp16 and bf16, plus comfy_kitchen.int8_attention. A forced backend that refuses a shape is
recorded as `n/a` with torch's reason, never retried on another backend.

Run with the interpreter that has torch (the SwarmUI ComfyUI venv on this box), `-s` so user site-packages stay out:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> "$VENV_PY" -s bench_attention.py --output OUT.tsv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HEADER = ("utc\tdevice\tshape\tarm\tcall\tkernel\tstatus\tdtype\tkv_dtype\tlayout\tmask\tcausal\tb\thq\thkv\tsq\tskv\td"
          "\twarmup\ttrials\tmedian_ms\tp10_ms\tp90_ms\tmin_ms\ttflops\trel_l2\tcos\tmax_abs\toracle_rows\tknobs"
          "\tpeak_vram_mb\tthrottle_frac\tsm_clock_med\tevent_median_ms")

BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
            "eff": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}


def uniform(shape, seed, dtype):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(shape, generator=g, device="cuda", dtype=torch.float32) * 2 - 1).to(dtype)


def make_inputs(s, dtype, seed=1234):
    q = uniform((s["b"], s["hq"], s["sq"], s["d"]), seed * 7 + 1, dtype)
    k = uniform((s["b"], s["hkv"], s["skv"], s["d"]), seed * 7 + 2, dtype)
    v = uniform((s["b"], s["hkv"], s["skv"], s["d"]), seed * 7 + 3, dtype)
    bias = None
    if s["mask"] == "bias-head":
        bias = uniform((1, s["hq"], s["sq"], s["skv"]), seed * 7 + 4, torch.float32) * 4
    elif s["mask"] == "bias-key":
        bias = torch.zeros((1, 1, 1, s["skv"]), device="cuda", dtype=torch.float32)
        bias[..., s["skv"] - max(1, s["skv"] // 10):] = -1e30
    return q, k, v, bias


def causal_bias(s):
    """Bottom-right aligned causal bias for a query block at the end of the cache (torch's is_causal is top-left)."""
    q_off = s["skv"] - s["sq"]
    i = torch.arange(s["sq"], device="cuda")[:, None] + q_off
    j = torch.arange(s["skv"], device="cuda")[None, :]
    return torch.where(j <= i, 0.0, float("-inf")).to(torch.float32)[None, None]


def sample(s, rows=64):
    heads = sorted({0, s["hq"] // 2, s["hq"] - 1})
    batches = sorted({0, s["b"] - 1})
    n = min(rows, s["sq"])
    qrows = sorted({0 if n == 1 else (i * (s["sq"] - 1)) // (n - 1) for i in range(n)})
    return [(b, h, r) for b in batches for h in heads for r in qrows]


def oracle_compare(s, q, k, v, bias, out, scale, rows=64):
    """float64 reference on the sampled rows; relative L2 floored at 1e-6·max|ref|, cosine, max abs."""
    group = s["hq"] // s["hkv"]
    q_off = s["skv"] - s["sq"] if s["causal"] else 0
    num = den = dot = na = nr = max_abs = 0.0
    refs = []
    ref_max = 0.0
    for (b, h, r) in sample(s, rows):
        kv = h // group
        qi = q[b, h, r].double()
        visible = min(s["skv"], q_off + r + 1) if s["causal"] else s["skv"]
        sc = (k[b, kv, :visible].double() @ qi) * scale
        if bias is not None:
            sc = sc + (bias[0, h, r, :visible] if bias.shape[1] > 1 else bias[0, 0, 0, :visible]).double()
        p = torch.softmax(sc, dim=0)
        ref = p @ v[b, kv, :visible].double()
        refs.append((ref, out[b, h, r].double()))
        ref_max = max(ref_max, ref.abs().max().item())
    floor = 1e-6 * ref_max
    for ref, act in refs:
        e = act - ref
        num += (e * e).sum().item()
        den += torch.clamp(ref * ref, min=floor * floor).sum().item()
        dot += (act * ref).sum().item()
        na += (act * act).sum().item()
        nr += (ref * ref).sum().item()
        max_abs = max(max_abs, e.abs().max().item())
    rel = math.sqrt(num / den) if den > 0 else float("nan")
    cos = dot / math.sqrt(na * nr) if na > 0 and nr > 0 else float("nan")
    return rel, cos, max_abs, len(refs)


def flops(s):
    return 4.0 * s["b"] * s["hq"] * s["sq"] * s["skv"] * s["d"] * (0.5 if s["causal"] and s["sq"] == s["skv"] else 1.0)


def arms_for(s):
    arms = []
    for dt in ("f16", "bf16"):
        arms.append((f"torch-def-{dt}", "torch", None, dt))
        for name in ("flash", "cudnn", "eff", "math"):
            arms.append((f"torch-{name}-{dt}", "torch", name, dt))
        if s["entry"] == "sdpa" and not s["causal"]:
            arms.append((f"ck-int8-{dt}", "ck", None, dt))
    return arms


def call_for(s, arm, q, k, v, bias, scale):
    _, kind, backend, _ = arm
    gqa = s["hq"] != s["hkv"]
    attn_mask = None
    is_causal = False
    if s["causal"]:
        if s["sq"] == s["skv"]:
            is_causal = True
        elif s["sq"] > 1:
            attn_mask = causal_bias(s).to(q.dtype)
    if bias is not None:
        attn_mask = bias.to(q.dtype)
    if kind == "ck":
        import comfy_kitchen.sage_attention as sa
        return lambda: sa.int8_attention(q, k, v, scale=scale, attn_mask=attn_mask)
    if backend is None:
        return lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale, enable_gqa=gqa)
    be = BACKENDS[backend]

    def fn():
        with sdpa_kernel([be]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale, enable_gqa=gqa)
    return fn


def pct(xs, p):
    xs = sorted(xs)
    r = p / 100 * (len(xs) - 1)
    lo, hi = math.floor(r), math.ceil(r)
    return xs[lo] + (xs[hi] - xs[lo]) * (r - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=str(Path(__file__).with_name("attention_shapes.json")))
    ap.add_argument("--output", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--arms", default="", help="comma list of arm names (default: all)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--smi-index", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                    help="nvidia-smi index of the card (nvidia-smi ignores the CUDA mask); defaults to the mask under PCI_BUS_ID order")
    a = ap.parse_args()

    shapes = json.load(open(a.shapes))["shapes"]
    only = set(filter(None, a.only.split(",")))
    want = set(filter(None, a.arms.split(",")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.cuda.get_device_name()
    try:
        import comfy_kitchen
        ck_ver = getattr(comfy_kitchen, "__version__", None) or __import__("importlib.metadata").metadata.version("comfy_kitchen")
    except Exception:  # noqa: BLE001
        ck_ver = "absent"
    knobs = f"torch={torch.__version__},comfy_kitchen={ck_ver},sageattention=absent"
    out = Path(a.output)
    new = not out.exists()
    smi_path = out.with_suffix(".smi.csv")
    smi = subprocess.Popen(["nvidia-smi", "-i", a.smi_index, "--query-gpu=timestamp,memory.used,clocks.sm,clocks_event_reasons.hw_slowdown,"
                            "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,"
                            "clocks_event_reasons.hw_power_brake_slowdown", "--format=csv,noheader,nounits", "-lms", "100"],
                           stdout=open(smi_path, "w"), stderr=subprocess.DEVNULL)
    windows = []
    rows_out = []
    try:
        for s in shapes:
            if only and s["id"] not in only:
                continue
            scale = 1.0 / math.sqrt(s["d"])
            for arm in arms_for(s):
                if want and arm[0] not in want:
                    continue
                name, kind, backend, dts = arm
                dtype = torch.float16 if dts == "f16" else torch.bfloat16
                status, kernel = "ok", (backend or ("int8" if kind == "ck" else "default"))
                ms, ev = [], []
                rel = cos = mx = float("nan")
                nrows, peak = 0, float("nan")
                if backend == "math" and s["b"] * s["hq"] * s["sq"] * s["skv"] * 4 > 4 << 30:
                    status = "n/a-memory"
                else:
                    torch.cuda.empty_cache()
                    q, k, v, bias = make_inputs(s, dtype)
                    t0 = time.time()
                    try:
                        fn = call_for(s, arm, q, k, v, bias, scale)
                        base = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        o = fn()
                        torch.cuda.synchronize()
                        for _ in range(a.warmup):
                            fn()
                        torch.cuda.synchronize()
                        for _ in range(a.trials):
                            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            w0 = time.perf_counter()
                            st.record()
                            o = fn()
                            en.record()
                            torch.cuda.synchronize()
                            ms.append((time.perf_counter() - w0) * 1e3)
                            ev.append(st.elapsed_time(en))
                        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
                        rel, cos, mx, nrows = oracle_compare(s, q, k, v, bias, o, scale, a.rows)
                    except Exception as e:  # noqa: BLE001
                        status = "n/a:" + (type(e).__name__ + ":" + str(e)).replace("\t", " ").replace("\n", " ")[:160]
                    windows.append((len(rows_out), t0, time.time()))
                    del q, k, v, bias
                med = statistics.median(ms) if ms else float("nan")
                f = lambda x, fmt: "NA" if x != x else format(x, fmt)  # noqa: E731
                row = [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), device, s["id"], name, "kernel", kernel, status,
                       dts, dts, s["layout"], s["mask"], "1" if s["causal"] else "0", s["b"], s["hq"], s["hkv"], s["sq"], s["skv"], s["d"],
                       a.warmup, a.trials, f(med, ".3f"), f(pct(ms, 10) if ms else float("nan"), ".3f"), f(pct(ms, 90) if ms else float("nan"), ".3f"),
                       f(min(ms) if ms else float("nan"), ".3f"), f(flops(s) / (med * 1e-3) / 1e12 if ms else float("nan"), ".2f"),
                       f(rel, ".3E"), f(cos, ".7f"), f(mx, ".3E"), nrows, knobs, f(peak, ".0f"), "NA", "NA",
                       f(statistics.median(ev) if ev else float("nan"), ".3f")]
                rows_out.append([str(x) for x in row])
                print(f"{s['id']:16s} {name:18s} {status[:40]:40s} {row[20]:>10s} ms  relL2 {row[25]}", flush=True)
    finally:
        smi.terminate()
        smi.wait()
    # Per-arm throttle fraction and SM clock from the sampler, by wall-clock window.
    samples = []
    for line in open(smi_path):
        p = [x.strip() for x in line.split(",")]
        if len(p) != 7:
            continue
        try:
            ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        samples.append((ts, int(p[2]), any(x == "Active" for x in p[3:])))
    for idx, t0, t1 in windows:
        w = [x for x in samples if t0 <= x[0] <= t1]
        if w:
            rows_out[idx][31] = f"{sum(1 for x in w if x[2]) / len(w):.3f}"
            rows_out[idx][32] = f"{statistics.median(x[1] for x in w)}"
    with open(out, "a") as fh:
        if new:
            fh.write(HEADER + "\n")
        for r in rows_out:
            fh.write("\t".join(r) + "\n")
    print(f"done: {out}")


if __name__ == "__main__":
    sys.exit(main())
