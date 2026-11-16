"""Registered analysis of timing v3 (registration v3/registration-v3.json). Gating statistic: per-process arithmetic mean of the
200 retained public calls; frozen statistics module (seed 274019, 20,000 resamples). Secondary statistics are reported only."""
import json, math, os, statistics, sys
DIAG = "/home/hartsy/evidence/attention-4090/diag-20261115-c"
V = f"{DIAG}/v3"
sys.path.insert(0, "/home/hartsy/Desktop/HartsyInference/.claude/worktrees/attn-h3/benchmarks/attention")
from statistics_protocol import analyze
reg = json.load(open(f"{V}/registration-v3.json"))
CONTROL_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{V}/aa-v3.1-mean200"
CONTROL_ID = sys.argv[2] if len(sys.argv) > 2 else "AA-v3.1-mean200"
def percentile(values, q):
    o = sorted(values); at = (len(o) - 1) * q; lo, hi = math.floor(at), math.ceil(at)
    return o[lo] + (o[hi] - o[lo]) * (at - lo)
def load(dirname, control_id):
    exits = {}
    try:
        timing_lines = open(f"{dirname}/timing-exits.txt").read().splitlines()
    except FileNotFoundError:
        timing_lines = []
    for line in timing_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == control_id and parts[1][:1] == 's' and parts[2].isdigit():
            exits[parts[1]] = int(parts[2])
    procs, problems = {}, []
    for s in range(1, 5):
        for p in range(1, 6):
            for role in ("cand", "ref"):
                name = f"s{s}-p{p}-{role}"
                path = f"{dirname}/{name}.json"
                if exits.get(name) != 0 or not os.path.exists(path):
                    problems.append(f"{name}: exit {exits.get(name)} or missing json"); continue
                d = json.load(open(path)); ms = d.get("milliseconds", [])
                ok = (d.get("branch") == "S.15" and d.get("quality_passed") is True and d.get("telemetry") == "disabled"
                      and len(ms) == 200 and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in ms))
                if not ok:
                    problems.append(f"{name}: validity"); continue
                med = statistics.median(ms)
                procs[name] = {"ms": ms, "mean": statistics.fmean(ms), "median": med,
                               "p90": percentile(ms, .90), "p99": percentile(ms, .99), "max": max(ms),
                               "stall_rate_gt2x_median": sum(x > 2 * med for x in ms) / 200,
                               "tail_time_share": sum(max(x - med, 0) for x in ms) / sum(ms)}
    return procs, problems
def pairs(procs, stat):
    out = {}
    for s in range(1, 5):
        out[s] = [{"candidate_ms": procs[f"s{s}-p{p}-cand"][stat], "reference_ms": procs[f"s{s}-p{p}-ref"][stat], "valid": True}
                  for p in range(1, 6)] if all(f"s{s}-p{p}-{r}" in procs for p in range(1, 6) for r in ("cand", "ref")) else None
    return out
def run(dirname, control_id, cand_label):
    procs, problems = load(dirname, control_id)
    result = {"id": control_id, "valid_processes": len(procs), "problems": problems}
    if problems or len(procs) != 40:
        result["verdict"] = "INVALID"; return result
    pr = pairs(procs, "mean")
    inf, conf = [pr[1], pr[2], pr[3]], pr[4]
    result["gating_statistic"] = {c: analyze(inf, conf, c) for c in ("noninferiority", "regression")}
    result["noninferiority_passed"] = result["gating_statistic"]["noninferiority"]["passed"]
    result["regression_passed"] = result["gating_statistic"]["regression"]["passed"]
    if result["regression_passed"]:
        result["outcome"] = "regression detected (lower95 > 1.03 and confirmation > 1.03)"
    elif result["noninferiority_passed"]:
        result["outcome"] = "noninferiority established (upper95 <= 1.03 and confirmation <= 1.03)"
    else:
        result["outcome"] = "inconclusive (neither rule met)"
    med_pr = pairs(procs, "median")
    result["secondary_median_pairs"] = {c: analyze([med_pr[1], med_pr[2], med_pr[3]], med_pr[4], c) for c in ("noninferiority", "regression")}
    result["secondary_per_process"] = {k: {kk: v[kk] for kk in ("mean", "median", "p90", "p99", "max", "stall_rate_gt2x_median", "tail_time_share")}
                                       for k, v in procs.items()}
    return result
out = {"registration_sha256": reg.get("registration_sha256_note", "see sha256 of registration-v3.json"),
       "control": run(CONTROL_DIR, CONTROL_ID, "B0"),
       "campaign": run(f"{V}/p1-vs-b0-v3-mean200", "P1-vs-B0-v3-mean200", "P1")}
json.dump(out, open(f"{V}/analysis-v3.json", "w"), indent=1)
for key in ("control", "campaign"):
    r = out[key]
    print(key, r["id"], "valid processes", r["valid_processes"], "problems", r["problems"][:3])
    if "gating_statistic" in r:
        g = r["gating_statistic"]
        for c in ("noninferiority", "regression"):
            print("  ", c, {k: round(v, 4) if isinstance(v, float) else v for k, v in g[c].items() if k != "ci95_two_sided"})
        print("   outcome:", r["outcome"])
