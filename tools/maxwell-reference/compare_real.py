"""Compares the engine's --scores-out TSVs with the BF16 reference TSV from dump_real.py.

    python -I tools/maxwell-reference/compare_real.py --ref /runs/maxwell-ref/reference.tsv \
        --engine-greedy /runs/engine/greedy.tsv --engine-forced /runs/engine/forced.tsv --tol-logprob 0.10

Pass (exit 0) only when all of these hold:
  1. The engine's greedy token ids equal the reference's, position for position, and both have the same length.
  2. With the reference tokens teacher-forced, the engine's argmax equals the reference argmax at every position.
  3. With the reference tokens teacher-forced, |engine logprob - reference logprob| of the reference token is at most --tol-logprob
     at every position.

On a failure the script names the first failing position and prints the reference's top-1 minus top-2 logprob margin there
(from the reference's second_id / second_logprob columns), so a near-tie reads differently from a real divergence.
"""

import argparse
import csv
import sys

REQUIRED = ["index", "token_id", "logprob", "top_id", "top_logprob", "second_id", "second_logprob"]


def read_tsv(path, required):
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        rows = list(reader)
    return [
        {
            "index": int(r["index"]), "token_id": int(r["token_id"]), "logprob": float(r["logprob"]),
            "top_id": int(r["top_id"]), "top_logprob": float(r["top_logprob"]),
            **({"second_id": int(r["second_id"]), "second_logprob": float(r["second_logprob"])} if "second_id" in r else {}),
        }
        for r in rows
    ]


def margin(ref_row):
    return ref_row["top_logprob"] - ref_row["second_logprob"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--engine-greedy", required=True)
    p.add_argument("--engine-forced", required=True)
    p.add_argument("--tol-logprob", type=float, default=0.10)
    a = p.parse_args()

    ref = read_tsv(a.ref, REQUIRED)
    greedy = read_tsv(a.engine_greedy, ["index", "token_id", "logprob", "top_id", "top_logprob"])
    forced = read_tsv(a.engine_forced, ["index", "token_id", "logprob", "top_id", "top_logprob"])

    for r in ref:
        if r["second_id"] == r["top_id"] or margin(r) < 0:
            raise SystemExit(f"reference row {r['index']} is malformed: top {r['top_id']} second {r['second_id']} margin {margin(r):.6f}")

    # Tokens first, over the common prefix: a divergence is the finding; a length difference is reported after it.
    for r, g in zip(ref, greedy):
        if r["token_id"] != g["token_id"]:
            print(f"FAIL greedy token at position {r['index']}: reference {r['token_id']}, engine {g['token_id']}; "
                  f"reference margin top-1 minus top-2 there {margin(r):.6f} nats (top {r['top_id']}, second {r['second_id']})")
            return 1
    if len(greedy) != len(ref):
        print(f"FAIL length: the first {min(len(greedy), len(ref))} greedy tokens match, but reference has {len(ref)} tokens and engine greedy "
              f"has {len(greedy)}; the reference stops are read from the checkpoint, so check the engine's stop handling")
        return 1

    if len(forced) != len(ref):
        print(f"FAIL length: reference {len(ref)} rows, engine forced {len(forced)} rows")
        return 1
    worst = 0.0
    for r, f in zip(ref, forced):
        if r["token_id"] != f["token_id"]:
            print(f"FAIL forced token at position {r['index']}: the forced ids differ from the reference file; check the --teacher-force input")
            return 1
        if f["top_id"] != r["top_id"]:
            print(f"FAIL argmax at position {r['index']}: reference top {r['top_id']} (margin {margin(r):.6f} nats over second {r['second_id']}), "
                  f"engine top {f['top_id']}")
            return 1
        diff = abs(f["logprob"] - r["logprob"])
        worst = max(worst, diff)
        if diff > a.tol_logprob:
            print(f"FAIL logprob at position {r['index']}: reference {r['logprob']:.6f}, engine {f['logprob']:.6f}, |diff| {diff:.4f}; "
                  f"reference margin {margin(r):.6f} nats")
            return 1

    print(f"PASS: {len(ref)} positions, greedy tokens identical, argmax identical, max |logprob diff| {worst:.5f} (tolerance {a.tol_logprob})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
