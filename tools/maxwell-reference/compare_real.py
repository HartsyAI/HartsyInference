"""Compares the engine's --scores-out TSV with the BF16 reference TSV from dump_real.py.

    python -I tools/maxwell-reference/compare_real.py --ref /runs/maxwell-ref/reference.tsv \
        --engine-greedy /runs/engine/greedy.tsv --engine-forced /runs/engine/forced.tsv --tol-logprob 0.10

Pass (exit 0) only when all of these hold:
  1. The engine's greedy token ids equal the reference's, position for position (all rows).
  2. With the reference tokens teacher-forced, the engine's argmax equals the reference argmax at every position.
  3. With the reference tokens teacher-forced, |engine logprob - reference logprob| of the reference token is at most --tol-logprob
     at every position.
Exit 1 names the first failing position, both logprobs and the reference top-2 gap there.
"""

import argparse
import csv
import sys


def read_tsv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        {"index": int(r["index"]), "token_id": int(r["token_id"]), "logprob": float(r["logprob"]),
         "top_id": int(r["top_id"]), "top_logprob": float(r["top_logprob"])}
        for r in rows
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--engine-greedy", required=True)
    p.add_argument("--engine-forced", required=True)
    p.add_argument("--tol-logprob", type=float, default=0.10)
    a = p.parse_args()

    ref = read_tsv(a.ref)
    greedy = read_tsv(a.engine_greedy)
    forced = read_tsv(a.engine_forced)

    if len(greedy) != len(ref) or len(forced) != len(ref):
        print(f"FAIL length: reference {len(ref)}, greedy {len(greedy)}, forced {len(forced)}")
        return 1
    for r, g in zip(ref, greedy):
        if r["token_id"] != g["token_id"]:
            print(f"FAIL greedy token at position {r['index']}: reference {r['token_id']}, engine {g['token_id']}")
            return 1

    worst = 0.0
    for r, f in zip(ref, forced):
        if r["token_id"] != f["token_id"]:
            print(f"FAIL forced token at position {r['index']}: the teacher-forced ids differ from the reference")
            return 1
        if f["top_id"] != r["top_id"]:
            gap = r["top_logprob"] - r["logprob"]
            print(f"FAIL argmax at position {r['index']}: reference {r['top_id']} (gap to token {gap:.4f}), engine {f['top_id']}")
            return 1
        diff = abs(f["logprob"] - r["logprob"])
        worst = max(worst, diff)
        if diff > a.tol_logprob:
            print(f"FAIL logprob at position {r['index']}: reference {r['logprob']:.6f}, engine {f['logprob']:.6f}, |diff| {diff:.4f}")
            return 1

    print(f"PASS: {len(ref)} positions, greedy tokens identical, argmax identical, max |logprob diff| {worst:.5f} (tolerance {a.tol_logprob})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
