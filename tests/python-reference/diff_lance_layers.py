#!/usr/bin/env python3
"""Layer-by-layer diff between the Lance engine dump (LANCE_DEBUG_DIR) and the reference dump
(dump_lance_reference.py). Prints avg/max abs error + correlation per stage, flagging the first
stage with avg > 1e-3 — the standard Lance parity workflow:

  1. python dump_lance_reference.py --ckpt-dir ... --ref-repo ... --out <ref>
  2. LANCE_3B_DIR=... LANCE_PARITY_DIR=<ref> dotnet test --filter LanceRealWeightParityTests
     (the test writes its own dumps to $TMP/lance_parity_engine_<pid>)
  3. python diff_lance_layers.py <engine dump dir> <ref dump dir>
"""
import os
import sys

import numpy as np


def load(path):
    return np.fromfile(path, dtype="<f4")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    eng_dir, ref_dir = sys.argv[1], sys.argv[2]

    names = ["layers/packed_in.bin"]
    i = 0
    while os.path.exists(os.path.join(ref_dir, "layers", f"layers_{i}.bin")):
        names.append(f"layers/layers_{i}.bin")
        i += 1
    names.append("output_velocity.bin")

    first_bad = None
    for name in names:
        e_path, r_path = os.path.join(eng_dir, name.replace("layers_", "layers_")), os.path.join(ref_dir, name)
        e_path = os.path.join(eng_dir, name) if os.path.exists(os.path.join(eng_dir, name)) else \
            os.path.join(eng_dir, name.replace("layers_", "layers_"))
        if not os.path.exists(e_path):
            # engine names layers via LanceDebugDump: layers.{i} → layers_{i}.bin (same), skip if absent
            print(f"{name:28s}  MISSING engine dump")
            continue
        if not os.path.exists(r_path):
            print(f"{name:28s}  MISSING reference dump")
            continue
        a, b = load(e_path), load(r_path)
        if a.shape != b.shape:
            print(f"{name:28s}  SHAPE MISMATCH {a.shape} vs {b.shape}")
            first_bad = first_bad or name
            continue
        err = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        avg, mx = float(err.mean()), float(err.max())
        flag = ""
        if avg > 1e-3 and first_bad is None:
            first_bad = name
            flag = "   <-- FIRST DIVERGENCE"
        print(f"{name:28s}  avg={avg:.3e}  max={mx:.3e}  corr={corr:.6f}{flag}")

    if first_bad:
        print(f"\nFirst stage with avg err > 1e-3: {first_bad}")
        sys.exit(2)
    print("\nAll dumped stages within tolerance.")


if __name__ == "__main__":
    main()
