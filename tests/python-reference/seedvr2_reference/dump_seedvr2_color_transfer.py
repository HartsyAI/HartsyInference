"""Dumps ComfyUI's SeedVR2 colour transfers (lab / adain) on deterministic planes, as the fixture for
SeedVr2ColorTransferTests.

Reads comfy/ldm/seedvr/color_fix.py from a ComfyUI checkout (the module imports only torch and its own constants),
so the fixture is the reference's own output, not a re-derivation of it.

Usage:
    python3 dump_seedvr2_color_transfer.py --comfy-root <ComfyUI dir> --out <fixture.json>
"""

import argparse
import importlib.util
import json
import math
import sys
import types

import torch


def load_color_fix(comfy_root):
    seedvr_dir = f"{comfy_root}/comfy/ldm/seedvr"
    # color_fix.py does `from comfy.ldm.seedvr.constants import ...`; load both modules without importing the
    # rest of ComfyUI (which pulls in the whole model stack).
    for name, path in [("comfy", None), ("comfy.ldm", None), ("comfy.ldm.seedvr", None)]:
        sys.modules.setdefault(name, types.ModuleType(name)).__path__ = [
            f"{comfy_root}/" + name.replace(".", "/")
        ]
    spec = importlib.util.spec_from_file_location("comfy.ldm.seedvr.constants", f"{seedvr_dir}/constants.py")
    constants = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = constants
    spec.loader.exec_module(constants)
    spec = importlib.util.spec_from_file_location("comfy.ldm.seedvr.color_fix", f"{seedvr_dir}/color_fix.py")
    color_fix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(color_fix)
    return color_fix


def planes(height, width, phase):
    """Deterministic, tie-free RGB planes in [-1,1]; irrational frequencies keep sort order unambiguous."""
    model = torch.zeros(1, 3, height, width)
    reference = torch.zeros(1, 3, height, width)
    for c in range(3):
        for y in range(height):
            for x in range(width):
                model[0, c, y, x] = 0.9 * math.sin(0.37 * x + 0.23 * y + 1.7 * c + phase)
                reference[0, c, y, x] = 0.8 * math.cos(0.21 * x - 0.41 * y + 0.9 * c) + 0.1 * math.sin(0.05 * x * y)
    return model, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    color_fix = load_color_fix(args.comfy_root)

    cases = []
    for height, width, phase in [(12, 20, 0.0), (16, 24, 0.6)]:
        model, reference = planes(height, width, phase)
        lab = color_fix.lab_color_transfer(model.clone(), reference.clone())
        adain = color_fix.adain_color_transfer(model.clone(), reference.clone())
        cases.append({
            "height": height,
            "width": width,
            "phase": phase,
            "model": model.flatten().tolist(),
            "reference": reference.flatten().tolist(),
            "lab": lab.flatten().tolist(),
            "adain": adain.flatten().tolist(),
        })
    with open(args.out, "w") as f:
        json.dump({"source": "comfy/ldm/seedvr/color_fix.py", "cases": cases}, f)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
