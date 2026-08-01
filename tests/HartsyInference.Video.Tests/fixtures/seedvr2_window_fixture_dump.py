"""Generates the SeedVR2 window-partition fixture by executing ByteDance's own window.py.

Output JSON: list of cases, each {size:[t,h,w], num:[nt,nh,nw], method, slices:[[t0,t1,h0,h1,w0,w1],...]}
Slice order preserved exactly as the reference emits it (w-major, then h, then t).
"""
import json
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("window", sys.argv[1])
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)

# (t, h, w) are PATCHIFIED latent token grids: pixel/8/patch2 spatial, (frames-1)/4+1 temporal.
# Chosen to cover: image (t=1), canonical 720p (45x80), non-divisible ragged cases, tall/wide
# aspect, tiny grids where windows collapse, t>30 temporal clamp, and prime dims.
CASES = [
    (1, 45, 80), (7, 45, 80), (2, 45, 80), (7, 44, 79), (13, 60, 106),
    (7, 90, 160), (1, 90, 160), (5, 17, 23), (7, 23, 17), (31, 45, 80),
    (33, 68, 120), (7, 45, 45), (7, 80, 45), (4, 3, 3), (1, 1, 1),
    (8, 34, 60), (25, 45, 80), (7, 135, 240), (6, 50, 89), (3, 8, 8),
]
NUM = (4, 3, 3)  # config: window ${num_layers} * [(4,3,3)]

out = []
for method_name, fn in [
    ("720pwin_by_size_bysize", window.make_720Pwindows_bysize),
    ("720pswin_by_size_bysize", window.make_shifted_720Pwindows_bysize),
]:
    for size in CASES:
        slices = fn(size, NUM)
        out.append({
            "size": list(size),
            "num": list(NUM),
            "method": method_name,
            "slices": [[s[0].start, s[0].stop, s[1].start, s[1].stop, s[2].start, s[2].stop]
                       for s in slices],
        })

with open(sys.argv[2], "w") as f:
    json.dump(out, f, separators=(",", ":"))
total = sum(len(c["slices"]) for c in out)
print(f"cases={len(out)} total_slices={total}")

# Sanity: every regular partition must tile the grid exactly (each token in exactly one window).
for c in out:
    if c["method"] != "720pwin_by_size_bysize":
        continue
    t, h, w = c["size"]
    covered = set()
    for t0, t1, h0, h1, w0, w1 in c["slices"]:
        for it in range(t0, t1):
            for ih in range(h0, h1):
                for iw in range(w0, w1):
                    key = (it, ih, iw)
                    assert key not in covered, f"overlap in {c['size']}"
                    covered.add(key)
    assert len(covered) == t * h * w, f"coverage gap in {c['size']}: {len(covered)} != {t*h*w}"
print("regular-window tiling verified: no overlap, full coverage")
