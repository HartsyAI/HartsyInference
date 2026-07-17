#!/usr/bin/env python3
"""Edge-band fringe analysis for the Flux-Depth conditioning map A/B.

Reads the Python reference dumps (dump_flux_depth_ab.py) and the engine dumps
(FluxDepthAbTests) and quantifies WHERE the maps differ:
  - global corr / MAE per pair
  - edge-band metrics: Canny edges of the source photo (at gen res), dilated; MAE/corr inside
    the band vs outside
  - overshoot metric: fraction of pixels outside the [local-min, local-max] envelope of the
    nearest-neighbor upsampled net-res map (bicubic ringing / halo detector)
Writes diff heatmap PNGs next to the engine dumps.

Usage:
  python compare_flux_depth_ab.py --ref-dir <flux_depth_reference_tensors> \
      --engine-dir <Output/fluxdepth_redux_ab/depth_csharp> --image bus.png
"""
import argparse
import json
import os

import cv2
import numpy as np


def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def metrics(name, a, b, band=None):
    out = {"pair": name, "corr": corr(a, b), "mae": float(np.abs(a - b).mean()),
           "max_abs": float(np.abs(a - b).max())}
    if band is not None:
        d = np.abs(a - b)
        out["edge_band_mae"] = float(d[band].mean())
        out["edge_band_max"] = float(d[band].max())
        out["off_band_mae"] = float(d[~band].mean())
        out["edge_band_corr"] = corr(a[band], b[band])
    return out


def overshoot_fraction(gen_map, net_map, gen_hw):
    """Pixels of gen_map outside the local [min,max] envelope of the NN-upsampled net map."""
    nn = cv2.resize(net_map, (gen_hw[1], gen_hw[0]), interpolation=cv2.INTER_NEAREST)
    k = np.ones((5, 5), np.uint8)
    lo = cv2.erode(nn, k)
    hi = cv2.dilate(nn, k)
    eps = 1e-3 * (nn.max() - nn.min())
    return float(((gen_map < lo - eps) | (gen_map > hi + eps)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.ref_dir, "meta.json")))
    gh, gw = meta["gen_h"], meta["gen_w"]
    nh, nw = meta["net_h"], meta["net_w"]
    sh, sw = meta["src_h"], meta["src_w"]

    eng_gen = load(os.path.join(args.engine_dir, "engine_gen_flux.bin"), (gh, gw))
    eng_net = load(os.path.join(args.engine_dir, "engine_net_depth.bin"), (nh, nw))
    eng_src = load(os.path.join(args.engine_dir, "engine_src_unit.bin"), (sh, sw))
    ref_bil = load(os.path.join(args.ref_dir, "gen_bilinear.bin"), (gh, gw))
    ref_bfl = load(os.path.join(args.ref_dir, "gen_bfl.bin"), (gh, gw))
    ref_net = load(os.path.join(args.ref_dir, "net_depth.bin"), (nh, nw))
    ref_src = load(os.path.join(args.ref_dir, "src_depth.bin"), (sh, sw))
    ref_src = (ref_src - ref_src.min()) / (ref_src.max() - ref_src.min())
    comfy = load(os.path.join(args.ref_dir, "comfy_map.bin"), (gh, gw))

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2.resize(bgr, (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    band = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2) > 0
    print(f"edge band covers {band.mean() * 100:.1f}% of pixels")

    # min-max normalize both gen maps identically for comfy comparison (comfy map is min-max)
    def mm(x):
        return (x - x.min()) / (x.max() - x.min())

    results = [
        metrics("engine_net vs ref_net (raw model)", eng_net, ref_net),
        metrics("engine_src_unit vs ref_src_unit", eng_src, ref_src),
        metrics("engine_gen vs ref_bilinear (kernel-matched)", eng_gen, ref_bil, band),
        metrics("engine_gen vs ref_bfl (bicubic-antialias)", eng_gen, ref_bfl, band),
        metrics("engine_gen(minmax) vs comfy_map", mm(eng_gen), comfy, band),
        metrics("ref_bfl vs ref_bilinear (kernel effect only)", ref_bfl, ref_bil, band),
    ]
    for r in results:
        print(json.dumps(r))

    print(json.dumps({
        "overshoot_frac_engine_gen": overshoot_fraction(eng_gen, eng_net / eng_net.max(), (gh, gw)),
        "overshoot_frac_ref_bfl": overshoot_fraction(ref_bfl, ref_net / ref_net.max(), (gh, gw)),
        "overshoot_frac_ref_bilinear": overshoot_fraction(ref_bil, ref_net / ref_net.max(), (gh, gw)),
        "overshoot_frac_comfy": overshoot_fraction(comfy, mm(ref_net), (gh, gw)),
    }, indent=2))

    for name, a, b in [("diff_engine_vs_bfl", eng_gen, ref_bfl),
                       ("diff_engine_vs_comfy", mm(eng_gen), comfy)]:
        d = np.abs(a - b)
        heat = cv2.applyColorMap((np.clip(d / max(d.max(), 1e-6), 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(args.engine_dir, name + ".png"), heat)
    cv2.imwrite(os.path.join(args.engine_dir, "engine_gen_flux.png"),
                (np.clip(eng_gen, 0, 1) * 255).round().astype(np.uint8))
    print("heatmaps written to", args.engine_dir)


if __name__ == "__main__":
    main()
