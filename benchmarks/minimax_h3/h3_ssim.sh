#!/bin/bash
# SSIM/PSNR between two MiniMax-H3 frame directories (lossless PNG in, so this measures the pixels
# the pipeline actually produced, not codec artifacts).
#
#   ./h3_ssim.sh <dir_a> <dir_b>
#
# Both dirs are the CLI's per-run output folder containing frame_%04d.png.
#
# WHAT TO COMPARE AGAINST. Not ComfyUI: it seeds its own noise (its bench workflow uses seed 42) and
# its RNG does not match ours, so two "correct" runs differ completely and SSIM lands near 0.3, which
# gates nothing. Compare against a GOLD run of this same pipeline at the same seed with the precision
# shortcuts disabled -- see h3_gold.sh. That reference is strictly more faithful than any optimized
# variant, so a change that improves fidelity moves SSIM UP rather than tripping a regression gate.
set -u

A=${1:?usage: h3_ssim.sh <dir_a> <dir_b>}
B=${2:?usage: h3_ssim.sh <dir_a> <dir_b>}

for d in "$A" "$B"; do
    [ -d "$d" ] || { echo "FATAL: not a directory: $d"; exit 1; }
    n=$(find "$d" -maxdepth 1 -name 'frame_*.png' | wc -l)
    [ "$n" -gt 0 ] || { echo "FATAL: no frame_*.png in $d"; exit 1; }
    echo "$d: $n frames"
done

NA=$(find "$A" -maxdepth 1 -name 'frame_*.png' | wc -l)
NB=$(find "$B" -maxdepth 1 -name 'frame_*.png' | wc -l)
[ "$NA" = "$NB" ] || { echo "FATAL: frame count differs ($NA vs $NB) — not comparable."; exit 1; }

python3 - "$A" "$B" <<'PY'
import re, subprocess, sys

a, b = sys.argv[1], sys.argv[2]
proc = subprocess.run(
    ["ffmpeg", "-hide_banner", "-loglevel", "info",
     "-f", "image2", "-i", f"{a}/frame_%04d.png",
     "-f", "image2", "-i", f"{b}/frame_%04d.png",
     "-lavfi", "[0:v][1:v]ssim;[0:v][1:v]psnr", "-f", "null", "-"],
    capture_output=True, text=True)
out = proc.stderr

m = re.search(r"SSIM.*All:([0-9.]+)", out)
p = re.search(r"PSNR.*average:([0-9.a-z]+)", out)
if not m:
    print("FAILED to parse ffmpeg SSIM output:")
    print(out[-2000:])
    sys.exit(1)
ssim = float(m.group(1))
print(f"\nSSIM  {ssim:.6f}")
if p:
    print(f"PSNR  {p.group(1)} dB")
print("identical" if ssim >= 0.99999 else
      "excellent (>=0.99)" if ssim >= 0.99 else
      "acceptable (>=0.98)" if ssim >= 0.98 else
      "DEGRADED (<0.98)")
PY
