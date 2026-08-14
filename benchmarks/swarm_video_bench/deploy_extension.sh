#!/bin/bash
# Deploy the engine into the SwarmUI extension, the step that has now cost two scoreboard-grade measurements.
#
# Both failures were the same shape: someone built the engine, copied the DLLs, and forgot that `Ptx/*.ptx` is a
# SEPARATE set of build artifacts the extension loads from disk at runtime. The result is new C# calling old
# kernels — or, on 2026-08-14, a brand-new kernel whose PTX was simply absent, so every SwarmUI-side number that
# day described code from the previous morning. `bench_ltx25.py` now refuses to run against a stale deployment;
# this script is the other half, making the correct deployment a single command instead of a remembered ritual.
#
#   ./deploy_extension.sh            # build, deploy, restart, verify
#   ./deploy_extension.sh --dry-run  # show what WOULD change, touch nothing
set -uo pipefail

REPO=/home/hartsy/Desktop/HartsyInference
EXT="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend"
BUILD="$REPO/src/HartsyInference.Cli/bin/Release/net8.0"
PTX_SRC="$REPO/src/HartsyInference.Cuda/Ptx"
DRY=false
[ "${1:-}" = "--dry-run" ] && DRY=true

[ -d "$EXT" ] || { echo "FATAL: extension dir not found: $EXT"; exit 1; }

# net8.0 on purpose: the extension targets net8.0 while the CLI runs on net10.0. Deploying net10.0 assemblies
# leaves SwarmUI silently loading whatever it can and is a very confusing failure.
echo ">>> building net8.0 …"
if ! dotnet build "$REPO/src/HartsyInference.Cli" -c Release -f net8.0 2>&1 | tail -3; then
    echo "FATAL: build failed — not deploying"; exit 1
fi

changed_dll=0; changed_ptx=0; missing=0
for f in "$BUILD"/HartsyInference.*.dll; do
    b=$(basename "$f"); d="$EXT/$b"
    if [ ! -f "$d" ]; then missing=$((missing+1)); changed_dll=$((changed_dll+1))
    elif [ "$(md5sum "$f" | cut -d' ' -f1)" != "$(md5sum "$d" | cut -d' ' -f1)" ]; then changed_dll=$((changed_dll+1)); fi
done
for f in "$PTX_SRC"/*.ptx; do
    b=$(basename "$f"); d="$EXT/Ptx/$b"
    if [ ! -f "$d" ]; then missing=$((missing+1)); changed_ptx=$((changed_ptx+1))
    elif [ "$(md5sum "$f" | cut -d' ' -f1)" != "$(md5sum "$d" | cut -d' ' -f1)" ]; then changed_ptx=$((changed_ptx+1)); fi
done
echo ">>> $changed_dll DLL(s) and $changed_ptx PTX file(s) differ ($missing absent from the deployment)"

if $DRY; then echo ">>> --dry-run: nothing changed"; exit 0; fi
if [ "$changed_dll" -eq 0 ] && [ "$changed_ptx" -eq 0 ]; then
    echo ">>> deployment already current; not restarting the service"; exit 0
fi

WAS_UP=0
systemctl --user is-active --quiet swarmui.service && WAS_UP=1
if [ "$WAS_UP" = 1 ]; then echo ">>> stopping swarmui.service …"; systemctl --user stop swarmui.service; sleep 6; fi

# Name every DLL whose bytes actually change. This deploys the WHOLE engine, so it can silently overwrite
# another session's work — on 2026-08-14 it replaced HartsyInference.Audio.dll out from under the MiniMax
# Music 3 session, whose SwarmUI-measured numbers then described someone else's build. Listing the changes is
# what lets whoever is affected be told.
for f in "$BUILD"/HartsyInference.*.dll; do
    b=$(basename "$f"); d="$EXT/$b"
    if [ ! -f "$d" ] || [ "$(md5sum "$f" | cut -d' ' -f1)" != "$(md5sum "$d" | cut -d' ' -f1)" ]; then
        echo "    CHANGED: $b"
    fi
done
echo "    ^ if any of the above are owned by another session, tell them to re-measure."
cp -f "$BUILD"/HartsyInference.*.dll "$EXT"/ || { echo "FATAL: DLL copy failed"; exit 1; }
mkdir -p "$EXT/Ptx"
cp -f "$PTX_SRC"/*.ptx "$EXT/Ptx"/ || { echo "FATAL: PTX copy failed"; exit 1; }
echo ">>> copied $(ls "$BUILD"/HartsyInference.*.dll | wc -l) DLLs and $(ls "$PTX_SRC"/*.ptx | wc -l) PTX files"

if [ "$WAS_UP" = 1 ]; then
    echo ">>> starting swarmui.service …"; systemctl --user start swarmui.service; sleep 20
    systemctl --user is-active --quiet swarmui.service && echo ">>> service active" || { echo "WARN: service not active"; exit 1; }
fi

# Verify with the same check the benchmark uses, so "deployed" means exactly what the benchmark demands.
echo ">>> verifying …"
"$REPO/benchmarks/.bench-venv/bin/python" - <<'PY'
import sys
sys.path.insert(0, "/home/hartsy/Desktop/HartsyInference/benchmarks/swarm_video_bench")
import bench_ltx25 as b
b.assert_extension_matches_build()
PY
