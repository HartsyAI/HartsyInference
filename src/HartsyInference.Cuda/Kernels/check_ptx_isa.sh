#!/usr/bin/env bash
# Fails when any PTX shipped in ../Ptx declares a PTX ISA above the fleet driver's JIT ceiling (9.0).
set -euo pipefail
dir="$(cd "$(dirname "$0")/../Ptx" && pwd)"
bad=0
total=0
for f in "$dir"/*.ptx; do
    total=$((total + 1))
    v=$(head -20 "$f" | sed -n 's/^\.version \([0-9.]*\)$/\1/p' | head -1)
    if [[ -z "$v" || "$(printf '%s\n' "$v" 9.0 | sort -V | tail -1)" != "9.0" ]]; then
        echo "BAD ISA ${v:-none}: $(basename "$f")"
        bad=$((bad + 1))
    fi
done
if (( bad > 0 )); then
    echo "$bad of $total PTX file(s) declare an ISA above 9.0 or none" >&2
    exit 1
fi
echo "all $total PTX files declare ISA 9.0 or lower"
