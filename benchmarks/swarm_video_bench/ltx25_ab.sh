#!/bin/bash
# Interleaved A/B for LTX-2.5 step time. Wraps ltx25_bench.sh.
#
#   ./ltx25_ab.sh VAR OFF_VALUE ON_VALUE [REPS] [STEPS]
#   ./ltx25_ab.sh HARTSY_GROUPED_LINEAR 0 1 4
#
# WHY THIS EXISTS: the same build, same seed, back to back, spreads ~25 ms/step on this box (1435.2 vs 1459.6
# measured 2026-08-13) while steps WITHIN one run vary by only ~5 ms. So the variance is per-process, not
# thermal, and a single run per arm cannot resolve anything under ~50 ms. Several deltas in VIDEO.md were
# accepted on one run per arm and are not established. Use this instead.
#
# Arms alternate (A B A B ...) so any drift over the campaign hits both equally. Reports each arm's mean,
# median and range, and the difference against the pooled spread.
set -u

VAR=${1:?usage: ltx25_ab.sh VAR OFF_VALUE ON_VALUE [REPS] [STEPS]}
OFF=${2:?}
ON=${3:?}
REPS=${4:-4}
STEPS=${5:-30}

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${LTX25_BENCH_OUT:-/tmp/ltx25_bench}

# Hold swarmui down for the WHOLE campaign; otherwise every rep pays its own stop/start cycle (~40 s each)
# and the service churn is itself a source of between-run state difference.
SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service && SWARM_WAS_UP=1
restore() { [ "$SWARM_WAS_UP" = "1" ] && systemctl --user start swarmui.service >/dev/null 2>&1; }
trap restore EXIT INT TERM
if [ "$SWARM_WAS_UP" = "1" ]; then
    echo "stopping swarmui.service for the whole campaign..."
    systemctl --user stop swarmui.service
    sleep 8
fi

stepmean() {
    awk '/ltx2-phase\] step / { n++; if (n>1) { gsub(/ms.*/,"",$0); sub(/.*: /,"",$0); s+=$0; c++ } }
         END { if (c) printf "%.1f", s/c; else printf "NaN" }' "$1"
}

A_VALS=(); B_VALS=()
for rep in $(seq 1 "$REPS"); do
    for arm in off on; do
        [ "$arm" = off ] && val=$OFF || val=$ON
        label=ab_${VAR}_${val}_${rep}
        env "$VAR=$val" "$HERE/ltx25_bench.sh" "$STEPS" "$label" 1 \
            "${LTX25_AB_FRAMES:-97}" "${LTX25_AB_WIDTH:-768}" "${LTX25_AB_HEIGHT:-512}" >/dev/null 2>&1
        ms=$(stepmean "$OUT/${label}_s${STEPS}_seed1.log")
        echo "  rep$rep  $VAR=$val  $ms ms/step"
        [ "$arm" = off ] && A_VALS+=("$ms") || B_VALS+=("$ms")
        rm -rf "$OUT/frames_$label"
    done
done

# Prints "<tag> n=.. mean .. median .. range .." to stdout and the bare mean to $2.
summarize() {
    local tag=$1 meanfile=$2; shift 2
    printf '%s\n' "$@" | sort -n | awk -v tag="$tag" -v mf="$meanfile" '
        { v[NR]=$1; s+=$1 }
        END { n=NR; mean=s/n; med=(n%2)?v[(n+1)/2]:(v[n/2]+v[n/2+1])/2;
              printf "%-30s n=%d  mean %.1f  median %.1f  range %.1f-%.1f (spread %.1f)\n",
                     tag, n, mean, med, v[1], v[n], v[n]-v[1];
              printf "%.4f\n", mean > mf }'
}

echo
summarize "$VAR=$OFF" /tmp/.ab_ma "${A_VALS[@]}"
summarize "$VAR=$ON"  /tmp/.ab_mb "${B_VALS[@]}"
awk 'NR==FNR { a=$1; next } { b=$1 }
     END { printf "\ndelta (on - off): %+.1f ms/step\n", b-a }' /tmp/.ab_ma /tmp/.ab_mb
echo "A delta smaller than the per-arm spread above is NOT established — raise REPS or drop the change."
