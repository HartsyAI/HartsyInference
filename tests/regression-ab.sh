#!/usr/bin/env bash
# Regression gate: the same generations on two builds of the engine, interleaved, and a report that says whether
# anything moved — in the pixels, in the tokens, in the time per step, in VRAM.
#
#   tests/regression-ab.sh                                       # origin/main vs this tree, the core cases, cuda
#   tests/regression-ab.sh --expect bounded:0.98 --backend cuda,vulkan
#   tests/regression-ab.sh --filter sd15,krea2 --reps 1           # a quick look while iterating
#   tests/regression-ab.sh --no-base --head-set numerics.fp4Native=true   # one build, a knob flipped
#
# Both arms are BUILT here, PTX and SPIR-V included: a gate that measures whatever DLL is on disk measures the wrong
# tree and cannot tell — new C# calling old kernels has invalidated two rounds of numbers before. The base arm's
# generations are cached by commit under the output root, so a stable main is generated once.
#
# Speed is the median over the warm seeds (every seed after the first) of the per-step ms the CLI prints, the wall
# clock, and peak VRAM from one nvidia-smi sampler. Quality is the head artifact against the base artifact at the
# same seed: SSIM over decoded pixels (PNG bytes are not stable; pixels are) and the raw-RGB digest
# migration-baseline.sh uses; text at temperature 0 compares the token stream. `--expect identical` (a refactor)
# needs every digest to match; `--expect bounded:<ssim>` (a numerics change) needs the minimum SSIM to clear the
# floor. Either way a step time more than --speed-tolerance percent slower fails the case.
#
# Arms alternate seed by seed (A s0, B s0, A s1, B s1, …) so thermal drift and page-cache state land on both equally.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=regression-cases.sh
. "$REPO/tests/regression-cases.sh"

OUT="${HARTSY_REGRESSION_OUT:-$HOME/Desktop/regression-ab}"
MODELS="${HARTSY_MODELS_ROOT:-/mnt/model-storage/Models}"
BASE_REF="origin/main"
HEAD_REF="$REPO"
BACKENDS="cuda"
FILTER=""
TAG="core"
EXPECT="identical"
REPS=3
GPU_NAME="RTX 4090"
GPU_INDEX=""
HEAD_SET=()
BASE_SET=()
NO_BASE=0
FRESH=0
SPEED_TOL=5
TIMEOUT=1800

while [ $# -gt 0 ]; do
    case "$1" in
        --base)     BASE_REF="$2"; shift 2 ;;
        --head)     HEAD_REF="$2"; shift 2 ;;
        --backend)  BACKENDS="$2"; shift 2 ;;
        --filter)   FILTER="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --expect)   EXPECT="$2"; shift 2 ;;
        --reps)     REPS="$2"; shift 2 ;;
        --gpu)      GPU_INDEX="$2"; shift 2 ;;
        --gpu-name) GPU_NAME="$2"; shift 2 ;;   # empty = whatever --gpu names
        --head-set) HEAD_SET+=(--set "$2"); shift 2 ;;
        --base-set) BASE_SET+=(--set "$2"); shift 2 ;;
        --no-base)  NO_BASE=1; shift ;;
        --fresh)    FRESH=1; shift ;;
        --speed-tolerance) SPEED_TOL="$2"; shift 2 ;;
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

log() { printf '[ab] %s\n' "$*" >&2; }
die() { log "$*"; exit 2; }
mkdir -p "$OUT/runs" "$OUT/reports"

SSIM_FLOOR=""
case "$EXPECT" in
    identical) ;;
    bounded:*) SSIM_FLOOR="${EXPECT#bounded:}" ;;
    *) die "--expect must be identical or bounded:<ssim>" ;;
esac
for tool in dotnet nvidia-smi ffmpeg md5sum git; do
    command -v "$tool" >/dev/null || die "$tool is required"
done

# ── GPU ──────────────────────────────────────────────────────────────────────────────────────────────────────
# CUDA enumerates fastest-first, nvidia-smi by PCI bus; PCI_BUS_ID order makes the two agree, so the ordinal the
# CLI is given and the index the sampler watches are the same card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "$GPU_INDEX" ]; then
    GPU_INDEX="$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v n="$GPU_NAME" '$2 ~ n {print $1; exit}')"
fi
[ -z "$GPU_INDEX" ] && [ -z "$GPU_NAME" ] && GPU_INDEX=0
[ -n "$GPU_INDEX" ] || die "no GPU named '$GPU_NAME' (pass --gpu <nvidia-smi index>)"
GPU_UUID="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"
GPU_LABEL="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name,driver_version --format=csv,noheader)"

# ── arms ─────────────────────────────────────────────────────────────────────────────────────────────────────
# A directory is measured as it stands (dirty trees are labelled and never cached); a ref becomes a detached
# worktree under the output root so the base is built from exactly that commit.
resolve_arm() {
    local ref="$1"
    if [ -d "$ref" ]; then
        local sha; sha="$(git -C "$ref" rev-parse --short=8 HEAD)"
        [ -n "$(git -C "$ref" status --porcelain)" ] && sha="$sha+dirty"
        printf '%s\t%s\n' "$(cd "$ref" && pwd)" "$sha"
    else
        local full; full="$(git -C "$REPO" rev-parse "$ref^{commit}" 2>/dev/null)" || die "cannot resolve $ref"
        local sha="${full:0:8}"
        local src="$OUT/src-$sha"
        [ -d "$src" ] || git -C "$REPO" worktree add -q --detach "$src" "$full" || die "worktree add failed for $ref"
        printf '%s\t%s\n' "$src" "$sha"
    fi
}

build_arm() {
    local src="$1" label="$2"
    local dir="$OUT/build-$label"
    if [ "$FRESH" = 0 ] && [[ "$label" != *+dirty ]] && [ -f "$dir/HartsyInference.Cli.dll" ] && [ -d "$dir/Ptx" ]; then
        log "reusing build $label"
    else
        log "building $label from $src ..."
        rm -rf "$dir"
        if ! dotnet build "$src/src/HartsyInference.Cli" -c Release -f net10.0 -o "$dir" --nologo -v q >"$OUT/build-$label.log" 2>&1; then
            tail -30 "$OUT/build-$label.log" >&2
            die "build of $label failed"
        fi
        [ -d "$dir/Ptx" ] || die "build $label produced no Ptx/ — the CUDA kernels would be whatever was on disk"
    fi
    printf '%s\n' "$dir"
}

IFS=$'\t' read -r HEAD_SRC HEAD_LABEL < <(resolve_arm "$HEAD_REF")
HEAD_BUILD="$(build_arm "$HEAD_SRC" "$HEAD_LABEL")" || exit 2
if [ "$NO_BASE" = 0 ]; then
    IFS=$'\t' read -r BASE_SRC BASE_LABEL < <(resolve_arm "$BASE_REF")
    BASE_BUILD="$(build_arm "$BASE_SRC" "$BASE_LABEL")" || exit 2
else
    BASE_LABEL="(none)"; BASE_BUILD=""
fi
# A knob makes a different arm of the same commit; the label carries a hash of the knobs so two arms of one build
# (off vs on) never share a run directory.
knob_label() { printf '%s' "$*" | md5sum | cut -c1-6; }
[ ${#HEAD_SET[@]} -gt 0 ] && HEAD_LABEL="$HEAD_LABEL+set-$(knob_label "${HEAD_SET[@]}")"
[ ${#BASE_SET[@]} -gt 0 ] && BASE_LABEL="$BASE_LABEL+set-$(knob_label "${BASE_SET[@]}")"

# ── the card must be ours for the duration ───────────────────────────────────────────────────────────────────
SWARM_WAS_UP=0
systemctl --user is-active --quiet swarmui.service 2>/dev/null && SWARM_WAS_UP=1
restore() { [ "$SWARM_WAS_UP" = 1 ] && systemctl --user start swarmui.service >/dev/null 2>&1; }
trap restore EXIT INT TERM
if [ "$SWARM_WAS_UP" = 1 ]; then
    log "stopping swarmui.service for the run (restored on exit)"
    systemctl --user stop swarmui.service
    sleep 5
fi
tenants="$(nvidia-smi -i "$GPU_UUID" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -vE 'Xorg|rustdesk|cinnamon|gnome-shell|chrome')"
[ -n "$tenants" ] && log "WARNING: other processes hold the card, numbers may not be comparable:"$'\n'"$tenants"

# ── one generation ───────────────────────────────────────────────────────────────────────────────────────────
digest_of() {
    local value
    value="$(md5sum "$1" | cut -d' ' -f1)" || return 1
    [[ "$value" =~ ^[0-9a-f]{32}$ ]] || return 1
    printf '%s' "$value"
}

# run_one <build> <label> <case> <ckpt> <cmd> <positional> <args> <seed> <backend> <dir> [--set k=v ...]
# Writes <dir>/metrics.env. A completed run of a clean commit is left alone.
run_one() {
    local build="$1" label="$2" id="$3" ckpt="$4" cmd="$5" positional="$6" args="$7" seed="$8" backend="$9" dir="${10}"
    shift 10
    local sets=("$@")
    # A clean commit's build is deterministic, so its completed runs are reused whichever arm it is; a dirty tree
    # is regenerated every time, and --fresh regenerates everything.
    if [ "$FRESH" = 0 ] && [[ "$label" != *+dirty* ]] && grep -qx 'status=ok' "$dir/metrics.env" 2>/dev/null; then
        return 0
    fi
    rm -rf "$dir"
    mkdir -p "$dir/out"
    local sel="$backend"
    [ "$backend" = cuda ] && sel="cuda:$GPU_INDEX"
    nvidia-smi -i "$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$dir/vram.log" 2>/dev/null &
    local sampler=$!
    local t0 t1; t0="$(date +%s.%N)"
    # shellcheck disable=SC2086
    timeout "$TIMEOUT" dotnet "$build/HartsyInference.Cli.dll" "$cmd" "$positional" --model-path "$MODELS/$ckpt" \
        -b "$sel" -o "$dir/out" --seed "$seed" $args "${sets[@]}" >"$dir/run.log" 2>&1 </dev/null
    local rc=$?
    t1="$(date +%s.%N)"
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local wall; wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')"
    local peak; peak="$(sort -n "$dir/vram.log" 2>/dev/null | tail -1)"
    # ConsoleStepProgress rewrites one line with \r; -o pulls each "[n/N] ms" tick out. Tick 1 carries first-touch
    # costs, so the median is over ticks 2..N — for text that is one tick per token, i.e. decode ms/token.
    local steps step_n step_med
    steps="$(grep -aoE '\[[0-9]+/[0-9]+\] [0-9]+ ms' "$dir/run.log" | awk -F'[][/ ]+' '$2 + 0 >= 2 { print $4 }' | sort -n)"
    step_n="$(printf '%s\n' "$steps" | grep -c .)"
    step_med="$(printf '%s\n' "$steps" | awk 'NF { a[++n] = $1 } END { print n ? a[int((n + 1) / 2)] : "-" }')"
    # Text prints no ticks; its decode rate (decode_tok_s, streamed pieces after the first) becomes ms per token.
    if [ "$step_n" = 0 ]; then
        local rate; rate="$(grep -aoE 'decode_tok_s[[:space:]]+[0-9.]+' "$dir/run.log" | tail -1 | awk '{ print $2 }')"
        if [ -n "$rate" ]; then step_n=1; step_med="$(awk -v r="$rate" 'BEGIN { printf "%.2f", 1000 / r }')"; fi
    fi
    local status="ok" digest="-" artifact=""
    artifact="$(find "$dir/out" -type f \( -name '*.png' -o -name '*.txt' \) | sort | head -1)"
    if [ "$rc" -ne 0 ] || [ -z "$artifact" ]; then
        status="CRASH"
    else
        case "$artifact" in
            *.png)
                if ffmpeg -nostdin -loglevel error -i "$artifact" -f rawvideo -pix_fmt rgb24 -y "$dir/pixels.rgb24" >>"$dir/run.log" 2>&1 \
                    && [ -s "$dir/pixels.rgb24" ]; then
                    digest="$(digest_of "$dir/pixels.rgb24")" || status="CRASH"
                    rm -f "$dir/pixels.rgb24"
                else
                    status="CRASH"
                fi ;;
            *) [ -s "$artifact" ] && digest="$(digest_of "$artifact")" || status="CRASH" ;;
        esac
    fi
    printf 'status=%s\nwall=%s\nstep_n=%s\nstep_med=%s\npeak=%s\ndigest=%s\nartifact=%s\n' \
        "$status" "$wall" "$step_n" "$step_med" "${peak:-0}" "$digest" "$artifact" >"$dir/metrics.env"
    [ "$status" = CRASH ] && log "  CRASH: $id $backend seed $seed on $label — see $dir/run.log"
    return 0
}

# ── the run ──────────────────────────────────────────────────────────────────────────────────────────────────
STAMP="$(date -u +%Y%m%d-%H%M%S)"
REPORT_DIR="$OUT/reports/$STAMP-${BASE_LABEL//\//_}-vs-${HEAD_LABEL//\//_}"
mkdir -p "$REPORT_DIR/pairs"
ROWS="$REPORT_DIR/rows.tsv"
: >"$ROWS"
SEED0=42
FAILURES=0
CASES_RUN=0

median() { sort -n | awk 'NF { a[++n] = $1 } END { print n ? a[int((n + 1) / 2)] : "-" }'; }
pct() {   # pct <base> <head> -> signed percent, or -
    awk -v a="$1" -v b="$2" 'BEGIN { if (a == "-" || b == "-" || a + 0 == 0) print "-"; else printf "%+.1f", (b - a) / a * 100 }'
}

for backend in ${BACKENDS//,/ }; do
    # A backend other than cuda runs the cases tagged with its name; cuda runs the whole selection.
    backend_tag=""; [ "$backend" != cuda ] && backend_tag="$backend"
    while IFS=$'\t' read -r id ckpt spec; do
        [ -z "$id" ] && continue
        if [ -n "$FILTER" ] && [[ ",$FILTER," != *",$id,"* ]]; then continue; fi
        IFS='|' read -r cmd positional args <<<"$spec"
        if [ ! -e "$MODELS/$ckpt" ]; then
            log "untested: $id — $MODELS/$ckpt is missing"
            printf '%s\t%s\tuntested\n' "$id" "$backend" >>"$ROWS"
            FAILURES=$((FAILURES + 1))
            continue
        fi
        CASES_RUN=$((CASES_RUN + 1))
        log "── $id on $backend"
        for i in $(seq 0 "$REPS"); do
            seed=$((SEED0 + i))
            if [ "$NO_BASE" = 0 ]; then
                run_one "$BASE_BUILD" "$BASE_LABEL" "$id" "$ckpt" "$cmd" "$positional" "$args" "$seed" "$backend" \
                    "$OUT/runs/$BASE_LABEL/$backend/$id/s$seed" "${BASE_SET[@]}"
            fi
            run_one "$HEAD_BUILD" "$HEAD_LABEL" "$id" "$ckpt" "$cmd" "$positional" "$args" "$seed" "$backend" \
                "$OUT/runs/$HEAD_LABEL/$backend/$id/s$seed" "${HEAD_SET[@]}"
        done

        # ── compare ──
        verdict="PASS"; notes=""
        min_ssim="-"; digests="n/a"
        bw=""; hw=""; bs=""; hs=""; bv=""; hv=""
        h_crashes=0; b_crashes=0; changed=0
        for i in $(seq 0 "$REPS"); do
            seed=$((SEED0 + i))
            hd="$OUT/runs/$HEAD_LABEL/$backend/$id/s$seed"
            # shellcheck disable=SC1091
            . "$hd/metrics.env"; h_status=$status; h_wall=$wall; h_step=$step_med; h_peak=$peak; h_digest=$digest; h_art=$artifact
            [ "$h_status" = CRASH ] && h_crashes=$((h_crashes + 1))
            if [ "$i" -ge 1 ] && [ "$h_status" != CRASH ]; then hw+="$h_wall"$'\n'; hs+="$h_step"$'\n'; hv+="$h_peak"$'\n'; fi
            [ "$NO_BASE" = 1 ] && continue
            bd="$OUT/runs/$BASE_LABEL/$backend/$id/s$seed"
            # shellcheck disable=SC1091
            . "$bd/metrics.env"; b_status=$status; b_wall=$wall; b_step=$step_med; b_peak=$peak; b_digest=$digest; b_art=$artifact
            [ "$b_status" = CRASH ] && b_crashes=$((b_crashes + 1))
            if [ "$i" -ge 1 ] && [ "$b_status" != CRASH ]; then bw+="$b_wall"$'\n'; bs+="$b_step"$'\n'; bv+="$b_peak"$'\n'; fi
            [ "$h_status" = CRASH ] || [ "$b_status" = CRASH ] && continue
            if [ "$b_digest" = "$h_digest" ]; then [ "$digests" != DIFFER ] && digests="equal"; else changed=1; digests="DIFFER"; fi
            case "$h_art" in
                *.png)
                    ssim="$(ffmpeg -nostdin -i "$b_art" -i "$h_art" -lavfi ssim -f null - 2>&1 | grep -oE 'All:[0-9.]+' | cut -d: -f2)"
                    [ -z "$ssim" ] && ssim="-"
                    if [ "$ssim" != "-" ] && { [ "$min_ssim" = "-" ] || awk -v a="$ssim" -v b="$min_ssim" 'BEGIN { exit !(a < b) }'; }; then min_ssim="$ssim"; fi
                    # before | after | abs-diff (black = identical), one montage per seed pair
                    ffmpeg -nostdin -loglevel error -y -i "$b_art" -i "$h_art" \
                        -filter_complex "[0:v][1:v]blend=all_mode=difference[d];[0:v][1:v][d]hstack=inputs=3" \
                        "$REPORT_DIR/pairs/${id}_${backend}_s${seed}.png" 2>/dev/null ;;
                *)
                    if [ "$b_digest" != "$h_digest" ]; then diff -u "$b_art" "$h_art" >"$REPORT_DIR/pairs/${id}_${backend}_s${seed}.diff" 2>/dev/null; fi ;;
            esac
        done
        h_wall_m="$(printf '%s' "$hw" | median)"; h_step_m="$(printf '%s' "$hs" | median)"; h_peak_m="$(printf '%s' "$hv" | median)"
        if [ "$NO_BASE" = 1 ]; then
            [ "$h_crashes" -gt 0 ] && verdict="FAIL crash"
            printf '%s\t%s\t%s\t-\t%s\t-\t-\t%s\t-\t-\t%s\t-\t-\t%s\n' "$id" "$backend" "$verdict" "$h_wall_m" "$h_step_m" "$h_peak_m" "$notes" >>"$ROWS"
            [ "$verdict" != PASS ] && FAILURES=$((FAILURES + 1))
            continue
        fi
        b_wall_m="$(printf '%s' "$bw" | median)"; b_step_m="$(printf '%s' "$bs" | median)"; b_peak_m="$(printf '%s' "$bv" | median)"
        d_wall="$(pct "$b_wall_m" "$h_wall_m")"; d_step="$(pct "$b_step_m" "$h_step_m")"; d_peak="$(pct "$b_peak_m" "$h_peak_m")"
        if [ "$h_crashes" -gt 0 ] && [ "$b_crashes" -eq "$h_crashes" ] && [ "$h_crashes" -eq $((REPS + 1)) ]; then
            # Not a regression: the case does not run on this backend on either arm. Reported, not failed.
            verdict="CRASH-both"; notes+="crashes on base too (pre-existing); "
        elif [ "$h_crashes" -gt 0 ]; then
            verdict="FAIL crash"; notes+="head crashed $h_crashes× (base $b_crashes×); "
        else
            [ "$b_crashes" -gt 0 ] && notes+="base crashed $b_crashes×, head ran every seed; "
            if [ -z "$SSIM_FLOOR" ]; then
                [ "$changed" = 1 ] && { verdict="FAIL"; notes+="output CHANGED; "; }
            else
                if [ "$min_ssim" != "-" ] && awk -v a="$min_ssim" -v f="$SSIM_FLOOR" 'BEGIN { exit !(a < f) }'; then
                    verdict="FAIL"; notes+="SSIM $min_ssim < $SSIM_FLOOR; "
                elif [ "$min_ssim" = "-" ] && [ "$changed" = 1 ]; then
                    verdict="FAIL"; notes+="text CHANGED; "
                fi
            fi
            # A speed verdict needs a noise floor: the base arm's own range over its warm seeds. A delta inside that
            # range, or one measured from a single warm rep, is reported rather than failed.
            speed="$d_step"; series="$bs"
            [ "$speed" = "-" ] && { speed="$d_wall"; series="$bw"; }
            spread="$(printf '%s' "$series" | awk 'NF { n++; if (n == 1 || $1 < lo) lo = $1; if (n == 1 || $1 > hi) hi = $1 } END { if (n < 2 || lo + 0 == 0) print "-"; else printf "%.1f", (hi - lo) / lo * 100 }')"
            if [ "$speed" != "-" ] && awk -v s="$speed" -v t="$SPEED_TOL" 'BEGIN { exit !(s > t) }'; then
                if [ "$spread" = "-" ]; then
                    notes+="slower by ${speed}% (one warm rep, not judged); "
                elif awk -v s="$speed" -v n="$spread" 'BEGIN { exit !(s > n) }'; then
                    verdict="FAIL"; notes+="slower by ${speed}% (base spread ${spread}%); "
                else
                    notes+="slower by ${speed}%, inside the base spread of ${spread}%; "
                fi
            elif [ "$speed" != "-" ] && awk -v s="$speed" -v t="$SPEED_TOL" 'BEGIN { exit !(s < -t) }'; then
                notes+="faster by ${speed#-}%; "
            fi
        fi
        [ "$verdict" != PASS ] && [ "$verdict" != CRASH-both ] && FAILURES=$((FAILURES + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$id" "$backend" "$verdict" "$b_wall_m" "$h_wall_m" "$d_wall" "$b_step_m" "$h_step_m" "$d_step" \
            "$b_peak_m" "$h_peak_m" "$d_peak" "$min_ssim" "$digests" "$notes" >>"$ROWS"
    done < <(regression_cases "$TAG" $backend_tag)
done

[ "$CASES_RUN" -eq 0 ] && die "no case matched --filter '$FILTER' with tag '$TAG'"

# ── report ───────────────────────────────────────────────────────────────────────────────────────────────────
{
    printf '# Regression gate — %s vs %s\n\n' "$BASE_LABEL" "$HEAD_LABEL"
    printf -- '- when: %s UTC · GPU: %s · expectation: %s · warm reps: %s · speed tolerance: ±%s%%\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU_LABEL" "$EXPECT" "$REPS" "$SPEED_TOL"
    [ ${#BASE_SET[@]} -gt 0 ] && printf -- '- base knobs: %s\n' "${BASE_SET[*]}"
    [ ${#HEAD_SET[@]} -gt 0 ] && printf -- '- head knobs: %s\n' "${HEAD_SET[*]}"
    printf -- '- runs: %s · pairs: %s\n\n' "$OUT/runs" "$REPORT_DIR/pairs"
    printf '| case | backend | verdict | wall s (base→head) | Δ | step ms (base→head) | Δ | peak VRAM MiB (base→head) | Δ | min SSIM | digests | notes |\n'
    printf '|---|---|---|---|---|---|---|---|---|---|---|---|\n'
    while IFS=$'\t' read -r id backend verdict bw hw dw bs hs ds bv hv dv ssim digests notes; do
        [ "$verdict" = untested ] && { printf '| %s | %s | untested | | | | | | | | | checkpoint missing |\n' "$id" "$backend"; continue; }
        printf '| %s | %s | %s | %s → %s | %s | %s → %s | %s | %s → %s | %s | %s | %s | %s |\n' \
            "$id" "$backend" "$verdict" "$bw" "$hw" "$dw" "$bs" "$hs" "$ds" "$bv" "$hv" "$dv" "$ssim" "$digests" "$notes"
    done <"$ROWS"
    printf '\nStep ms is the median over warm seeds of the CLI'"'"'s per-step interval (per token for text). SSIM is the minimum over seed pairs; digests compare decoded pixels.\n'
} >"$REPORT_DIR/report.md"

cat "$REPORT_DIR/report.md"
log "report: $REPORT_DIR/report.md"
if [ "$FAILURES" -gt 0 ]; then
    log "$FAILURES case(s) failed the gate"
    exit 1
fi
log "all $CASES_RUN case(s) passed the gate"
