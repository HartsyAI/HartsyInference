#!/usr/bin/env bash
# The rented-GPU run: everything a Blackwell card has to answer, in the order that fails cheapest first, with
# nothing waiting on a person. Rehearse it on the 4090 (--rehearsal) before renting; on the pod it produces a
# bundle, prints a summary, and can stop the pod itself. Stages resume from markers, so a rerun after a fix
# repeats only what did not complete.
#
#   tests/blackwell-run.sh --rehearsal --gpu 1                # on the local card (nvidia-smi index; a one-card pod is 0): everything except the Blackwell-only rows
#   tests/blackwell-run.sh --budget-minutes 60 --auto-stop     # on the pod
#   tests/blackwell-run.sh --stage gpu-tests                   # rerun one stage
#
# Stages: preflight → bootstrap → probe → gpu-tests → models → vulkan (opt-in) → collect. `preflight` is the only
# stage that may fail cheaply: the driver must be 580+ (every nvcc-built PTX here is ISA 9.0, which a 570/12.8
# driver refuses to JIT), the card must be the one the session is paying for, and cuBLAS must resolve.
#
# The engine itself reads no environment variables; the HARTSY_* ones below are test-harness switches that turn a
# skipped GPU test into a failure, so a green run means the tests ran.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${BLACKWELL_RUN_OUT:-$HOME/blackwell-run}"
MODELS="${HARTSY_MODELS_ROOT:-/mnt/model-storage/Models}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-}"
BUDGET_MINUTES=0
ONLY_STAGE=""
REHEARSAL=0
AUTO_STOP=0
WITH_VULKAN=0
WITH_BENCH=0
BASE_RUNS=""
BASE_SHA=""
GPU_INDEX="${GPU_INDEX:-0}"

while [ $# -gt 0 ]; do
    case "$1" in
        --out)            OUT="$2"; shift 2 ;;
        --models-root)    MODELS="$2"; shift 2 ;;
        --cuda-lib-dir)   CUDA_LIB_DIR="$2"; shift 2 ;;
        --budget-minutes) BUDGET_MINUTES="$2"; shift 2 ;;
        --stage)          ONLY_STAGE="$2"; shift 2 ;;
        --rehearsal)      REHEARSAL=1; shift ;;
        --auto-stop)      AUTO_STOP=1; shift ;;
        --with-vulkan)    WITH_VULKAN=1; shift ;;
        --with-bench)     WITH_BENCH=1; shift ;;
        --base-runs)      BASE_RUNS="$2"; shift 2 ;;   # a tar of the 4090's regression-ab runs, for cross-card SSIM
        --base-sha)       BASE_SHA="$2"; shift 2 ;;    # the commit those runs were made from
        --gpu)            GPU_INDEX="$2"; shift 2 ;;
        -h|--help)        sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The GPU test projects build their backend on CUDA device 0: pin the card the session is for, by UUID, around
# every dotnet test below. The CLI runs take the nvidia-smi index through regression-ab.sh instead.
GPU_UUID="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader 2>/dev/null || true)"
FAILURES=0
mkdir -p "$OUT/logs"
START=$(date +%s)
LOG="$OUT/logs/run.log"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" >&2; }
die() { log "FATAL: $*"; exit 1; }
elapsed_min() { echo $(( ($(date +%s) - START) / 60 )); }
over_budget() { [ "$BUDGET_MINUTES" -gt 0 ] && [ "$(elapsed_min)" -ge "$BUDGET_MINUTES" ]; }
done_marker() { echo "$OUT/.done-$1"; }
stage_wanted() {   # stage_wanted <name>: run it unless --stage names another, or it already completed
    local s="$1"
    if [ -n "$ONLY_STAGE" ] && [ "$ONLY_STAGE" != "$s" ]; then return 1; fi
    if [ -z "$ONLY_STAGE" ] && [ -f "$(done_marker "$s")" ]; then log "── $s: already done (rm $(done_marker "$s") to repeat)"; return 1; fi
    if over_budget && [ "$s" != collect ]; then log "── $s: skipped, budget of ${BUDGET_MINUTES} min spent"; return 1; fi
    log "── $s"
    return 0
}
finish_stage() { touch "$(done_marker "$1")"; log "   $1 done at $(elapsed_min) min"; }
# within_budget <cmd...>: a stage already running when the budget ends is cut at the deadline, not left to finish.
within_budget() {
    if [ "$BUDGET_MINUTES" -gt 0 ]; then
        local left=$(( BUDGET_MINUTES * 60 - ($(date +%s) - START) ))
        [ "$left" -lt 1 ] && left=1
        timeout "$left" "$@"
    else
        "$@"
    fi
}
# The pod stops on ANY exit when --auto-stop was asked for — a fatal preflight is exactly the unattended case it exists for.
STOPPED=0
auto_stop() {
    [ "$AUTO_STOP" = 1 ] || return 0
    [ "$STOPPED" = 0 ] || return 0
    STOPPED=1
    if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        log "stopping pod $RUNPOD_POD_ID"; runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl stop failed — stop the pod from the console"
    else
        log "--auto-stop: no runpodctl/RUNPOD_POD_ID here; stop the machine from the provider console"
    fi
}
trap auto_stop EXIT

CLI_DLL="$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll"
hartsy() { dotnet "$CLI_DLL" "$@"; }
export HARTSY_REQUIRE_BACKENDS=1
export HARTSY_REQUIRE_REAL_WEIGHTS=1
export HARTSYINFERENCE_MODELS_DIR="$MODELS"   # the real-weight tests resolve their assets here, not under the repo
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ── preflight ────────────────────────────────────────────────────────────────────────────────────────────────
if stage_wanted preflight; then
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
    driver="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=driver_version --format=csv,noheader)"
    name="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader)"
    cc="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo unknown)"
    log "GPU $GPU_INDEX: $name, driver $driver, compute capability $cc"
    [ "${driver%%.*}" -ge 580 ] || die "driver $driver is below 580: the shipped PTX is ISA 9.0 and this driver cannot JIT it (see the header)"
    if [ "$REHEARSAL" = 0 ]; then
        case "$name" in
            *5090*|*Blackwell*|*B200*|*B300*|*"RTX PRO"*) ;;
            *) die "GPU is '$name', not the Blackwell card this session is for (pass --rehearsal to run elsewhere)" ;;
        esac
    fi
    if [ -n "$CUDA_LIB_DIR" ]; then
        ls "$CUDA_LIB_DIR"/libcublasLt.so* >/dev/null 2>&1 || die "no libcublasLt in $CUDA_LIB_DIR"
    elif ! ls "$HOME"/.local/lib/cuda13/libcublasLt.so* >/dev/null 2>&1 && ! ldconfig -p 2>/dev/null | grep -q libcublasLt; then
        die "cuBLASLt not found: pass --cuda-lib-dir <dir with libcublas*.so> (the engine probes ~/.local/lib/cuda13 and the loader path)"
    fi
    finish_stage preflight
fi

# ── bootstrap ────────────────────────────────────────────────────────────────────────────────────────────────
if stage_wanted bootstrap; then
    if ! command -v dotnet >/dev/null || ! dotnet --list-sdks 2>/dev/null | grep -q '^10\.'; then
        log "installing the .NET 10 SDK"
        (command -v apt-get >/dev/null && (sudo -n true 2>/dev/null && sudo apt-get install -y -q dotnet-sdk-10.0 || apt-get install -y -q dotnet-sdk-10.0)) >>"$LOG" 2>&1 \
            || die "could not install dotnet-sdk-10.0; install it and rerun --stage bootstrap"
    fi
    command -v ffmpeg >/dev/null || (sudo -n true 2>/dev/null && sudo apt-get install -y -q ffmpeg || apt-get install -y -q ffmpeg) >>"$LOG" 2>&1 || log "WARNING: ffmpeg missing; image digests and SSIM will not work"
    log "building the CLI and the GPU test projects from $(git -C "$REPO" rev-parse --short HEAD)"
    for proj in src/HartsyInference.Cli tests/HartsyInference.Cuda.Tests tests/HartsyInference.Gpu.Tests tests/HartsyInference.API.Tests; do
        dotnet build "$REPO/$proj" -c Release --nologo -v q >>"$LOG" 2>&1 || die "build of $proj failed — see $LOG"
    done
    [ -f "$CLI_DLL" ] || die "CLI did not build at $CLI_DLL"
    hartsy settings set paths.modelsRoot "$MODELS" >>"$LOG" 2>&1 || die "settings set failed"
    [ -n "$CUDA_LIB_DIR" ] && { hartsy settings set paths.cudaLibDir "$CUDA_LIB_DIR" >>"$LOG" 2>&1 || die "settings set cudaLibDir failed"; }
    finish_stage bootstrap
fi

# ── probe ────────────────────────────────────────────────────────────────────────────────────────────────────
if stage_wanted probe; then
    # Constructing CudaBackend + CudaKernels loads every shipped module: this is where the 27 hand-written
    # `.version 7.0 / .target sm_70` PTX either JIT onto this card or do not. Answered in seconds. A test that
    # skips for lack of CUDA passes, so the log is read for the house SKIPPED line as well as the exit status.
    rm -f "$OUT/.probe-failed"
    if within_budget env ${GPU_UUID:+CUDA_VISIBLE_DEVICES=$GPU_UUID} dotnet test "$REPO/tests/HartsyInference.Cuda.Tests" -c Release --no-build --nologo -v q \
        --filter "FullyQualifiedName~GgufGpuDequantTests" --logger "console;verbosity=normal" >"$OUT/logs/probe.log" 2>&1 \
        && ! grep -q "SKIPPED" "$OUT/logs/probe.log"; then
        log "probe: CUDA backend constructs, every shipped PTX module loads, and a kernel runs on this card"
    else
        log "probe FAILED — the shipped PTX does not run here; see $OUT/logs/probe.log"
        grep -iE "error|unsupported|version|target" "$OUT/logs/probe.log" | head -20 | tee -a "$LOG" >&2
        touch "$OUT/.probe-failed"
    fi
    {
        printf '{\n  "when": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "commit": "%s",\n' "$(git -C "$REPO" rev-parse HEAD)"
        printf '  "gpu": "%s",\n' "$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name,driver_version,compute_cap,memory.total,uuid --format=csv,noheader)"
        printf '  "dotnet": "%s",\n' "$(dotnet --version)"
        printf '  "cublas": "%s",\n' "$(ls "${CUDA_LIB_DIR:-$HOME/.local/lib/cuda13}"/libcublas.so* 2>/dev/null | head -1)"
        printf '  "probe": "%s"\n}\n' "$([ -f "$OUT/.probe-failed" ] && echo failed || echo ok)"
    } >"$OUT/env.json"
    log "env: $(tr -d '\n' <"$OUT/env.json")"
    if [ "$WITH_BENCH" = 1 ]; then
        dotnet publish "$REPO/benchmarks/HartsyInference.BenchmarkRunner" -c Release -r linux-x64 --self-contained true \
            -p:BenchmarkRevision="$(git -C "$REPO" rev-parse HEAD)" -o "$OUT/bench" --nologo -v q >>"$LOG" 2>&1 \
            && "$OUT/bench/hartsy-bench" doctor --device "cuda:$GPU_INDEX" >"$OUT/logs/doctor.log" 2>&1 \
            && log "hartsy-bench doctor: ok" || log "hartsy-bench doctor failed (see $OUT/logs/doctor.log)"
    fi
    [ -f "$OUT/.probe-failed" ] && die "stopping after the probe: nothing downstream can pass"
    finish_stage probe
fi

# ── gpu-tests ────────────────────────────────────────────────────────────────────────────────────────────────
# A skipped test reports as passed, so the log is grepped for the house SKIPPED line; in rehearsal the
# Blackwell-only rows are allowed to skip and named.
run_tests() {   # run_tests <project> <filter> <logname>
    local proj="$1" filter="$2" name="$3" logfile="$OUT/logs/$3.log"
    if within_budget env ${GPU_UUID:+CUDA_VISIBLE_DEVICES=$GPU_UUID} dotnet test "$REPO/$proj" -c Release --no-build --nologo -v q --filter "$filter" \
        --logger "console;verbosity=normal" >"$logfile" 2>&1; then
        local passed; passed="$(grep -oE "Passed:\s+[0-9]+" "$logfile" | tail -1)"
        local skipped; skipped="$(grep -c "SKIPPED" "$logfile" || true)"
        if [ "$skipped" -gt 0 ] && [ "$REHEARSAL" = 0 ]; then
            log "   $name: $passed but $skipped SKIPPED line(s) — FAIL (green must mean ran)"; grep SKIPPED "$logfile" | head -5 | tee -a "$LOG" >&2
            TEST_FAILURES=$((TEST_FAILURES + 1))
        elif [ "$skipped" -gt 0 ]; then
            local blackwell; blackwell="$(grep -c "Blackwell" "$logfile" || true)"
            log "   $name: $passed, $skipped skipped ($blackwell Blackwell-only; rehearsal)"
        else
            log "   $name: $passed, 0 skipped"
        fi
    else
        log "   $name: FAILED — see $logfile"; grep -E "\[FAIL\]|error" "$logfile" | head -8 | tee -a "$LOG" >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}
TEST_FAILURES=0
if stage_wanted gpu-tests; then
    run_tests tests/HartsyInference.Cuda.Tests "Category=GpuIntegration" cuda-gpuintegration
    run_tests tests/HartsyInference.Gpu.Tests "Category=GpuIntegration" gpu-gpuintegration
    run_tests tests/HartsyInference.Cuda.Tests "FullyQualifiedName~BlockScaledGemmTests|FullyQualifiedName~BlockQuantKernelTests|FullyQualifiedName~Fp8NativeGemmTests|FullyQualifiedName~PtxVariantResolutionTests" blackwell-named
    grep -h "TIMING\|rel_err\|corr=" "$OUT"/logs/blackwell-named.log 2>/dev/null | sed 's/^\s*//' | tee -a "$LOG" >&2
    [ "$TEST_FAILURES" -eq 0 ] && finish_stage gpu-tests || { FAILURES=$((FAILURES + 1)); log "gpu-tests: $TEST_FAILURES failure(s); not marking done"; }
fi

# ── models ───────────────────────────────────────────────────────────────────────────────────────────────────
if stage_wanted models; then
    AB="$REPO/tests/regression-ab.sh"
    ABOUT="$OUT/regression"
    models_fail=0
    if [ -n "$BASE_RUNS" ] && [ -n "$BASE_SHA" ]; then
        mkdir -p "$ABOUT/runs" && tar -xzf "$BASE_RUNS" -C "$ABOUT/runs" && log "unpacked the 4090 base runs for $BASE_SHA"
        # Cross-card: same seeds against the 4090's images. A different arch never reproduces bytes; the number is
        # read against the cross-arch ceiling (a 4090↔3060 pair tops out near 0.78), not a same-card floor.
        within_budget "$AB" --base "$BASE_SHA" --head "$REPO" --out "$ABOUT" --gpu "$GPU_INDEX" --gpu-name "" --expect bounded:0.50 --reps 1 \
            >"$OUT/logs/ab-cross-card.log" 2>&1 || log "cross-card run reported failures (expected to differ; read the SSIM column)"
        grep -E "^\| " "$OUT/logs/ab-cross-card.log" | tee -a "$LOG" >&2
    fi
    # Same card, this build only: timing, VRAM, crashes across the core set.
    within_budget "$AB" --no-base --head "$REPO" --out "$ABOUT" --gpu "$GPU_INDEX" --gpu-name "" --reps 2 >"$OUT/logs/ab-head-only.log" 2>&1 \
        || { models_fail=1; log "head-only core run reported failures — see $OUT/logs/ab-head-only.log"; }
    grep -E "^\| " "$OUT/logs/ab-head-only.log" | tee -a "$LOG" >&2
    # The question this session exists for: native block-scaled GEMM off vs on, same card, same build, same seeds.
    within_budget "$AB" --base "$REPO" --head "$REPO" --base-set numerics.fp4Native=false --head-set numerics.fp4Native=true \
        --out "$ABOUT" --gpu "$GPU_INDEX" --gpu-name "" --filter zimage,lens-mxfp8 --tag "" --expect bounded:0.90 --reps 2 \
        >"$OUT/logs/ab-fp4native.log" 2>&1 || { models_fail=1; log "fp4Native off/on run reported failures — see $OUT/logs/ab-fp4native.log"; }
    grep -E "^\| |knobs" "$OUT/logs/ab-fp4native.log" | tee -a "$LOG" >&2
    # Within-session determinism: the same build twice must reproduce its own bytes on this card, pinned to it like
    # the tests; a recording that fails is a failure, not a stale baseline to compare against.
    if within_budget env ${GPU_UUID:+CUDA_VISIBLE_DEVICES=$GPU_UUID} HARTSY_MIGRATION_BASELINE="$OUT/migration" "$REPO/tests/migration-baseline.sh" --backend cuda --record --no-build >"$OUT/logs/baseline-record.log" 2>&1; then
        within_budget env ${GPU_UUID:+CUDA_VISIBLE_DEVICES=$GPU_UUID} HARTSY_MIGRATION_BASELINE="$OUT/migration" "$REPO/tests/migration-baseline.sh" --backend cuda --no-build >"$OUT/logs/baseline-compare.log" 2>&1 \
            && log "determinism: this card reproduces its own generations" || { models_fail=1; log "determinism: CHANGED between two identical runs — see $OUT/logs/baseline-compare.log"; }
    else
        models_fail=1; log "determinism: recording failed — see $OUT/logs/baseline-record.log"
    fi
    if [ "$WITH_BENCH" = 1 ] && [ -x "$OUT/bench/hartsy-bench" ]; then
        "$OUT/bench/hartsy-bench" fetch --suite quick-v1 --cache "$OUT/bench-cache" >>"$LOG" 2>&1 \
            && "$OUT/bench/hartsy-bench" run --suite quick-v1 --device "cuda:$GPU_INDEX" --cache "$OUT/bench-cache" --output "$OUT/bench-run" >>"$LOG" 2>&1 \
            && "$OUT/bench/hartsy-bench" export --input "$OUT/bench-run" --bundle "$OUT/bench-run.zip" >>"$LOG" 2>&1 \
            && log "hartsy-bench quick-v1 exported" || log "hartsy-bench run did not complete"
    fi
    [ "$models_fail" = 0 ] && finish_stage models || { FAILURES=$((FAILURES + 1)); log "models: failures; not marking done"; }
fi

# ── vulkan (opt-in, capped) ──────────────────────────────────────────────────────────────────────────────────
if [ "$WITH_VULKAN" = 1 ] && stage_wanted vulkan; then
    (command -v vulkaninfo >/dev/null || (sudo -n true 2>/dev/null && sudo apt-get install -y -q libvulkan1 vulkan-tools || apt-get install -y -q libvulkan1 vulkan-tools)) >>"$LOG" 2>&1
    if vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
        dotnet build "$REPO/tests/HartsyInference.Vulkan.Tests" -c Release --nologo -v q >>"$LOG" 2>&1
        vk_fail=0
        timeout 300 bash -c "$(declare -f run_tests log); TEST_FAILURES=0; REHEARSAL=$REHEARSAL; OUT='$OUT'; REPO='$REPO'; LOG='$LOG'; GPU_UUID='$GPU_UUID'; run_tests tests/HartsyInference.Vulkan.Tests Category=GpuIntegration vulkan-gpuintegration; exit \$TEST_FAILURES" \
            || { vk_fail=1; log "vulkan tests failed or hit the 5-minute cap"; }
        timeout 300 "$REPO/tests/vulkan-smoke-matrix.sh" --filter sd15 >"$OUT/logs/vulkan-smoke.tsv" 2>"$OUT/logs/vulkan-smoke.log" || { vk_fail=1; log "vulkan smoke hit the cap or failed"; }
        [ "$vk_fail" = 0 ] && finish_stage vulkan || { FAILURES=$((FAILURES + 1)); log "vulkan: failures; not marking done"; }
    else
        log "vulkan: no NVIDIA ICD visible to vulkaninfo; skipped"
    fi
fi

# ── collect ──────────────────────────────────────────────────────────────────────────────────────────────────
if stage_wanted collect; then
    collect_fail=0
    if tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"; then
        log "bundle: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
    else
        collect_fail=1; log "bundle FAILED — the results stay under $OUT; the pod is left running so they can be fetched"
        AUTO_STOP=0
    fi
    {
        echo "# Blackwell run — $(date -u +%Y-%m-%dT%H:%MZ), $(elapsed_min) min"
        echo; cat "$OUT/env.json" 2>/dev/null; echo
        echo "## Stages"; for s in preflight bootstrap probe gpu-tests models vulkan; do printf -- '- %s: %s\n' "$s" "$([ -f "$(done_marker "$s")" ] && echo done || echo 'not done')"; done
        echo; echo "## Blackwell-named tests"; grep -h "TIMING\|rel_err\|corr=\|SKIPPED\|FAILED" "$OUT"/logs/blackwell-named.log 2>/dev/null | sed 's/^\s*/- /'
        echo; echo "## fp4Native off → on"; grep -E "^\| " "$OUT/logs/ab-fp4native.log" 2>/dev/null
        echo; echo "## Head-only core set"; grep -E "^\| " "$OUT/logs/ab-head-only.log" 2>/dev/null
    } >"$OUT/SUMMARY.md"
    cat "$OUT/SUMMARY.md" >&2
    [ "$collect_fail" = 0 ] && finish_stage collect || FAILURES=$((FAILURES + 1))
fi
log "finished at $(elapsed_min) min, $FAILURES failing stage(s)"
exit $(( FAILURES > 0 ))
