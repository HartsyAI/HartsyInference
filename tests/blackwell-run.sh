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
# Stages: preflight → bootstrap → probe → gpu-tests → models → vulkan (opt-in) → swarm (opt-in) → collect. `preflight` is the only
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
WITH_SWARM=0
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
        --with-swarm)     WITH_SWARM=1; shift ;;
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
    vulkaninfo --summary >"$OUT/logs/vulkaninfo.txt" 2>&1 || true
    if vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
        dotnet build "$REPO/tests/HartsyInference.Vulkan.Tests" -c Release --nologo -v q >>"$LOG" 2>&1
        vk_fail=0
        timeout 300 bash -c "$(declare -f run_tests log); TEST_FAILURES=0; REHEARSAL=$REHEARSAL; OUT='$OUT'; REPO='$REPO'; LOG='$LOG'; GPU_UUID='$GPU_UUID'; run_tests tests/HartsyInference.Vulkan.Tests Category=GpuIntegration vulkan-gpuintegration; exit \$TEST_FAILURES" \
            || { vk_fail=1; log "vulkan tests failed or hit the 5-minute cap"; }
        timeout 300 "$REPO/tests/vulkan-smoke-matrix.sh" --filter sd15 >"$OUT/logs/vulkan-smoke.tsv" 2>"$OUT/logs/vulkan-smoke.log" || { vk_fail=1; log "vulkan smoke hit the cap or failed"; }
        [ "$vk_fail" = 0 ] && finish_stage vulkan || { FAILURES=$((FAILURES + 1)); log "vulkan: failures; not marking done"; }
    else
        # Not a code problem and not worth a retry: the container was started without graphics capability, so
        # the loader sees no NVIDIA ICD at all. Recreate the pod with NVIDIA_DRIVER_CAPABILITIES=all.
        log "vulkan: no NVIDIA ICD (see $OUT/logs/vulkaninfo.txt). The pod needs NVIDIA_DRIVER_CAPABILITIES=all; skipped"
    fi
fi

# ── swarm (opt-in, capped): the extension driving this engine through the SwarmUI API ────────────────────────
# The engine generating from the CLI does not prove the extension does: Swarm loads it in its own AssemblyLoadContext
# against the PINNED NuGet engine, not this checkout. This stage is the only thing that exercises that pairing.
if [ "$WITH_SWARM" = 1 ] && stage_wanted swarm; then
    SWARM_DIR="${SWARM_DIR:-$HOME/swarm-test}"
    SWARM_DATA="$SWARM_DIR/TestData"
    SWARM_PORT="${SWARM_PORT:-7899}"
    SWARM_HOST="${SWARM_HOST:-127.0.0.1}"   # a pod sets 0.0.0.0 so the provider's proxy can reach it
    swarm_fail=0
    # HEAD, not a branch name: the two repositories do not agree on what their default branch is called.
    SWARM_REF="${SWARM_REF:-HEAD}"; SWARM_EXT_REF="${SWARM_EXT_REF:-HEAD}"
    [ -d "$SWARM_DIR/.git" ] || git clone -q https://github.com/mcmonkeyprojects/SwarmUI "$SWARM_DIR" >>"$LOG" 2>&1 || { swarm_fail=1; log "swarm: clone failed"; }
    if [ "$swarm_fail" = 0 ]; then
        EXT_DIR="$SWARM_DIR/src/Extensions/SwarmUI-HartsyInference-Backend"
        [ -d "$EXT_DIR/.git" ] || git clone -q https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend "$EXT_DIR" >>"$LOG" 2>&1
        # A reused directory would otherwise rebuild whatever revision it happened to hold, and report a pass that
        # names no pairing. Pin both, and write the SHAs into the bundle so the result says what was tested.
        # fetch <ref> + FETCH_HEAD, not origin/<ref>: a directory left by an earlier run may be a shallow clone
        # with no remote-tracking branch, and this form works for both.
        ( cd "$SWARM_DIR" && git fetch -q origin "$SWARM_REF" && git checkout -q --detach FETCH_HEAD ) >>"$LOG" 2>&1 || { swarm_fail=1; log "swarm: could not check out SwarmUI $SWARM_REF"; }
        ( cd "$EXT_DIR" && git fetch -q origin "$SWARM_EXT_REF" && git checkout -q --detach FETCH_HEAD ) >>"$LOG" 2>&1 || { swarm_fail=1; log "swarm: could not check out the extension $SWARM_EXT_REF"; }
        printf 'swarmui=%s\nextension=%s\nengine_pin=%s\n' \
            "$(git -C "$SWARM_DIR" rev-parse --short HEAD 2>/dev/null)" \
            "$(git -C "$EXT_DIR" rev-parse --short HEAD 2>/dev/null)" \
            "$(grep -oE 'Include="HartsyInference" Version="[^"]+"' "$EXT_DIR/SwarmUI-HartsyInference.csproj" 2>/dev/null | grep -oE '[0-9][^"]*')" \
            >"$OUT/swarm-revisions.txt"
        log "swarm: $(tr '\n' ' ' <"$OUT/swarm-revisions.txt")"
        mkdir -p "$SWARM_DATA"
        # IsInstalled is the whole trick: without it every request lands on the installer page and the API never
        # answers. LaunchMode none keeps it from trying to open a browser on a headless box.
        # printf, not a heredoc: FDS indents with real tabs and a heredoc would write a literal backslash-t.
        printf 'IsInstalled: true\nInstallDate: %s\nInstallVersion: 0.9.8.0\nLaunchMode: none\nPaths:\n\tModelRoot: %s\nNetwork:\n\tHost: %s\n\tPort: %s\n\tPortCanChange: false\n' \
            "$(date -u +%Y-%m-%d)" "$MODELS" "$SWARM_HOST" "$SWARM_PORT" >"$SWARM_DATA/Settings.fds"
        # GPU_ID is a CUDA ordinal, and this script exports CUDA_DEVICE_ORDER=PCI_BUS_ID, which the server
        # inherits — so the ordinal is the nvidia-smi index, not the fastest-first default. Getting this wrong
        # silently generates on the other card: same seed, same parameters, different pixels.
        printf '0:\n\ttype: hartsyinference\n\ttitle: HartsyInference\n\tenabled: true\n\tsettings:\n\t\tComputeBackend: cuda\n\t\tGPU_ID: %s\n' "$GPU_INDEX" >"$SWARM_DATA/Backends.fds"
        within_budget env -C "$SWARM_DIR" dotnet build src/SwarmUI.csproj -c Release -o src/bin/live_release --nologo -v q >>"$LOG" 2>&1 \
            || { swarm_fail=1; log "swarm: build failed or ran out of budget"; }
    fi
    if [ "$swarm_fail" = 0 ]; then
        # The launcher is a native host, NOT `dotnet SwarmUI.dll` — dotnet reads that path as a subcommand and
        # reports the file missing, which is a confusing way to lose twenty minutes on a rented card.
        ( cd "$SWARM_DIR" && ./src/bin/live_release/SwarmUI --data_dir "$SWARM_DATA" --settings_file "$SWARM_DATA/Settings.fds" ) >"$OUT/logs/swarm-server.log" 2>&1 &
        SWARM_PID=$!
        # The server is a background process: a budget timeout that kills the foreground would otherwise leave it
        # holding the card. Reap it AND keep auto_stop — replacing the EXIT trap outright would leave the rented
        # pod running forever, which is the opposite of what this script is for.
        trap 'kill "${SWARM_PID:-}" 2>/dev/null; auto_stop' EXIT
        swarm_up=0
        for _ in $(seq 1 90); do
            [ "$BUDGET_MINUTES" -gt 0 ] && [ $(( BUDGET_MINUTES * 60 - ($(date +%s) - START) )) -lt 1 ] && { log "swarm: out of budget while waiting for the API"; break; }
            curl -s -m 4 -X POST "http://127.0.0.1:$SWARM_PORT/API/GetNewSession" -H 'Content-Type: application/json' -d '{}' 2>/dev/null | grep -q session_id && { swarm_up=1; break; }
            sleep 2
        done
        if [ "$swarm_up" = 1 ]; then
            SID=$(curl -s -m 10 -X POST "http://127.0.0.1:$SWARM_PORT/API/GetNewSession" -H 'Content-Type: application/json' -d '{}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
            REL=$(within_budget curl -s -m 900 -X POST "http://127.0.0.1:$SWARM_PORT/API/GenerateText2Image" -H 'Content-Type: application/json' \
                  -d "{\"session_id\":\"$SID\",\"images\":1,\"prompt\":\"a red fox sitting in a snowy forest, sharp detail\",\"model\":\"SD15/v1-5-pruned-emaonly.safetensors\",\"width\":512,\"height\":512,\"steps\":8,\"cfgscale\":7,\"seed\":42}" \
                  | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d["images"][0] if d.get("images") else "ERR")
except Exception: print("ERR")')
            if [ "$REL" = "ERR" ]; then
                swarm_fail=1; log "swarm: the API refused the generation — see $OUT/logs/swarm-server.log"
            else
                # GenerateText2Image answers before the file is flushed.
                F="$SWARM_DIR/Output/${REL#View/}"
                for _ in $(seq 1 60); do [ -s "$F" ] && break; sleep 0.5; done
                if [ -s "$F" ]; then
                    cp "$F" "$OUT/swarm-sd15-s42.png"
                    log "swarm: generated $(du -h "$OUT/swarm-sd15-s42.png" | cut -f1) through the API"
                    grep -oE "Step [0-9]+/[0-9]+ .*done in [0-9]+ms" "$OUT/logs/swarm-server.log" | tail -8 >"$OUT/logs/swarm-steps.txt"
                else
                    swarm_fail=1; log "swarm: no output file at $F"
                fi
            fi
        else
            swarm_fail=1; log "swarm: API never answered — see $OUT/logs/swarm-server.log"
        fi
        kill "$SWARM_PID" 2>/dev/null; wait "$SWARM_PID" 2>/dev/null
        trap auto_stop EXIT
    fi
    [ "$swarm_fail" = 0 ] && finish_stage swarm || { FAILURES=$((FAILURES + 1)); log "swarm: failures; not marking done"; }
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
        echo "## Stages"; for s in preflight bootstrap probe gpu-tests models vulkan swarm; do printf -- '- %s: %s\n' "$s" "$([ -f "$(done_marker "$s")" ] && echo done || echo 'not done')"; done
        echo; echo "## Blackwell-named tests"; grep -h "TIMING\|rel_err\|corr=\|SKIPPED\|FAILED" "$OUT"/logs/blackwell-named.log 2>/dev/null | sed 's/^\s*/- /'
        echo; echo "## fp4Native off → on"; grep -E "^\| " "$OUT/logs/ab-fp4native.log" 2>/dev/null
        echo; echo "## Head-only core set"; grep -E "^\| " "$OUT/logs/ab-head-only.log" 2>/dev/null
    } >"$OUT/SUMMARY.md"
    cat "$OUT/SUMMARY.md" >&2
    [ "$collect_fail" = 0 ] && finish_stage collect || FAILURES=$((FAILURES + 1))
fi
log "finished at $(elapsed_min) min, $FAILURES failing stage(s)"
exit $(( FAILURES > 0 ))
