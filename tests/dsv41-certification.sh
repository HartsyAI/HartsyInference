#!/usr/bin/env bash
# DeepSeek-V4.1-Flash certification runner. One lane per invocation. Inside a lane each test class is its own
# `dotnet test --filter`, run one at a time, so one crashing class cannot hide the others. HARTSY_REQUIRE_REAL_WEIGHTS=1
# is exported, so a missing asset fails its class instead of logging SKIPPED. A class is PASS only when at least one test
# ran and none failed or were skipped. A lane that cannot run on this machine is BLOCKED with its reason, and BLOCKED is
# not green: a certification never passes by leaving work out.
# What to rent for each lane: docs/Checklists/DSV41_RENTAL_RUNBOOK.md. The gates it enforces: docs/Checklists/DSV41_CAMPAIGN_FREEZE.md.
#
# Usage: tests/dsv41-certification.sh <lane> [--dry-run] [--out DIR]
#   cpu-oracle   real-weight CPU oracles against the official checkpoint (no GPU)
#   gpu-expert   CUDA expert-cache and MoE suites on sm_80 or newer, in the order of MOE_RIG_READINESS.md step 3
#   gpu-native   block-scaled FP4 suites through cuBLASLt: every device must be SM 10.0 (datacenter Blackwell) or 12.0 (consumer)
#   multi-gpu    BLOCKED: tensor and expert parallel (plan PRs 20-21) are not built
#   two-node     BLOCKED: rank runner, rendezvous and remote Engram rows (plan PR 22) are not built
#   vulkan-amd   BLOCKED: V4.1 Vulkan kernels and AMD hardware (plan PR 24b) are not built
#   offload      BLOCKED: the home-lab first real token and offloaded residency (plan PR 14) are not built
# --dry-run prints the commands and the preflight result without running anything. Exit codes: 0 green (or a runnable
# dry run), 1 a class failed, 2 the lane is blocked or its preflight failed, 64 bad usage.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
LANE=""
DRY_RUN=0
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --out)
            [ $# -ge 2 ] || { echo "--out needs a directory" >&2; exit 64; }
            OUT="$2"
            shift
            ;;
        --*) echo "unknown argument: $1" >&2; exit 64 ;;
        *)
            [ -z "${LANE}" ] || { echo "one lane per run, got '${LANE}' and '$1'" >&2; exit 64; }
            LANE="$1"
            ;;
    esac
    shift
done
[ -n "${LANE}" ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 64; }
OUT_DIR="${OUT:-${REPO_ROOT}/Output/dsv41-certification/${STAMP}/${LANE}}"

# "Project Class" pairs. Each class runs alone, in the order listed.
CPU_ORACLE_CLASSES=(
    "HartsyInference.LLM.Tests DeepSeekV41RealWeightsTests"
    "HartsyInference.LLM.Tests DeepSeekV41DSparkTests"
    "HartsyInference.LLM.Tests DeepSeekV41DSparkChainTests"
    "HartsyInference.LLM.Tests DeepSeekV41VisionRealWeightsTests"
    "HartsyInference.LLM.Tests DeepSeekV41EncoderTokenizerParityTests"
    "HartsyInference.LLM.Tests DeepSeekV41OutputParserRealTokenizerTests"
    "HartsyInference.ModelAssets.Tokenizers.Tests DeepSeekV41TokenizerParityTests"
    "HartsyInference.ModelAssets.Tests DeepSeekV41Shard3ParityTests"
    "HartsyInference.Core.Tests EngramRealShardTests"
)
GPU_EXPERT_CLASSES=(
    "HartsyInference.Cuda.Tests CudaExpertCacheTests"
    "HartsyInference.Cuda.Tests CudaExpertM1FixtureTests"
    "HartsyInference.Cuda.Tests CudaMoePrimitiveTests"
    "HartsyInference.Cuda.Tests CudaMoeTests"
    "HartsyInference.Cuda.Tests CudaQuantWorkspaceTests"
    "HartsyInference.Cuda.Tests CudaStreamingWeightCacheTests"
)
GPU_NATIVE_CLASSES=(
    "HartsyInference.Cuda.Tests Nvfp4GemmReferenceTests"
    "HartsyInference.Cuda.Tests Nvfp4ResidentCudaParityTests"
    "HartsyInference.Cuda.Tests CudaQuantDerivativeDequantTests"
    "HartsyInference.Cuda.Tests CudaExl3DequantTests"
)

PROBLEMS=()
problem() { PROBLEMS+=("$1"); }

# ── Preflight ─────────────────────────────────────────────────────────────

check_dir() { # name value
    if [ -z "$2" ]; then problem "$1 is not set"
    elif [ ! -d "$2" ]; then problem "$1=$2 is not a directory"; fi
}

check_file() { # name value
    if [ -z "$2" ]; then problem "$1 is not set"
    elif [ ! -f "$2" ]; then problem "$1=$2 is not a file"; fi
}

check_checkpoint() {
    local dir="${HARTSY_DSV41_FLASH_DIR:-}"
    check_dir HARTSY_DSV41_FLASH_DIR "${dir}"
    [ -d "${dir}" ] || return 0
    [ -f "${dir}/model.safetensors.index.json" ] || problem "no model.safetensors.index.json in ${dir}"
    local shards
    shards=$(find "${dir}" -maxdepth 1 -name '*.safetensors' | wc -l)
    [ "${shards}" -eq 48 ] || problem "${dir} holds ${shards} .safetensors shards, the pinned checkpoint has 48"
}

check_cuda() { # "min VALUE" or "set \"V1 V2 ...\"": the rule every GPU must meet
    if ! command -v nvidia-smi >/dev/null 2>&1; then problem "nvidia-smi not found"; return; fi
    local caps cc holders
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -z "${caps}" ]; then problem "nvidia-smi did not report compute capability: no GPU, or a driver too old to query it"; return; fi
    # GPU suites run alone (AGENTS.md): a process already holding a device would contend with the run and skew it.
    # A listing line starts with the PID ("1475, name"). Anything else printed with a non-zero exit is a failed query, which
    # fails closed; an empty listing is an idle GPU whatever its exit code.
    local apps apps_rc
    apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)
    apps_rc=$?
    holders=$(grep -cE '^[0-9]' <<<"${apps}" || true)
    if [ "${holders:-0}" -gt 0 ]; then
        problem "${holders} compute process(es) already hold a GPU; stop them first, GPU suites run one at a time"
    elif [ "${apps_rc}" -ne 0 ] && [ -n "${apps}" ]; then
        problem "could not list compute processes, so the GPU cannot be shown idle: ${apps%%$'\n'*}"
    fi
    # Every GPU must qualify: the tests may land on any device, and a mixed box cannot vouch for one card.
    for cc in ${caps}; do
        case "$1" in
            min)
                awk -v c="${cc}" -v m="$2" 'BEGIN { exit !(c >= m) }' || problem "a GPU has compute capability ${cc}; this lane needs $2 or newer on every device"
                ;;
            set)
                case " $2 " in
                    *" ${cc} "*) ;;
                    *) problem "a GPU has compute capability ${cc}; this lane needs one of: $2 on every device" ;;
                esac
                ;;
        esac
    done
}

blocked_lane() { # reason
    echo "BLOCKED ${LANE}: $1"
    if [ "${DRY_RUN}" -eq 0 ]; then
        mkdir -p "${OUT_DIR}"
        echo "BLOCKED ${LANE}: $1" >>"${OUT_DIR}/summary.txt"
    fi
    exit 2
}

case "${LANE}" in
    cpu-oracle)
        CLASSES=("${CPU_ORACLE_CLASSES[@]}")
        check_checkpoint
        check_dir DSV41_ORACLE_DIR "${DSV41_ORACLE_DIR:-}"
        check_dir DSV41_DSPARK_ORACLE "${DSV41_DSPARK_ORACLE:-}"
        check_dir DSV41_REAL_SLICES "${DSV41_REAL_SLICES:-}"
        check_file DSV41_TOKENIZER_JSON "${DSV41_TOKENIZER_JSON:-}"
        check_dir DSV41_VISION_ORACLE "${DSV41_VISION_ORACLE:-}"
        check_dir HARTSY_DSV41_SHARD3_FIXTURES "${HARTSY_DSV41_SHARD3_FIXTURES:-}"
        # The 40-layer class returns early with a SKIPPED line, not a skip, unless this is set, so it would pass without running.
        [ "${DSV41_FULL_RUN:-}" = "1" ] || problem "DSV41_FULL_RUN=1 is required: the 40-layer class is part of the certification"
        ;;
    gpu-expert)
        CLASSES=("${GPU_EXPERT_CLASSES[@]}")
        check_cuda min 8.0
        check_checkpoint
        check_dir HARTSY_DSV41_SHARD3_FIXTURES "${HARTSY_DSV41_SHARD3_FIXTURES:-}"
        ;;
    gpu-native)
        CLASSES=("${GPU_NATIVE_CLASSES[@]}")
        check_cuda set "10.0 12.0"
        ;;
    multi-gpu) blocked_lane "tensor and expert parallel (plan PRs 20-21) are not built; a run here would certify nothing" ;;
    two-node) blocked_lane "rank runner, rendezvous and remote Engram rows (plan PR 22) are not built" ;;
    vulkan-amd) blocked_lane "V4.1 Vulkan kernels and AMD hardware (plan PR 24b) are not built" ;;
    offload) blocked_lane "the home-lab first real token and offloaded residency (plan PR 14) are not built" ;;
    *) echo "unknown lane: ${LANE} (cpu-oracle, gpu-expert, gpu-native, multi-gpu, two-node, vulkan-amd, offload)" >&2; exit 64 ;;
esac

if [ "${#PROBLEMS[@]}" -gt 0 ]; then
    for p in "${PROBLEMS[@]}"; do echo "PREFLIGHT ${LANE}: ${p}"; done
    if [ "${DRY_RUN}" -eq 0 ]; then
        mkdir -p "${OUT_DIR}"
        for p in "${PROBLEMS[@]}"; do echo "BLOCKED ${LANE}: ${p}" >>"${OUT_DIR}/summary.txt"; done
    fi
    echo "BLOCKED ${LANE}: preflight failed, nothing ran"
    exit 2
fi

# ── Plan ──────────────────────────────────────────────────────────────────

filter_for() { echo "FullyQualifiedName~.${1}."; }

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "DRY RUN ${LANE}: preflight passed; the real run would execute these, one at a time:"
    echo "  export HARTSY_REQUIRE_REAL_WEIGHTS=1"
    for pair in "${CLASSES[@]}"; do
        # shellcheck disable=SC2086 # "Project Class" pairs word-split on purpose
        set -- ${pair}
        echo "  dotnet test tests/$1 --filter \"$(filter_for "$2")\""
    done
    echo "  logs: ${OUT_DIR}/logs/   summary: ${OUT_DIR}/summary.txt"
    exit 0
fi

# ── Run ───────────────────────────────────────────────────────────────────

mkdir -p "${OUT_DIR}/logs"
SUMMARY="${OUT_DIR}/summary.txt"
export HARTSY_REQUIRE_REAL_WEIGHTS=1
FAILED=0

record() { # status name detail
    echo "$1 ${LANE} $2: $3" | tee -a "${SUMMARY}"
    [ "$1" = "PASS" ] || FAILED=1
}

run_class() { # name, dotnet args...
    local name="$1"
    shift
    local log="${OUT_DIR}/logs/${name}.log"
    # Detailed logging shows each test's output, so a SKIPPED line a test writes and then returns from is visible.
    dotnet test "$@" --logger "console;verbosity=detailed" >"${log}" 2>&1
    local rc=$?
    # Sum every summary line, one per target framework, so no framework's failures are missed.
    local runs total failed skipped
    read -r runs total failed skipped < <(awk -f "${REPO_ROOT}/tests/dsv41-certification-summary.awk" "${log}")
    if [ "${runs}" -eq 0 ]; then record FAIL "${name}" "no test summary (exit ${rc}); see logs/${name}.log"
    elif [ "${rc}" -ne 0 ] || [ "${failed}" -ne 0 ]; then record FAIL "${name}" "${failed} failed of ${total} (exit ${rc})"
    elif [ "${skipped}" -ne 0 ]; then record FAIL "${name}" "${skipped} skipped of ${total}: a skip is not a pass"
    elif grep -q "SKIPPED" "${log}"; then record FAIL "${name}" "a test wrote SKIPPED and returned without running: $(grep -m1 "SKIPPED" "${log}" | cut -c1-160)"
    elif [ "${total}" -eq 0 ]; then record FAIL "${name}" "no test matched"
    else record PASS "${name}" "${total} passed"
    fi
}

for pair in "${CLASSES[@]}"; do
    # shellcheck disable=SC2086 # "Project Class" pairs word-split on purpose
    set -- ${pair}
    run_class "$2" "tests/$1" --filter "$(filter_for "$2")"
done

if [ "${FAILED}" -eq 0 ]; then
    echo "GREEN ${LANE}: every class passed with nothing skipped. Summary: ${SUMMARY}"
    exit 0
fi
echo "RED ${LANE}: at least one class did not pass. Summary: ${SUMMARY}"
exit 1
