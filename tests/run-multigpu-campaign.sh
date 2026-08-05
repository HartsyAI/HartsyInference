#!/usr/bin/env bash
# Multi-GPU verification campaign: runs every real-weight placement/sharding/CFG-parallel e2e test class,
# one isolated `dotnet test --filter` per class (never whole suites — Diffusion.Tests has a known pre-existing
# nondeterministic native-heap abort in untagged tests, and HARTSY_KEEP_MODELS-style static-init reads require
# process isolation). Under HARTSY_REQUIRE_REAL_WEIGHTS=1 (set below) a missing checkpoint FAILS the run via
# RealWeightGate instead of silently skipping, so a green exit genuinely means every listed test executed.
#
# Usage: tests/run-multigpu-campaign.sh [phaseA|phaseB|all]   (default: phaseA)
#   phaseA — needs only checkpoints already on this box (Krea2, Flux1, SDXL, Qwen-Image-Edit, MiniMaxH3 fp8)
#   phaseB — additionally needs the downloaded checkpoints (chroma, qwen-image, hunyuan-image, wan)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-phaseA}"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/Output/multigpu-campaign/${STAMP}"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/summary.txt"
FAILURES=0

log() { echo "[campaign] $*" | tee -a "${SUMMARY}"; }

# ── Pre-flight ────────────────────────────────────────────────────────────────────────────────────────────

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found — this campaign needs the CUDA box"; exit 1; }

# Stray engine processes hold multi-GB VRAM allocations that break the free-VRAM-gated tests. Kill our own
# CLI/test leftovers automatically; anything else (especially swarmui.service) is only reported — stopping the
# user's live service is an operator decision, never this script's.
while read -r pid name mem; do
    [ -z "${pid}" ] && continue
    case "${name}" in
        *HartsyInference.Cli*|*testhost*)
            log "pre-flight: killing stray ${name} (pid ${pid}, ${mem} MiB VRAM)"
            kill "${pid}" 2>/dev/null || true
            sleep 2
            ;;
        *Xorg*|*rustdesk*|*cinnamon*|*chrome*) ;;  # desktop residents, small and expected
        *)
            log "pre-flight: WARNING — ${name} (pid ${pid}) holds ${mem} MiB VRAM; stop it manually if a VRAM gate fails"
            ;;
    esac
done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | tr ',' ' ')

if systemctl --user is-active swarmui.service >/dev/null 2>&1; then
    log "pre-flight: WARNING — swarmui.service is ACTIVE and may hold VRAM. Stop it before the campaign if VRAM gates fail (systemctl --user stop swarmui.service)."
fi

nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv | tee -a "${SUMMARY}"

# Free-VRAM gate: the sharding/replication tests need most of both cards.
insufficient=0
while IFS=',' read -r idx name free; do
    free_mib=$(echo "${free}" | tr -dc '0-9')
    if echo "${name}" | grep -q "4090"; then min=19000; else min=11000; fi
    if [ "${free_mib}" -lt "${min}" ]; then
        log "pre-flight: FAIL — GPU ${idx} (${name}) has only ${free_mib} MiB free (< ${min})"
        insufficient=1
    fi
done < <(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits)
[ "${insufficient}" -ne 0 ] && { log "aborting: free-VRAM gate failed"; exit 1; }

export HARTSY_REQUIRE_REAL_WEIGHTS=1
export HARTSY_ASSERT_AMBIENT=1

# ── Runner ────────────────────────────────────────────────────────────────────────────────────────────────

# run_class <project-dir-under-tests/> <test-class-name> [extra "ENV=VAL" pairs...]
run_class() {
    local project="$1" cls="$2"
    shift 2
    local logfile="${LOG_DIR}/${cls}.log"
    log "── ${cls} (${project}) $*"
    ( cd "${REPO_ROOT}" && env "$@" dotnet test "tests/${project}/${project}.csproj" \
        --filter "FullyQualifiedName~${cls}" --nologo -v minimal --logger "console;verbosity=detailed" \
        > "${logfile}" 2>&1 )
    local rc=$?
    if [ ${rc} -ne 0 ]; then
        log "   FAILED (exit ${rc}) — see ${logfile}"
        FAILURES=$((FAILURES + 1))
    elif grep -q "SKIPPED" "${logfile}"; then
        log "   FAILED — test self-skipped despite HARTSY_REQUIRE_REAL_WEIGHTS=1 (non-checkpoint guard hit):"
        grep "SKIPPED" "${logfile}" | head -5 | tee -a "${SUMMARY}"
        FAILURES=$((FAILURES + 1))
    elif ! grep -qE "Passed! .* Passed: +[1-9]" "${logfile}"; then
        log "   FAILED — no tests matched/executed for filter ${cls}"
        FAILURES=$((FAILURES + 1))
    else
        grep -E "Passed! " "${logfile}" | tail -1 | tee -a "${SUMMARY}"
    fi
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >> "${SUMMARY}"
}

# ── Phase A: on-disk checkpoints only ─────────────────────────────────────────────────────────────────────

phase_a() {
    run_class HartsyInference.Cuda.Tests       CudaOrdinalMapTests
    run_class HartsyInference.API.Tests        PlacementPlannerTests
    run_class HartsyInference.Core.Tests       PlacementConfigTests
    run_class HartsyInference.LLM.Tests        LlmPlacementTests
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingTests
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingVramTests
    # HARTSY_KEEP_MODELS is a static-readonly read → each fact of the engine class runs in its own process.
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingEngineTests.DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingEngineTests.DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations HARTSY_KEEP_MODELS=0
    # New Phase A classes land with their implementation parts:
    run_class HartsyInference.Diffusion.Tests  FluxComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  SdxlComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  FluxCfgParallelFallbackTests
    run_class HartsyInference.Diffusion.Tests  SameGpuConcurrentRealWeightTests HARTSY_SAME_GPU_CONCURRENT=1
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingTests
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingVramTests
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingEngineTests
    run_class HartsyInference.Diffusion.Tests  FluxDitShardingTests
    run_class HartsyInference.Diffusion.Tests  FluxDitShardingVramTests
    run_class HartsyInference.Video.Tests      MiniMaxH3DitShardingTests
    run_class HartsyInference.Video.Tests      MiniMaxH3DitShardingVramTests
}

# ── Phase B: post-download (hartsy pull chroma qwen-image hunyuan-image wan) ─────────────────────────────

phase_b() {
    run_class HartsyInference.Diffusion.Tests  ChromaDitShardingTests
    run_class HartsyInference.Diffusion.Tests  ChromaDitShardingVramTests
    run_class HartsyInference.Diffusion.Tests  ChromaComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  HunyuanImageDitShardingTests
    run_class HartsyInference.Diffusion.Tests  HunyuanImageDitShardingVramTests
    run_class HartsyInference.Diffusion.Tests  HunyuanImageComponentPlacementEngineTests
    run_class HartsyInference.Video.Tests      WanComponentPlacementEngineTests
    run_class HartsyInference.Video.Tests      CfgBranchParallelWanTests
    run_class HartsyInference.Video.Tests      WanCfgParallelEngineTests
    run_class HartsyInference.Video.Tests      LtxComponentPlacementEngineTests
}

case "${PHASE}" in
    phaseA) phase_a ;;
    phaseB) phase_b ;;
    all)    phase_a; phase_b ;;
    *) echo "unknown phase '${PHASE}' (use phaseA|phaseB|all)"; exit 1 ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────────────────────────────────

log ""
if [ ${FAILURES} -eq 0 ]; then
    log "CAMPAIGN GREEN — every listed class executed and passed with real weights required. Logs: ${LOG_DIR}"
    exit 0
fi
log "CAMPAIGN FAILED — ${FAILURES} class(es) failed. Logs: ${LOG_DIR}"
exit 1
