#!/usr/bin/env bash
# Multi-GPU verification campaign: runs every real-weight placement/sharding/CFG-parallel e2e test class,
# one isolated `dotnet test --filter` per class (never whole suites — Diffusion.Tests has a known pre-existing
# nondeterministic native-heap abort in untagged tests, and HARTSY_KEEP_MODELS-style static-init reads require
# process isolation). Under HARTSY_REQUIRE_REAL_WEIGHTS=1 (set below) a missing checkpoint FAILS the run via
# RealWeightGate instead of silently skipping, so a green exit genuinely means every listed test executed.
#
# Usage: tests/run-multigpu-campaign.sh [phaseA|phaseB|all]   (default: phaseA)
#   phaseA — needs only checkpoints already on this box (Krea2, Flux1, SDXL, Qwen-Image-Edit, MiniMaxH3 fp8)
#   phaseB — additionally needs the downloaded checkpoints (chroma, qwen-image, hunyuan-image, wan,
#            ltx-video [~9.4 GB, Lightricks/LTX-Video], ltx-2 [~22 GB DiT, Kijai/LTX2.3_comfy split — NOT
#            pulled on this box as of 2026-08-05, disk-constrained; its test skips cleanly until it is])
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
        *HartsyInference.Cli*|*testhost*|*/dotnet)
            # Bare "/usr/lib/dotnet/dotnet" is how a leftover `dotnet test` testhost reports itself —
            # on this box every GPU-holding dotnet process is ours.
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
    elif ! grep -qE "Passed:\s+[1-9]" "${logfile}"; then
        # Covers both vstest summary formats: the single-line "Passed!  - Failed: 0, Passed: N" (-v minimal
        # alone) and the multi-line "Test Run Successful. / Total tests: N / Passed: N" (detailed console logger).
        log "   FAILED — no tests matched/executed for filter ${cls}"
        FAILURES=$((FAILURES + 1))
    else
        grep -E "Passed!|Test Run Successful|Total tests:|^ *Passed:" "${logfile}" | tail -2 | tee -a "${SUMMARY}"
    fi
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >> "${SUMMARY}"
    # Driver VRAM teardown lags a heavy test process's exit by a few seconds; the NEXT process's backend
    # construction sizes its weight-cache budget from a probe taken during that window and then OOMs with a
    # near-zero budget (observed twice). Wait until both cards are back above their gate before continuing.
    for _ in $(seq 1 20); do
        low=0
        while IFS=',' read -r idx name free; do
            free_mib=$(echo "${free}" | tr -dc '0-9')
            if echo "${name}" | grep -q "4090"; then min=19000; else min=10000; fi
            [ "${free_mib}" -lt "${min}" ] && low=1
        done < <(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits)
        [ "${low}" -eq 0 ] && break
        sleep 2
    done
}

# ── Phase A: on-disk checkpoints only ─────────────────────────────────────────────────────────────────────

phase_a() {
    run_class HartsyInference.Cuda.Tests       CudaOrdinalMapTests
    run_class HartsyInference.API.Tests        PlacementPlannerTests
    run_class HartsyInference.Core.Tests       PlacementConfigTests
    run_class HartsyInference.LLM.Tests        LlmPlacementTests
    # No checkpoint or GPU needed — rejects a CPU stage mixed into an LLM shard list before any file access.
    run_class HartsyInference.LLM.Tests        TextServiceShardValidationTests
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingTests
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingVramTests
    # HARTSY_KEEP_MODELS is a static-readonly read → each fact of the engine class runs in its own process.
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingEngineTests.DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingEngineTests.DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations HARTSY_KEEP_MODELS=0
    # New Phase A classes land with their implementation parts:
    run_class HartsyInference.Diffusion.Tests  FluxComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  SdxlComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  FluxCfgParallelFallbackTests
    run_class HartsyInference.Diffusion.Tests  SdxlCfgParallelEngineTests
    # FIXED 2026-08-05: two engines co-resident on one ordinal near VRAM capacity forced Flux into
    # block-streaming mode (not enough VRAM to stay resident), which conflicts with step-graph capture —
    # AwaitWeights' cross-stream event wait trips CUDA_ERROR_STREAM_CAPTURE_ISOLATION mid-capture. The abort
    # path correctly ended the capture but never purged the activations/casts CacheActivation'd during that
    # now-discarded capture window, so their graph-private VAs (released the instant the never-instantiated
    # graph was discarded) lingered in GpuTransferHelper's per-backend caches — the next FreeActivations (and,
    # for anything that survived to teardown, Dispose) tried to free them for real and got
    # CUDA_ERROR_INVALID_VALUE. Not actually a concurrency race: it reproduced in this test's SERIAL phase,
    # before either generation ran concurrently — the trigger is VRAM co-residency forcing streaming, not
    # DeviceGate bypass. Fixed by CudaBackend.StepGraphReset calling the new
    # GpuTransferHelper.PurgeAbortedCaptureAllocs on abort, which purges only the entries allocated on the
    # CAPTURING stream (State.CaptureAllocs now records the allocating stream) — the streaming weight cache's
    # own uploads, made on a separate non-capturing stream, are left untouched since they're real allocations
    # regardless of the compute stream's capture outcome. Verified: 7 consecutive real passes (bit-identical
    # concurrent-vs-serial output every time), 2 unrelated environmental interruptions along the way (a
    # concurrent session's transient build collision in unrelated files, and a concurrent session's transient
    # VRAM spike), both resolved cleanly on retry. Re-enabled in the green gate.
    run_class HartsyInference.Diffusion.Tests  SameGpuConcurrentRealWeightTests HARTSY_SAME_GPU_CONCURRENT=1
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingTests
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingVramTests
    # HARTSY_KEEP_MODELS static-init read → each engine fact runs filter-isolated in its own process.
    # KNOWN RED as of 2026-08-05 (N-way sharding generalization pass): this SSIM-vs-unsharded gate fails
    # (~0.176, was documented as 0.9734) for a reason UNRELATED to the N-way generalization — see the
    # 2026-08-05 benchmark doc "Qwen-Image cross-ordinal fidelity finding" section. Root-caused via a same-
    # ordinal-two-backend-instances control (bit-exact, err=0/262144) vs a genuinely-cross-ordinal control
    # (catastrophic at split=41 AND split=59): running this fp8mixed checkpoint's blocks through the 3060's
    # non-native fp8→BF16 dequant GEMM path diverges sharply from the 4090's native fp8 path — the same class
    # of drift MiniMaxH3DitShardingEngineTests already documents and gates informationally (SSIM > 0.05, not
    # a real bar) rather than at 0.75. This test's threshold was NOT changed in this pass (not this task's
    # call) — flagged for the user to decide whether Qwen-Image's gate should be reclassified the same way.
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingEngineTests.DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded
    # NOT RE-EVALUATED in this pass (self-skipped on its own free-VRAM gate — the live swarmui.service was
    # holding the 4090 at the time). No SSIM gate on this fact (VRAM-drift check only), so it is not expected
    # to be affected by the finding above, but that is inference, not a measurement — re-run to confirm.
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingEngineTests.DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations HARTSY_KEEP_MODELS=0
    # NOT RE-EVALUATED in this pass either (same VRAM contention). Both configs it compares carry the same
    # cross-ordinal DiT split, so it is EXPECTED to be affected by the same-class finding above — but that is
    # inference, not a measurement (it compares sharded-vs-sharded, which could plausibly be self-consistent
    # and pass even if each side individually diverges from an unsharded baseline). Re-run to confirm either way.
    run_class HartsyInference.Diffusion.Tests  QwenImageCombinedPlacementShardingEngineTests
    # N-way (Phase 8+) generalization landed 2026-08-05, Qwen-Image only (other 5 sharded families stay 2-way —
    # ROADMAP.md item 7). Mechanism proof: tiny synthetic weights, 3 stages [cuda:0, cuda:1, cuda:0] — TWO
    # boundary crossings, THREE distinct backend instances, tight tolerance (this box has only 2 physical GPUs,
    # so stage 0 and stage 2 share ordinal 0 via a second independent CudaBackend instance — NOT a 3-different-
    # physical-card proof, which stays untested/untestable here).
    run_class HartsyInference.Diffusion.Tests  QwenImageDitSharding3StageTests
    # Real-weight VRAM-pooling proof for the same 3-stage shape: no SSIM gate (deliberately — see the fp8
    # finding above), asserts 3 distinct backend instances hold provably distinct, disjoint block ranges and the
    # forward completes finite.
    run_class HartsyInference.Diffusion.Tests  QwenImageDitSharding3StageVramTests
    run_class HartsyInference.Diffusion.Tests  FluxDitShardingVramTests
    run_class HartsyInference.Diffusion.Tests  FluxDitShardingEngineTests
    run_class HartsyInference.Diffusion.Tests  MiniMaxH3DitShardingTests
    run_class HartsyInference.Diffusion.Tests  MiniMaxH3DitShardingVramTests
    # Full-engine real-generation variant (the 6th sibling's engine test). Cross-device SSIM for THIS model
    # is informational only (fp8 dequant on SM 8.6 vs native fp8 on SM 8.9 legitimately drifts hard on this
    # checkpoint's activation scale — see the benchmarks doc footnote); the test's real gate is same-device
    # split exactness (SSIM 1.0000) plus VRAM pooling.
    run_class HartsyInference.Diffusion.Tests  MiniMaxH3DitShardingEngineTests
    # Audio-LM layer split (YuE Stage-1 un-quantized bf16 pooled across both cards; weights on this box).
    # HARTSY_AUDIO_LM_QUANT is process-wide env the second fact mutates → each fact runs filter-isolated.
    # The primary fact also gates on a Whisper content-word-recall check (real [verse]/[chorus] lyrics).
    run_class HartsyInference.Diffusion.Tests  YueLmShardingEngineTests.LmSharding_RealEngine_UnquantizedStage1_PooledAcrossGpus_ProducesAudio
    run_class HartsyInference.Diffusion.Tests  YueLmShardingEngineTests.LmSharding_EnvQuantOverride_WinsOverShardedDefault
    # Audio-LM layer split, second consumer: CosyVoice 2's Qwen2.5-0.5B backbone (tiny — a pure placement/pooling
    # demonstration, not a can't-fit-one-card case like YuE). Real-weight-verified 2026-08-05: 100% Whisper
    # content-word recall, real VRAM pooling on both cards. Gates on llm.pt/flow.pt/hift.pt (FunAudioLLM/
    # CosyVoice2-0.5B, pulled via `hartsy pull cosyvoice`) + s3gen.safetensors (ResembleAI/chatterbox, frozen
    # S3/CAM++ encoders) + the vendored jfk.wav reference clip. If a box is missing the checkpoint, run
    # `hartsy pull cosyvoice` then hardlink Models/Audio/CosyVoice/{llm.pt,flow.pt,hift.pt} into
    # Models/audio/tts/FunAudioLLM--CosyVoice2-0.5B/ and s3gen.safetensors into
    # Models/audio/tts/ResembleAI--chatterbox/ (the CLI pull path and the runtime AudioModelCache layout
    # differ — same data-layout mismatch YuE hit earlier).
    run_class HartsyInference.Diffusion.Tests  CosyVoiceLmShardingEngineTests
    # Oasis world model: VaeDevice overlaps a finished frame's VAE decode with the next frame's DiT denoise
    # (~3.35 GB checkpoint set, camenduru/oasis-500m mirror .pt->safetensors — see MODEL_STATUS_WORLD.md).
    run_class HartsyInference.World.Tests      OasisVaeDeviceOverlapEngineTests
    # LLM layer-split real-checkpoint parity (Llama-3.2-1B exact-token-match vs unsharded, greedy/fixed-seed).
    run_class HartsyInference.LLM.Tests        LlmShardingEngineTests.Sharded_TwoGpus_ExactTokenParity_VsUnsharded_GreedyFixedSeed
    # VLM layer split (Phase 5, 2026-08-05/06): mllama's gated cross-attention states now peer-copy per stage
    # (GenericTransformer.ForwardEmbedsStaged) and TextService.LoadSharded no longer skips the mmproj sidecar —
    # exact-text-match vs unsharded, greedy/fixed-seed, real image. Both checkpoints already on this box.
    run_class HartsyInference.LLM.Tests        VlmShardingEngineTests.Sharded_TwoGpus_Mllama_ExactTextParity_VsUnsharded_GreedyFixedSeed
    # Splice-style VLM (Qwen2.5-VL-7B): no cross-attention states to copy — the image tokens ride the same
    # embeds sequence text tokens do, so ForwardEmbedsStaged already handled it once the sidecar loaded.
    run_class HartsyInference.LLM.Tests        VlmShardingEngineTests.Sharded_TwoGpus_Qwen25Vl_ExactTextParity_VsUnsharded_GreedyFixedSeed
    # SLOW + HOST-RAM-GATED: needs ~46 GB free host RAM (2.5x the ~18.4 GB Qwen3-32B Q4_K_M file, per
    # TextService.EnsureRamHeadroomFor) REGARDLESS of GPU sharding — the gate is on CPU-side dequantization
    # staging, not final VRAM residency. Will self-skip (→ FAILED under this script's strict gate) whenever
    # swarmui.service or another heavy process is holding host RAM; stop it first if this line fails only
    # on a RAM-skip message, not a real regression.
    run_class HartsyInference.LLM.Tests        LlmShardingEngineTests.Sharded_TwoGpus_Qwen3_32B_DecodeTokPerSec
    # VaeDevice placement for Wan — the checkpoint (wan2.2_ti2v_5B_fp16.safetensors) is already on-disk on
    # this box, so per this campaign's phaseA/phaseB split rule this belongs here, NOT next to its TE twin
    # (WanComponentPlacementEngineTests, phase B) — that placement predates this VAE work and was left
    # as-is rather than reclassified. Wan had zero VAE placement until WanVideoPipeline.VaeBackend landed.
    run_class HartsyInference.Diffusion.Tests  WanVaeComponentPlacementEngineTests
    # SD3.5 Medium DiT sharding (JointBlock loop, real checkpoint on this box — transformer + separate CLIP-L/
    # CLIP-G/T5-XXL side models). Synthetic bit-parity is SAME-GPU by design (cross-SM GEMM/SDPA rounding
    # differs in the last ULP, unrelated to the split logic — see Sd3DitShardingTests' class doc); the VRAM
    # and engine tests below are the real cross-device verification (measured SSIM 0.9794).
    run_class HartsyInference.Diffusion.Tests  Sd3DitShardingTests
    run_class HartsyInference.Diffusion.Tests  Sd3DitShardingVramTests
    run_class HartsyInference.Diffusion.Tests  Sd3DitShardingEngineTests.DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded
    run_class HartsyInference.Diffusion.Tests  Sd3DitShardingEngineTests.DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations HARTSY_KEEP_MODELS=0
    # Lumina-Image-2.0 DiT sharding — SYNTHETIC ONLY: no checkpoint on this box (Models/Stable-Diffusion/Lumina2
    # only has config.json + the safetensors INDEX, no actual weight shards; `hartsy pull lumina2` would fetch
    # them). No VRAM/engine real-weight fact exists yet — add once pulled.
    run_class HartsyInference.Diffusion.Tests  Lumina2DitShardingTests
    # HiDream-I1 DiT sharding — SYNTHETIC ONLY: Models/Stable-Diffusion/HiDream is empty on this box
    # (`hartsy pull hidream` would fetch it; HiDream also needs the embedded Llama-3.1 tokenizer assets — see
    # HiDreamRecipe's construction-time check). [Theory] over 5 split points covering every distinct
    # double/single boundary-crossing shape. No VRAM/engine real-weight fact exists yet — add once pulled.
    run_class HartsyInference.Diffusion.Tests  HiDreamDitShardingTests
}

# ── Phase B: post-download (hartsy pull chroma qwen-image hunyuan-image wan) ─────────────────────────────

phase_b() {
    run_class HartsyInference.Diffusion.Tests  ChromaDitShardingTests
    run_class HartsyInference.Diffusion.Tests  ChromaDitShardingEngineTests.DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded
    run_class HartsyInference.Diffusion.Tests  ChromaDitShardingEngineTests.DitSharding_NonResident HARTSY_KEEP_MODELS=0
    run_class HartsyInference.Diffusion.Tests  HunyuanImageDitShardingTests
    run_class HartsyInference.Diffusion.Tests  HunyuanImageDitShardingEngineTests
    run_class HartsyInference.Video.Tests      CfgBranchParallelWanTests
    run_class HartsyInference.Diffusion.Tests  WanComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  WanCfgParallelEngineTests
    # LTX-1 (ltx-video-2b-v0.9.safetensors, ~9.4 GB, Lightricks/LTX-Video) — needed a download on this box.
    # TE was already wired (LtxVideoRecipePipeline._textBackend); VAE placement is new (LtxVideoPipeline.VaeBackend).
    run_class HartsyInference.Diffusion.Tests  LtxVideoComponentPlacementEngineTests
    # LTX-2.3 (~22 GB transformer-only split, Kijai/LTX2.3_comfy + already-cached video/audio VAE + text-projection
    # side models) — NOT downloaded on this box as of 2026-08-05 (47 GB free, shared with concurrent agents; the
    # DiT alone is ~22 GB). Both TE and VAE placement were already wired in LtxVideo2Pipeline; this closes the
    # "wired but UNVERIFIED" gap docs/MULTI_GPU.md flagged once the checkpoint lands. RealWeightGate skips cleanly
    # until then — `hartsy pull ltx-2` (or download the Kijai split file to
    # Models/Stable-Diffusion/LtxVideo2/ltx-2.3-22b-dev-fp8.safetensors) before running this line.
    run_class HartsyInference.Diffusion.Tests  LtxVideo2ComponentPlacementEngineTests
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
