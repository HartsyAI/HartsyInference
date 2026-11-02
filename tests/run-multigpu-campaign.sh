#!/usr/bin/env bash
# Multi-GPU verification campaign: runs every real-weight placement/sharding/CFG-parallel e2e test class,
# one isolated `dotnet test --filter` per class (never whole suites — Diffusion.Tests has a known pre-existing
# nondeterministic native-heap abort in untagged tests, and HARTSY_KEEP_MODELS-style static-init reads require
# process isolation). Under HARTSY_REQUIRE_REAL_WEIGHTS=1 (set below) a missing checkpoint FAILS the run via
# RealWeightGate instead of silently skipping, so a green exit genuinely means every listed test executed.
#
# Usage: tests/run-multigpu-campaign.sh [phaseA|phaseB|all|list]   (default: phaseA; list = stale-filter check, no GPU)
#   phaseA — the real-weight DiT/LLM/VLM/audio classes in phase_a(): Krea2, Qwen-Image, MiniMax-H3, SD3.5,
#            Chroma HD, Wan (TI2V-5B and the CFG/context-parallel classes), Lumina2/HiDream (synthetic only),
#            YuE, CosyVoice, Oasis, Llama/Mllama/Qwen2.5-VL, plus the CPU and two-GPU NCCL classes.
#            BLOCKED: HunyuanImageDitShardingTests needs Models/Stable-Diffusion/HunyuanImage/hunyuan_image_2_1.safetensors,
#            which is NOT at that path (the box has hunyuanimage21-fp8-* and hunyuanimage2.1-q4_0.gguf). Under strict
#            mode that class FAILS until the file lands; every other phase A class has its checkpoint on disk.
#   phaseB — the Flux placement/fallback/VRAM classes, DitShardingEngineTests (Chroma/Flux/HunyuanImage/Krea2/
#            QwenImage/Sd3 in one Theory) and VideoComponentPlacementEngineTests (LTX-1, LTX-2.3, Wan TE/VAE).
#            BLOCKED: LTX-2.3 is NOT at Models/Stable-Diffusion/LtxVideo2/ltx-2.3-22b-dev-fp8.safetensors, so
#            VideoComponentPlacementEngineTests fails; DitShardingEngineTests also fails on the HunyuanImage
#            Q4_K_M and SD3.5 transformer-only files (per the comment on that line, not re-verified here).
#            HARTSY_REQUIRE_REAL_WEIGHTS=1 is exported below and RealWeightGate THROWS on a missing file, so a
#            blocked class fails loudly (no clean skip).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-phaseA}"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/Output/multigpu-campaign/${STAMP}"
SUMMARY="${LOG_DIR}/summary.txt"
FAILURES=0
RUN_MODE=run   # plan: run_class only records its entry (used by the stale-filter guard)
PLAN=()

log() {
    if [ "${RUN_MODE}" = plan ]; then echo "[campaign] $*"; else echo "[campaign] $*" | tee -a "${SUMMARY}"; fi
}

# ── Pre-flight (GPU box only; runs after the stale-filter guard) ──────────────────────────────────────────

gpu_preflight() {
    mkdir -p "${LOG_DIR}"
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
}

# ── Runner ────────────────────────────────────────────────────────────────────────────────────────────────

# run_class <project-dir-under-tests/> <test-class-name> [extra "ENV=VAL" pairs...]
run_class() {
    local project="$1" cls="$2"
    shift 2
    if [ "${RUN_MODE}" = plan ]; then
        PLAN+=("${project}|${cls}|$*")
        return 0
    fi
    log "── ${cls} (${project}) $*"
    # tests/require-executed.sh owns the checks: exit code, SKIPPED and backend-guard lines, nothing passed. It writes
    # ${cls}.log, ${cls}.trx and ${cls}.executed.md into LOG_DIR.
    if ! "${REPO_ROOT}/tests/require-executed.sh" --project "${REPO_ROOT}/tests/${project}/${project}.csproj" \
        --filter "FullyQualifiedName~${cls}" --out "${LOG_DIR}" --name "${cls}" "$@" -- -v minimal \
        | while IFS= read -r line; do log "   ${line}"; done; then
        FAILURES=$((FAILURES + 1))
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
    # Phase 1 collective transport (2026-08-06): NCCL AllReduce BIT-exact cross-device + AllGather layout +
    # measured bandwidth, plus the host-staged fallback and the duplicate-ordinal factory decision. Synthetic
    # data (no checkpoint) but REAL 2-GPU NCCL communication — libnccl resolves from the standard probe dirs.
    run_class HartsyInference.Cuda.Tests       CollectiveCommTests
    run_class HartsyInference.API.Tests        PlacementPlannerTests
    # Phase 4 auto-strategy planner (2026-08-07): measured-fact rules on synthetic topologies (CPU-only).
    run_class HartsyInference.API.Tests        ParallelPlannerTests
    run_class HartsyInference.Core.Tests       PlacementConfigTests
    run_class HartsyInference.LLM.Tests        LlmPlacementTests
    # Phase 3 tensor parallelism (2026-08-07): CPU synthetic parity (degree-2 vs unstaged, prefill+decode,
    # KV bookkeeping, slice-reassembly incl. Q8_0) + config gating. GPU engine tier further below.
    run_class HartsyInference.LLM.Tests        TensorParallelTests
    run_class HartsyInference.LLM.Tests        TensorParallelPlacementValidationTests
    # No checkpoint or GPU needed — rejects a CPU stage mixed into an LLM shard list before any file access.
    run_class HartsyInference.LLM.Tests        TextServiceShardValidationTests
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingTests
    run_class HartsyInference.Diffusion.Tests  Krea2DitShardingVramTests
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
    # GATED tiers pairing the informational floor above (added 2026-08-06, H3 precedent): same-device split
    # shares one GEMM regime → SSIM > 0.99; HARTSY_FP8_NATIVE=0 matches regimes cross-device (measured
    # 0.9922) → SSIM > 0.95. Env read at backend construction → each fact filter-isolated in its own process.
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingEngineTests.DitSharding_RealEngine_SameDeviceSplit_MatchesUnshardedBaseline
    run_class HartsyInference.Diffusion.Tests  QwenImageDitShardingEngineTests.DitSharding_RealEngine_CrossDevice_MatchedFp8Regime_WithinGatedToleranceOfUnsharded HARTSY_FP8_NATIVE=0
    # Re-run 2026-08-06: measured SSIM 0.9876, comfortably above its 0.75 bar, which stays a REAL gate (not
    # informational) — it compares two sharded runs against each other, so both sides share the same
    # cross-ordinal GEMM regime and the fp8 finding above does not apply.
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
    run_class HartsyInference.Diffusion.Tests  MiniMaxH3DitShardingTests
    run_class HartsyInference.Diffusion.Tests  MiniMaxH3DitShardingVramTests
    # Full-engine real-generation variant (the 6th sibling's engine test). 2026-08-06: the cross-device
    # mosaic was root-caused (LinearImpl's SM<8.9 fallback dropped the Modulate-emitted fp8 input scale) and
    # FIXED — cross-device SSIM went 0.17 → 0.9597 at defaults and is a REAL gate again (> 0.90), alongside
    # same-device split exactness (SSIM 1.0000) and VRAM pooling. See the test's class remarks.
    run_class HartsyInference.Diffusion.Tests  MiniMaxH3DitShardingEngineTests
    # Audio-LM layer split (YuE Stage-1 un-quantized bf16 pooled across both cards; weights on this box).
    # HARTSY_AUDIO_LM_QUANT is process-wide env the second fact mutates → each fact runs filter-isolated.
    # The primary fact also gates on a Whisper content-word-recall check (real [verse]/[chorus] lyrics).
    run_class HartsyInference.Diffusion.Tests  YueLmShardingEngineTests.LmSharding_RealEngine_UnquantizedStage1_PooledAcrossGpus_ProducesAudio
    run_class HartsyInference.Diffusion.Tests  YueLmShardingEngineTests.LmSharding_SettingQuantOverride_WinsOverShardedDefault
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
    # Phase 3 TP engine tier (2026-08-07): real-GGUF degree-2 greedy parity + BOTH-cards shard-VRAM floor
    # + the '[TensorParallel] active' marker assert (discriminates real TP from a silent layer-split).
    run_class HartsyInference.LLM.Tests        LlmTensorParallelEngineTests
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
    # Wan (wan2.2_ti2v_5B_fp16.safetensors) and Chroma (Chroma1-HD-fp8mixed-final.safetensors) are on disk. HunyuanImage
    # is listed here but its test-path checkpoint is missing (see the header). The Wan TE/VAE placement rows live in
    # phase B (VideoComponentPlacementEngineTests) with the other video placement rows.
    run_class HartsyInference.Video.Tests      CfgBranchParallelWanTests
    run_class HartsyInference.Diffusion.Tests  WanCfgParallelEngineTests
    # 2026-08-06 Phase 2: Wan context parallelism (frame-aligned sequence split, replicated weights, per-block
    # K/V exchange). Synthetic parity is same-GPU dual-backend (byte-exact — the mechanism gate); the engine
    # test is the real cross-device run: active SSIM gated > 0.90 (measured 0.9616; NOT 0.99 — cross-ARCH
    # kernel drift on the 4090+3060 pair, see the test's class remarks) or an observable preload fallback.
    # The diagnostic class measures the drift CEILING (whole model per-card, no CP code; measured 0.7774) so
    # the CP number always has its control alongside it.
    run_class HartsyInference.Video.Tests      ContextParallelWanTests
    run_class HartsyInference.Diffusion.Tests  WanContextParallelEngineTests
    run_class HartsyInference.Diffusion.Tests  WanCrossGpuRegimeDiagnosticTests
    # 2026-08-07: Qwen-Image context parallelism (img-row split, replicated weights + replicated txt stream,
    # per-block joint K/V exchange). Synthetic parity is same-GPU dual-backend: single-head fact byte-exact
    # (the mechanics gate); multi-head carries a relative bar for SDPA tiling drift (see class remarks). The
    # engine default-regime SSIM is INFORMATIONAL (fp8 seam: whole-model-on-3060 ~0.14) and active CP is
    # expected UNREACHABLE here (~19 GB replica vs the 3060's 12 GB) — the observable preload-OOM fallback is
    # what gets verified; the matched-regime fact is the gate (filter-isolated: env read once at backend ctor).
    run_class HartsyInference.Diffusion.Tests  ContextParallelQwenImageTests
    run_class HartsyInference.Diffusion.Tests  QwenImageContextParallelEngineTests.ContextParallel_RealEngine_EitherOutcome_ObservableAndCoherent
    run_class HartsyInference.Diffusion.Tests  QwenImageContextParallelEngineTests.ContextParallel_RealEngine_MatchedFp8Regime_WithinGatedToleranceOfBaseline HARTSY_FP8_NATIVE=0
    # Data-parallel serving (throughput): one engine per GPU, same model replicated, concurrent requests via
    # Task.WhenAll — the "serve MORE requests" pattern. Llama-3.2-1B (image DP honestly not demonstrable here:
    # SDXL's F32 replica OOMs the 3060 even at 512^2 — see the test's class doc). Measured 1.71x (2026-08-07).
    run_class HartsyInference.Diffusion.Tests  DataParallelServingEngineTests
    run_class HartsyInference.Diffusion.Tests  ChromaDitShardingTests
    run_class HartsyInference.Diffusion.Tests  HunyuanImageDitShardingTests
    # SD3.5 Medium DiT sharding (JointBlock loop, real checkpoint on this box — transformer + separate CLIP-L/
    # CLIP-G/T5-XXL side models). Synthetic bit-parity is SAME-GPU by design (cross-SM GEMM/SDPA rounding
    # differs in the last ULP, unrelated to the split logic — see Sd3DitShardingTests' class doc); the VRAM
    # and engine tests below are the real cross-device verification (measured SSIM 0.9794).
    run_class HartsyInference.Diffusion.Tests  Sd3DitShardingTests
    run_class HartsyInference.Diffusion.Tests  Sd3DitShardingVramTests
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

# ── Phase B: LTX family (Chroma/HunyuanImage/Wan moved to phase A 2026-08-06 — checkpoints on disk) ──────

phase_b() {
    # Flux dev fp8 is present again at Models/Stable-Diffusion/BFL/Flux1/flux1-dev-fp8.safetensors (checked 2026-10-23);
    # TestPaths.Flux.Dev resolves there via its fallback. Placement kept in phase B; not re-run on GPU.
    run_class HartsyInference.Diffusion.Tests  FluxComponentPlacementEngineTests
    run_class HartsyInference.Diffusion.Tests  FluxCfgParallelFallbackTests
    run_class HartsyInference.Diffusion.Tests  FluxDitShardingVramTests
    # DiT sharding theory (DitShardingEngineTests): Chroma/Flux/HunyuanImage/Krea2/QwenImage/Sd3 cases share one Theory,
    # so each entry runs every case of that fact in one process. Phase B because the HunyuanImage (Q4_K_M), SD3.5
    # transformer-only and LTX-2.3 checkpoints are absent at their test paths (strict mode fails the whole fact). NonResident covers Chroma/Krea2/QwenImage/Sd3 only.
    run_class HartsyInference.Diffusion.Tests  DitShardingEngineTests.DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded
    run_class HartsyInference.Diffusion.Tests  DitShardingEngineTests.DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations HARTSY_KEEP_MODELS=0
    # Video TE/VAE placement, one Theory covering LTX-1, LTX-2.3 and Wan TE/VAE (replaces the per-model
    # LtxVideo/LtxVideo2/WanComponentPlacement/WanVaeComponentPlacement classes). LTX-1 (ltx-video-2b-v0.9.safetensors,
    # ~9.4 GB) is on disk. LTX-2.3 (~22 GB) is NOT at Models/Stable-Diffusion/LtxVideo2/ltx-2.3-22b-dev-fp8.safetensors;
    # a Lightricks LTX-2.3 fp8 file sits at Models/Stable-Diffusion/ with an unverified identity, so the test is not
    # pointed at it. Under strict mode this FAILS LOUDLY until the expected file lands at the test path.
    # TE and VAE placement are wired in LtxVideo2Pipeline (LtxVideoRecipePipeline._textBackend; LtxVideoPipeline.VaeBackend).
    run_class HartsyInference.Diffusion.Tests  VideoComponentPlacementEngineTests
}

run_phase() {
    case "${PHASE}" in
        phaseA) phase_a ;;
        phaseB) phase_b ;;
        all)    phase_a; phase_b ;;
        list)   phase_a; phase_b ;;
        *) echo "unknown phase '${PHASE}' (use phaseA|phaseB|all|list)"; exit 1 ;;
    esac
}

# ── Stale-filter guard: every run_class filter must match >=1 test (dotnet --list-tests, no GPU) ──────────

validate_filters() {
    local entry project cls rest out build_out rc count bad=0 hosterr=0 built=" "
    RUN_MODE=plan
    PLAN=()
    run_phase
    RUN_MODE=run
    for entry in "${PLAN[@]}"; do
        IFS='|' read -r project cls rest <<< "${entry}"
        if [[ "${built}" != *" ${project} "* ]]; then
            if ! build_out="$( cd "${REPO_ROOT}" && dotnet build "tests/${project}/${project}.csproj" --nologo -v quiet 2>&1 )"; then
                echo "[campaign] build failed for ${project}; last output:"
                printf '%s\n' "${build_out}" | tail -40
                exit 1
            fi
            built+="${project} "
        fi
        out="$( cd "${REPO_ROOT}" && dotnet test "tests/${project}/${project}.csproj" --no-build --list-tests \
            --filter "FullyQualifiedName~${cls}" --nologo 2>&1 )"
        rc=$?
        # The "available tests" header only appears when the test host actually listed tests. Without it the
        # failure is the host (or the build), not a filter that matches nothing.
        if ! printf '%s\n' "${out}" | grep -q "The following Tests are available:"; then
            echo "[campaign] TEST HOST ERROR: could not list tests for ${cls} (${project}); dotnet test exit ${rc}. Last output:"
            printf '%s\n' "${out}" | tail -20
            hosterr=$((hosterr + 1))
            continue
        fi
        count=$(printf '%s\n' "${out}" | sed -n '/The following Tests are available:/,$p' | grep -cE '^\s+\S')
        if [ "${count}" -eq 0 ]; then
            echo "[campaign] STALE FILTER: ${cls} (${project}) matches zero tests"
            bad=$((bad + 1))
        fi
    done
    if [ "${hosterr}" -ne 0 ]; then
        echo "[campaign] ${hosterr} test-host error(s) while listing tests. This is NOT a stale filter. Nothing was run."
        exit 1
    fi
    if [ "${bad}" -ne 0 ]; then
        echo "[campaign] ${bad} stale filter(s); fix tests/run-multigpu-campaign.sh. Nothing was run."
        exit 1
    fi
    echo "[campaign] guard: all ${#PLAN[@]} filters match at least one test"
}

validate_filters
[ "${PHASE}" = list ] && exit 0
gpu_preflight
RUN_MODE=run
run_phase

# ── Summary ───────────────────────────────────────────────────────────────────────────────────────────────

log ""
if [ ${FAILURES} -eq 0 ]; then
    log "CAMPAIGN GREEN — every listed class executed and passed with real weights required. Logs: ${LOG_DIR}"
    exit 0
fi
log "CAMPAIGN FAILED — ${FAILURES} class(es) failed. Logs: ${LOG_DIR}"
exit 1
