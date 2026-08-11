# HartsyInference — Engineering Roadmap

> The single forward-looking checklist for the whole engine: cross-cutting features and campaigns that
> aren't tied to one modality. **Per-model open work lives in the matching `MODEL_STATUS_*` doc**; this
> file is for infrastructure that spans modalities — multi-GPU, GPU-vendor support, kernel/quant perf,
> serving/CLI/API, and release. Only *open* work belongs here; when an item ships, move it to the
> relevant status doc's "done" evidence and delete the line. Distilled from the old `MULTI_GPU_*`,
> `PHASE_B_GPU_PERFORMANCE`, `*_PERF_GRIND`, `*_BENCHMARK`, `QUANT_GEMM_*`, `W8A8_HANDOFF`,
> `PRODUCTION_RELEASE_CRITERIA`, and `RELEASE_NUGET` docs (recoverable from git history).

**Legend:** `[ ]` open · `[~]` in progress · `[x]` done (delete on move) · 🔒 blocked (reason noted).

---

## 1. Multi-GPU / model sharding (headline feature)

Not "one gen per GPU" — a single large model **spread across GPUs**: big LLM/MoE servers, and stringing
together several small consumer cards (e.g. 8 GB each) into one usable pool. Unblocks the build-deferred
giants (Kimi-K2, DeepSeek-V3, Mixtral, Qwen3-MoE, Qwen2.5-VL-7B).

- [x] **M0 — device topology + placement** — **done** (2026-08-02): `CudaTopology.Probe()` (per-device
  name/VRAM/CC), `PlacementConfig` (shard devices/ratios, TE/VAE/CFG-parallel device, all-defaults =
  byte-identical to pre-plan behavior), `PlacementPlanner` (`LlmSplitPlan` proportional-to-free-VRAM,
  `DiffusionAutoPlan`), `InferenceEngine` backend pool keyed by selector.
- [x] **M1 — layer-split (pipeline) parallel** — **done** (2026-08-02): `LlmPlacement`/`LlmStage`,
  `GenericTransformer.EnumerateStageWeights`/`ForwardEmbedsStaged`, per-stage VRAM budgeting in
  `TextGenerationPipeline`. Verified on the 4090+3060 dev box: Llama-3.2-1B split 8/8 across both cards
  gives **exact token parity** vs single-GPU with VRAM genuinely pooled (2058 MB on one card → 528 MB +
  796 MB). v1 gaps (documented, not silent): per-stage CUDA graph decode disabled (eager only), SSM-family
  excluded (host-recurrence incompatible with staging). mllama's gated cross-attention layers were also
  excluded in v1 — **unblocked 2026-08-06**: `ForwardEmbedsStaged` now peer-copies the vision
  cross-attention states onto every stage that owns one (see `steady-brass-heron.md` item 5).
- [x] **Same-GPU multi-backend support** (not in the original M0–M5 taxonomy — a real gap it missed:
  "many small cards" and "one big card, two tenants" are different problems) — **done** (2026-08-02):
  per-backend `State` keyed by a process-unique token (not context handle) so two `CudaBackend` instances
  can share one device with isolated streams/caches/mempool-threshold and guarded disposal;
  `Engine/DeviceGate.cs` serializes concurrent generations per ordinal by default
  (`HARTSY_SAME_GPU_CONCURRENT=1` opts into the audited-safe concurrent path). 8/8 same-device isolation
  tests pass, including interleaved-op and step-graph-capture cases.
- [x] **Peer-copy layer** — **done** (2026-08-02): `cuDeviceCanAccessPeer`/`cuCtxEnablePeerAccess`/
  `cuMemcpyPeer{,Async}` bound in `CudaDriverApi.cs` (the "Missing bindings" this doc's Implementation
  Notes section used to list — no longer missing); `IBackend.CopyFromPeer` with a P2P path and a
  host-staged fallback for consumer hardware without P2P (`HARTSY_P2P_DISABLE=1` forces the fallback
  deterministically for testing); `CudaPeerAccess` per-pair probe/enable memo.
- [x] **Diffusion component placement (TE/VAE on another GPU)** — **done** (2026-08-02, extended
  2026-08-04): `RecipeContext` gained `TextEncoderBackend`/`VaeBackend`, wired for Wan (umT5) → Flux
  (T5/CLIP + VAE) → SDXL (CLIP), then Qwen-Image, Chroma, HunyuanImage, LTX-1, LTX-2 (audio VAE + vocoder
  included) — see `MULTI_GPU_COMPONENT_PLACEMENT.md` for the per-pipeline hazards (pin owners, the
  HunyuanImage device-resident latent bridge, LTX-2's Gemma-evict skip). Host-materialization at the
  existing stage boundaries means no peer copy is needed. **Measured on real hardware 2026-08-04**:
  `FluxComponentPlacementEngineTests` (real dev fp8, TE+VAE on the second card, SSIM 0.81 vs baseline —
  the mismatched-SM fp8 T5 paths legitimately drift; on matched cards the expectation stays bit-identical)
  through the full `InferenceEngine` path. `VaeDevice` became settable from the extension (`VaeGpuId`)
  and the CLI (`--vae-gpu`, `--te-gpu`).
- [x] **CFG-branch parallelism** (Wan first, then Flux true-CFG) — **done** (2026-08-02):
  `Diffusion/Utilities/CfgBranchRunner.cs` (dedicated background `Thread`, not `Task.Run` — a
  seconds-long blocking GPU call must not invite thread-pool injection stalls); `PlacementConfig.CfgParallelDevice`
  → `RecipeContext.CfgParallelBackend` → `DiffusionPipelineBase.CfgParallelBackend`, resolved in
  `InferenceEngine.cs` at both recipe-construction sites; extension setting `CfgParallelGpuId`
  (`HartsyInferenceBackend.cs`) mirroring `TextEncoderGpuId`. **Wan** (`WanVideoPipeline.cs`, the T2V/TI2V
  loop matching the on-disk checkpoint): weights preloaded on both backends before the loop (load-bearing
  for correctness — makes every in-loop weight read a per-backend cache-hit, not just a perf win), the
  one shared mutable tensor (the per-step latent) cloned before the fork, both backends' activations freed
  per step, MoE (A14B) falls back to sequential (`SwapToExpert` is Backend-only). **Verified**: same-GPU
  (two `CudaBackend` on one ordinal) bit-parity vs sequential on a synthetic tiny Wan config, real CUDA
  kernels, `HARTSY_ASSERT_AMBIENT=1` clean (`CfgBranchParallelWanTests.cs`). **Flux** (`FluxPipeline.cs`,
  the `drainFree` true-CFG branch): structurally excludes block-streaming (the streaming controller's
  `BeforeBlockForward` hook is a single field on the shared transformer — two backends' controllers can't
  both drive it), gated on the ACTUAL per-generation resident-vs-streaming placement decision (VRAM
  pressure varies run to run, so this can't be a static config check), falls back to sequential silently
  (one log line, never a throw) if the second backend can't also hold the whole DiT resident. Uses
  `CopyFromPeer` (not `.DataPointer`) for both the latent hand-off and the velocity hand-back, preserving
  `drainFree`'s whole point (device-resident latent, no per-step D2H). **Update 2026-08-04**: a real Flux
  dev fp8 checkpoint now lives at `Models/Stable-Diffusion/BFL/Flux1/` (the earlier "no checkpoint on this
  box" note is stale) and real-weight Flux verification runs through `FluxDitShardingVramTests`/
  `FluxComponentPlacementEngineTests`. On this 4090+3060 box the CFG-parallel CONCURRENT path stays
  untestable for Flux (the ~12 GB fp8 DiT cannot replicate onto the 12 GB card) — the fallback path is what
  is verified, now observably: every CFG-parallel decision is recorded via
  `DiffusionPipelineBase.LastCfgParallelDecision` + a `[CfgParallel]` log line (`active`,
  `fell-back(<reason>)`, `inapplicable(no-true-cfg)`), asserted by `FluxCfgParallelFallbackTests`. Wan's
  CfgParallel preload additionally gained the missing OOM→sequential fallback (was: a too-small second
  card killed the generation).
  **Found + fixed along the way** (pre-existing, not scoped to this feature — CFG-parallel is just what
  surfaced it, since it's the first code path to have two threads genuinely touch a shared tensor at the
  same wall-clock time): a real double-free race in `Tensor.EnsureCpuData` (`src/HartsyInference.Core/Tensors/Tensor.cs`)
  — the primary GPU-binding slot was read/cleared without the lock every other mutator uses. Fixed with a
  claim-under-lock/invoke-outside pattern matching the file's own `TakeExtraBindings`; regression tests in
  `tests/HartsyInference.Core.Tests/TensorConcurrentSyncTests.cs` reproduce the double-invoke on the old
  code and confirm the fix. Confirmed independently live TODAY (no CFG-parallel needed) via `ReduxResolver`'s
  process-wide cached SigLIP/projector weights, reachable by two ordinary concurrent one-backend-per-GPU
  generations sharing a style model.
- [x] **DiT sharding, experimental** (block-range split across GPUs, VRAM-pooling not latency) — Flux
  investigation 2026-08-02 **paused before writing code**: `FluxTransformer.Forward` threads F16 loop-mode
  casting, ControlNet residual injection, and Kontext ref-token bookkeeping through the same block loops a
  split would need to isolate, and no Flux checkpoint was on this box to verify against. **Retargeted to
  `Krea2Transformer`** (checkpoint already on disk; genuinely simpler — no ControlNet, no Kontext) and
  shipped 2026-08-02: `ForwardEmbedIn`/`ForwardBlocksRange`/`ForwardHeadOut` extracted from `ForwardCore`;
  new `ForwardSharded`/`ForwardPatchedSharded` run blocks `[0,splitBlock)` on backend A and
  `[splitBlock,BlockCount)` on backend B, handing the joint activation + `tembMod` across via `CopyFromPeer`
  at the boundary and back for the head. New `EnumerateBlockRangeWeights(start,end)` for the ASYMMETRIC
  preload the VRAM-pooling claim depends on (A: shared + its range; B: its range ONLY — never the full
  `EnumerateWeights()` on both, which would replicate instead of pool). Always eager: no step-graph route
  (bakes one context's pointers) and no step-cache (its block-0-indicator shape doesn't compose with a
  fixed boundary) — same narrowing precedent as excluding block streaming and Wan. Verified two ways: (1)
  same-GPU bit-parity — 4-block synthetic config, split at block 2, cross-device (ordinal 0+1), compared
  bit-for-bit against unsharded `Forward` with identically-seeded weights — 0 mismatches (F16/step-graph
  off ⇒ deterministic F32, exact-equality bar, not tolerance); (2) real Krea2 Turbo fp8 checkpoint (~13 GB),
  split at block 14 across the 3060+4090, `Context.GetMemoryInfo()` before/after preload —
  **6.65 GB + 5.69 GB = 12.34 GB resident, neither card holding the whole checkpoint** — real VRAM pooling,
  not "it ran." Found and fixed along the way (see `FluxRope.cs`): a live thread-safety bug in already-shipped
  Phase 7 Flux CFG-parallel — `FluxRope`'s GPU cos/sin table cache was a single unkeyed slot shared by both
  cond/uncond backends; fixed with a per-backend-keyed, locked cache, plus a `condTxtSeqLen == txtSeqLen`
  guard in `FluxPipeline` closing a signature-mismatch corruption path the lock alone couldn't.
  **Update 2026-08-04 — sharding is now a fleet feature, wired end-to-end (pipeline + recipe + planner)
  and hardware-verified per model**: Krea2 (the 2026-08-03 engine e2e), **Qwen-Image** (20B Edit fp8:
  19.6 GB pooled 13.4+6.2 across 4090+3060 at a live 41/60 split; engine e2e SSIM 0.9734 vs unsharded;
  same-device split BIT-EXACT on the synthetic config; a per-backend `QwenImageRope` table cache fixed the
  cross-backend host-tensor staging crash), **MiniMax-H3** (fp8 pruned 19.76 GB pooled 13.9+5.8 at a live
  34/50 split with finite video+audio output — the headline "cannot fit one card" case; fp8-build-only,
  bf16 66 GB exceeds any 2-consumer-card pool by design), **Flux v1 plain-path** (real dev fp8: same-device
  split bit-exact over 262k velocity values on the default F16 hot path; cross-device byte-weighted 30/57
  split pools 7.7+3.7 GB; engine e2e SSIM 0.9075; ControlNet/Kontext/inpaint/regional generations fall back
  to unsharded with a logged warning), with Chroma + HunyuanImage in flight on the same template. Planner
  gained a byte-weighted `DitSplitPlan` overload (heterogeneous double/single block sizes — a count split
  misallocates by GBs on Flux/Chroma/Hunyuan). Cross-device splits are tolerance-gated, same-device splits
  bit-exact: cross-SM cuBLAS reduction order legitimately differs (the Krea2 cross-device bit-exact result
  was tiny-GEMM luck, not a general guarantee).
- [x] **Multi-GPU verification campaign** — added 2026-08-04: `tests/run-multigpu-campaign.sh` runs every
  placement/sharding/CFG-parallel real-weight e2e class filter-isolated (never whole suites) under
  `HARTSY_REQUIRE_REAL_WEIGHTS=1`, where a missing checkpoint FAILS via `RealWeightGate` instead of
  silently skipping — a green campaign genuinely means every listed test executed on real weights.
  `CudaOrdinalMapTests` prints the live ordinal→card map first (CUDA enumerates fastest-first; on this box
  ordinal 0 = 4090, 1 = 3060 — REVERSED from nvidia-smi).
- [x] **Audio-LM layer split (YuE Stage-1)** — added 2026-08-05: the first audio consumer of M1's
  machinery. `Qwen2Model`/`YueStage1Lm` gained `ForwardStaged`/`EnumerateStageWeights` passthroughs;
  `MusicService` builds a `MusicLoadContext` (shard backends + quant policy) from the engine
  `PlacementConfig.ShardDevices`; the load-time Q4_K quantization became a policy
  (`HARTSY_AUDIO_LM_QUANT=q4k|q8|off`, auto = Q4_K single-device / **un-quantized when sharded**).
  Verified on real weights (`YueLmShardingEngineTests`, campaign phase A): bf16 7B Stage-1 pooled at
  8.7 + 4.3 GB across 4090+3060, full 15 s WAV, both cards' VRAM rise asserted in-test. CLI
  `--lm-shard-gpu`, extension `LmShardGpuId` (ShardDevices without the DiT flag). Still bespoke-runner
  audio LMs NOT covered: HeartMuLa (CSM-shaped), YuE Stage-2 (1B, no need), MusicGen; and
  `AudioRuntime`'s eviction strategy remains single-backend-minded (see §1 open items).
- **Two finished features deliberately still opt-in** (plain-language, for the flip-it-later decision):
  `HARTSY_SAME_GPU_CONCURRENT=1` lets two backends sharing one physical GPU run generations at the same
  time instead of taking turns — **update 2026-08-06: the VRAM-capacity failure was root-caused (a
  step-graph-capture-abort virtual-address leak, NOT a concurrency race), fixed via
  `GpuTransferHelper.PurgeAbortedCaptureAllocs`, and `SameGpuConcurrentRealWeightTests` is back in the
  campaign green gate. Still opt-in pending longer soak.**
  `HARTSY_KV_F16=1` halves LLM conversation-memory VRAM by storing the KV cache at half precision —
  bit-verified for short generations, can pick different (equally valid) words in very long ones.
  Flipping either default is a one-line change; nothing else in the multi-GPU work depends on them.
- [x] **Phase 0 hardening (2026-08-06):** a 6-agent full-code audit of the multi-GPU implementation
  confirmed the core mechanisms sound and surfaced 7 P1 defects, all fixed + verified same day:
  same-device `CfgParallelDevice` now nulls out loudly (two branch threads must never share a backend);
  sharded-LLM reload no longer leaks per-stage CUDA contexts; staged GPT-2/StarCoder/BLOOM embedding
  preamble now first-stage-only (pinned by `LlmPlacementTests`); model-switch eviction sweeps placement
  backends (was stacking full DiT replicas on GPU B per switch); CosyVoice sets `HighPrecisionGemm` on
  every shard stage; `DeviceGate` now gates EVERY placement ordinal (ascending-order multi-acquire);
  Flux CFG-parallel cached negatives host-materialize on the TE-cache-hit path too. Test gates
  hardened: Qwen-Image regained GATED tiers (same-device split SSIM 1.0000 > 0.99; matched-fp8-regime
  cross-device 0.9929 > 0.95); LLM/VLM/CosyVoice VRAM-pooling floors raised from context-noise level to
  model-share-derived bounds. See `benchmarks/results/2026-08-05_multigpu_speeds.md` (2026-08-06
  sections).
  **Documented-deferred from the same audit** (recorded here so nobody re-finds them): GameCraft loader
  lacks the BF16-cast/`DisableCacheWeightCasts` handling `HunyuanVideoRecipe` documents as load-bearing
  (unverifiable until the ~51 GB checkpoint is pulled); `CrossAttentionBlock` two-slot K/V cache has no
  pinning/teardown (safe under current per-pipeline construction; becomes real work if components are
  ever shared or mid-loop `FreeActivations` added); mllama staged decode re-peer-copies vision features
  every token (~100 MB/token cross-stage — a persistent per-stage cache is a measured follow-up);
  CosyVoice's tied Qwen embed table (~544 MB) uploads to the last shard stage for a head that never
  runs (needs a "headless driver" flag); `LlmSplitPlan` never charges lm_head bytes to the last stage
  (`lastStageExtraBytes` exists but isn't passed); ~800 lines of near-duplicate `*ShardingEngineTests`
  could parameterize.
- [ ] **M2 — tensor parallel:** split individual GEMMs (QKV, MLP, lm_head) across devices; all-reduce on
  the seam. Plan-level only (`NcclApi` design in `MULTI_GPU_PARALLELISM.md`); needs NVLink hardware this
  box doesn't have to pay off.
- [ ] **M3 — expert parallel (MoE):** route experts to devices; ties to on-device MoE routing (§4).
- [~] **M2 — tensor parallel v1 SHIPPED (Phase 3, 2026-08-07):** `TensorParallelDegree` consumed for
  real — Megatron-style column/row split (quant-block-aligned), 2 NCCL all-reduces/layer via the Phase 1
  collectives, per-rank KV head slices, TextService branch with the asserted `[TensorParallel] active`
  marker (precedes layer-split so a TP config can never silently layer-split). Verified: 17/17 CPU
  synthetic parity + real-GGUF degree-2 EXACT token parity on 4090+3060 (+3,511/+1,730 MiB shards).
  v1 bounds: dense Llama/Qwen2/Qwen3 only (loud refusals), single driving thread, no graph/spec decode —
  perf on PCIe is pre-declared honest (correctness + harness, not a speed claim). Follow-ups: threaded
  RankRunner driver, per-token comm profiling, more architectures.
- [~] **M4 — sequence/context parallel (Phase 2, 2026-08-06): Wan v1 LANDED.**
  `PlacementConfig.ContextParallelDevices` → frame-aligned proportional token split, per-block
  self-attention K/V exchange (2-rank two-phase barrier, host-assembled), weights replicated. Mechanism
  byte-exact (synthetic `ContextParallelWanTests`); real-weight cross-device SSIM 0.9616 gated > 0.90
  (cross-ARCH drift ceiling on this 4090+3060 pair measured 0.7774 via `WanCrossGpuRegimeDiagnosticTests`
  — regime flags can't equalize architectures, so 0.99 is unreachable on heterogeneous cards by physics,
  not by defect). HONEST perf: slower at the 675-token test geometry (exchange/imbalance-bound); the win
  case is long sequences on balanced links. **Update 2026-08-07:** Qwen-Image CP shipped (img-row split,
  replicated txt stream; on this box the 19 GB replica can't fit the 3060 → the verified behavior is the
  observable preload-OOM fallback, single-GPU completion, SSIM 1.0000; active CP needs a same-VRAM pair);
  CLI `--cp-gpu` shipped; the large-geometry perf point was MEASURED via CLI (832×480×25f: CP 3.5× slower
  than single-GPU — on a no-P2P heterogeneous pair Wan CP loses at every geometry; mechanism correct,
  the latency win needs NVLink-class links + balanced GPUs); data-parallel serving pattern pinned by
  `DataParallelServingEngineTests` (1.71× for 4 concurrent requests). Not yet: >2 ranks, NCCL-backed
  exchange, Ulysses head-parallel variant, CP×CFG-parallel composition, threaded TP driver.
- [ ] **M5 — disaggregated serving:** separate prefill and decode pools.
- [x] **Collectives (Phase 1, 2026-08-06):** `NcclApi` P/Invoke (runtime-resolved libnccl.so.2, no system
  install — torch-venv copy hardlinked into the probe dir), `ICollectiveComm` with `NcclComm` +
  `HostStagedComm` fallback and a logged factory decision, `CudaTopology.ProbeLinks()` P2P/NVLink matrix.
  Verified on real GPUs: cross-device AllReduce BIT-exact, 256 MB AllGather 4.79 GB/s (SHM transport,
  no-P2P box). RCCL stays a library-swap possibility (symbol-compatible), untested — no AMD hardware.
- [~] **Tiers to validate:** consumer (4090+3060, no P2P) verified for M0/M1/same-GPU above; enterprise
  (multi-datacenter-GPU, P2P/NVLink) still unvalidated — no such hardware available to this session.
- [ ] **Diffusion D1–D3:** cross-device latent/sequence parallel for video (Wan/LTX) once M4 lands.

## 2. GPU kernel performance

The dedicated benchmark harness is greenfield — build it first so every optimization is measured, not
assumed (see the profiling pitfalls in `TROUBLESHOOTING.md`).

- [ ] **Benchmark harness:** C# runner + PyTorch/llama.cpp baselines + NVTX ranges + run scripts;
  multi-device baselines (3060 / L40S / A100 / H100).
- [ ] **FlashAttention-2 PTX** (varlen/packed): `IBackend.PackedAttention` for video; general FA2 kernel.
- [ ] **cuDNN-free Winograd conv** for the conv-heavy VAEs/vision. Evaluated as a Vulkan-side "reverse gap"
  opportunity too (2026-07-29, see §3 Phase 5) since neither backend has one — **deferred on both**: only
  3x3-stride-1 convs benefit (~41% of this engine's `Conv2D` call sites, per a grep across
  `src/HartsyInference.Diffusion`), concentrated in the VAE + classic SD1.5/SDXL UNet path; DiT models
  (Flux, SD3.5 — the newer direction) call `Conv2D` once per forward pass as a strided patchify op, which
  Winograd can't accelerate at all (it doesn't support stride > 1). No profiling on either backend shows
  Conv2D as an actual bottleneck today (CUDA's own SDXL wins came from residency/reshape fixes, not conv
  algorithm choice — see `MODEL_STATUS_IMAGE.md`). Revisit only if real profiling flags VAE/UNet conv as a
  measured hotspot.
- [ ] **Kernel fusion** (QKV projection, gate-up, norm+activation, coopmat bias) where launch-bound.
- [ ] **Activation memory pool** + **CUDA-graph denoise** capture across the DiT/UNet loop.
- [ ] **F16/BF16 tensor-core sweep**; **FA3 / WGMMA** on Hopper.
- [x] **LLM decode: F16-storage KV cache** (`HARTSY_KV_F16=1`, opt-in — halves KV VRAM, the actual fix
  for 12 GB cards OOM'ing) — **done** (2026-08-02): two new CUDA kernels, `lm_kv_append_f16` (K/V
  projection stays F32; converts on write) and `lm_flash_attn_f16kv_f32` (upconverts to F32 on load —
  Q/scores/softmax/accumulator all stay F32, this is a storage/bandwidth change, not a numerically new
  kernel). `FixedKvCache` gained a `kvDtype` param (default F32, unchanged); `KvCaches.F16Enabled` is the
  one place the env var is read, threaded to every direct `FixedKvCache` construction site EXCEPT the
  CUDA-graph-decode path, which stays F32 on purpose (`FlashAttentionDev` refuses F16 KV structurally —
  its split-K/graph fast paths have no F16 variant in v1, so engaging them falls back to the monolithic
  eager kernel silently rather than corrupting or crashing). Verified: kernel-level round-trip + tolerance
  tests (`KvF16StorageTests.cs`, includes a split-K-force stress case); real-weight Llama-3.2-1B —
  short generation byte-identical to F32, longer generation diverges partway through (expected: F16 is
  ~3 decimal digits, greedy decoding cascades after any near-tied argmax flip — output stays fully
  coherent both sides, F16 mode itself is deterministic across repeated runs, ruling out a race/bug); VRAM
  delta measured 960 MiB saved at 30K max-seq-len vs a ~938 MiB theoretical prediction (16 layers × 8 KV
  heads × 64 head-dim × 2 bytes saved × 2 (K+V)) — matches within noise. Default stays F32 (same
  soak-before-flip precedent as `DeviceGate`'s `HARTSY_SAME_GPU_CONCURRENT`).
- [ ] **Video shared-infra:** `DenoiseKvCache` (~2–3×), `DistilledFlowMatchEuler` (DMD/CM/Lightning few-step),
  `IDiscreteVideoTokenizer` (Cosmos), sparse video attention (Wan/LTX — measure-then-design).
- [ ] **Vision GPU-native ops:** `MaxPool2D`, `Conv2dDepthwise` (currently CPU); JPEG/WebP decoders.
- [ ] **3D residuals:** GPU marching cubes; DINO per-op fusion.
- [ ] **Kernel-source hygiene (maintainability):** ~28 shipped CUDA kernels (`softmax`, `groupnorm`,
  `layernorm`, `geglu`, `cast_*`, `elementwise_*`, `transpose`, `broadcast_add`, `spatial`) are hand-written
  PTX with no `.cu` source. Backfill a `.cu` for each opportunistically (verify PTX parity, then delete the
  hand-PTX); keep `hgemm_mma_sm80` hand-authored (intentional MMA). See `src/HartsyInference.Cuda/Kernels/README.md`.

## 3. AMD / ROCm + cross-vendor (Vulkan) support

The Vulkan backend exists (SPIR-V/GLSL) but is unvalidated on AMD/Intel hardware — that validation is the
whole point of the backend. See the Vulkan pitfalls in `TROUBLESHOOTING.md`. **A working SPIR-V compiler is
now available** (2026-07-28): Ubuntu's `glslang-tools` apt package installs but cannot compile
`matmul_int8.comp.glsl` (no `GL_EXT_integer_dot_product` in its GLSL frontend); use the LunarG Vulkan SDK's
bundled `glslangValidator`/`spirv-val` instead (extract-and-run, no install needed — see
`TROUBLESHOOTING.md`). A full rebuild of `src/HartsyInference.Vulkan/Spirv/*.spv` reproduced all prior kernels
byte-identical and additionally fixed `MaxPool2D`/`Conv2dDepthwise` (dispatched shader names with no
compiled artifact — now built) and wired up `Snake`/`Conv1d`/`ConvTranspose1d` (source existed, was never
built or connected in `VulkanBackend.cs`; `ConvTranspose1d` is dense/groups=1 only — depthwise falls back
to the CPU backend with a clear throw). `Tanh`/`Elu` were already correct in the committed `.spv`, despite
a stale code comment claiming otherwise.

- [ ] **AMD/Intel cross-vendor bring-up** on real hardware (🔒 needs a dual-vendor box); the "Anticipated
  Categories" AMD/Intel row is unvalidated. Mesa llvmpipe (software Vulkan, subgroup 8) is available on
  NVIDIA-only boxes as a substitute for small-subgroup *correctness* checks, not a perf substitute.
- [ ] **Vulkan kernel/perf tuning:** the old "~6.5× CUDA / ≤1.6× target" figures had no benchmark
  artifact anywhere in the repo. **Now measured** (2026-07-28, RTX 4090, see
  `benchmarks/scoreboards/VULKAN.md`): GEMM is **~30×–160× slower** depending on shape/dtype — far worse
  than the old claim — with F32 (no TF32-equivalent tensor-core path on Vulkan) the worst offender, but
  F16 (which should hit `matmul_coopmat`) also 30–157× slower, meaning either coopmat isn't actually
  engaging for these shapes or its real throughput is far below cuBLAS's tuned tensor-core GEMM — needs
  a `HARTSYINFERENCE_VK_PROFILE=1` check before any further tuning. **Answered definitively (2026-07-31,
  §3's coopmat-engagement entries below):** it's the latter. Coopmat engagement on a real Krea2 run was
  raised from 0% to 84.8% (the M/N/K-alignment gap that blocked it is now closed — see below), and it
  bought ~0 measured wall-clock (a same-shape A/B: 78.9s coopmat-off vs. 80.5s at 60.7% engagement — noise,
  not a real difference). The GEMM ceiling really is coopmat's own per-op throughput, not an engagement
  problem — tile size, register blocking, and coopmat2 (below) are the actual remaining levers, not
  anything about WHETHER coopmat runs. **A second finding was found AND
  fixed in the same pass:** Silu/Gelu/RmsNorm/LayerNorm showed a sharply non-linear cliff at ~5.24M
  elements (Silu: 956 μs at 1.31M elements → 29,652 μs at 5.24M elements, a 31× jump for a 4× size
  increase) that `BroadcastAdd` (writes in place, no fresh output allocation) did NOT show at the same
  sizes. Root cause: `VulkanMemoryAllocator` destroyed any "dedicated" (>= 16 MB) block the instant it
  emptied instead of pooling it like slab blocks — every >= 16 MB transient buffer paid a real
  `vkAllocateMemory`/`vkFreeMemory` round trip per dispatch. **Fixed** (2026-07-28, see
  `TROUBLESHOOTING.md`): dedicated blocks are now pooled exactly like slabs; the 5.24M-element case
  dropped from 29,652 μs to 4,420 μs (~7×), regression-guarded by
  `VulkanLeakTests.Vulkan_100Iter_LargeTransient_PoolsInsteadOfReallocating`. The remaining ~27× gap at
  that size is now a normal (roughly-linear-with-size) kernel/dispatch throughput gap, folded into the
  GEMM-class finding above rather than a separate pathology. The
  isolated-Linear per-dispatch-overhead finding (~94% of Linear time, `VulkanLinearProfileMeasurement`)
  and the GPU-residency question (CPU-loop-default `IBackend` fallthrough) are now both instrumented AND
  largely closed for the LLM-decode-step shape: the synthetic step
  (`Measure_LlmDecodeStep_ResidencyVsDispatchOverhead`) started at 2.58 ms/step with 5.0 D2H syncs +
  10.0 transfer-cache misses per step; wiring real GPU dispatches for `SliceLastDim`/`ApplyRope`/
  `KvCacheAppend` (none had a `VulkanBackend` override before — every call fell through to `IBackend`'s
  CPU-loop default) plus a device-to-device fast path for `CopyTo` (new: `TryGetCached` peeks the
  weight/activation cache without forcing an upload, so a GPU-resident source copies via
  `vkCmdCopyBuffer` instead of a D2H-sync-then-H2D-reupload round trip) dropped it to **1.38 ms/step
  (~1.87×) with 0 D2H syncs and 4.0 misses/step** — the remaining misses are genuinely-always-host
  per-step inputs (the token, the RoPE cos/sin table) that need Phase 6's device-resident RoPE table to
  close, not more residency work here. Still no real head-to-head full-pipeline number vs CUDA — that's
  a Phase 4+ target once flash attention exists (the current naive SDPA dominates any real model's step
  time enough that the residency win above wouldn't show through end-to-end yet). Remaining levers once
  the coopmat gap is understood: QKV fusion, pre-cast FP8 weights, coopmat bias fusion, vendor
  tile-size auto-tuner.
- [x] **GGUF dequant shaders** — **done** (2026-07-29): all 6 (`dequant_q4_0/q5_0/q8_0/q4_k/q5_k/q6_k`),
  mirroring `native/cuda/dequant/*.cu` exactly, wired into `VulkanBackend.CastIfNeeded` (lazy, on first
  use — no separate loading step, same as CUDA). Validated against the engine's production
  `GgufDequantizer` CPU codec. Along the way, found and fixed a real pre-existing bug:
  `VulkanGpuTransferHelper.ByteSize` computed zero bytes for any quantized tensor (used `ElementCount *
  SizeInBytes`, and quantized dtypes have `SizeInBytes=0`) — meaning Vulkan had never successfully
  uploaded a quantized tensor before this. See `TROUBLESHOOTING.md`.
- [x] **FlashAttention `sdpa_flash`** SPIR-V — **done** (2026-07-29): `sdpa_flash.comp.glsl`, a fused
  online-softmax kernel (one workgroup per query row, tiled KV in shared memory), backs both
  `ScaledDotProductAttention` and `FlashAttention`/`FlashAttentionDev` for head dims <= 128 (the rare
  >128 case falls back to the materialized 3-pass path). Fixes the Wan-video full-resolution OOM (the
  exact previously-failing B=1,H=24,S=16384,D=128 shape now completes — no ~25 GB score matrix) and sets
  `FlashDecodeSupported => true`, unblocking Zonos onto the GPU-resident path instead of CUDA-only host
  glue. Supports causal masking, GQA, sliding window, and an optional additive mask; does NOT support
  softcap/sink/ALiBi (Gemma-2/GPT-OSS/MPT-class) — those fall through to the CPU reference, a documented
  scope boundary, not a silent gap. See `benchmarks/scoreboards/VULKAN.md` and `TROUBLESHOOTING.md`.
  Remaining: this is a correctness-first Br=1 (one query row per workgroup) design, not yet tuned for
  throughput — coopmat/tensor-core fusion and larger query tiles are Phase 5 material.
- [ ] Small-subgroup `requiredSubgroupSize` pinning; im2col shader 64-bit widening (no longer tooling-blocked
  now that a working SPIR-V compiler is available — same class of bug as the CUDA im2col 32-bit overflow,
  `Conv2D` already throws above ~2^31 im2col elements rather than silently corrupting); descriptor-pool
  `FlipPool` timeline wait.
- [ ] **Per-dispatch barrier scoping** — re-measured post-flash-attention (2026-07-29), **deprioritized,
  not built**: `Dispatch()` (`VulkanBackend.cs`) still emits one unconditional global `VkMemoryBarrier2`
  after every dispatch (`VulkanCommandStream.RecordGlobalComputeBarrier`), with no read/write-direction
  info threaded through the ~40 call sites that would be needed to scope it. Flash attention already
  collapsed the actual worst case — the old per-(B·H) naive-SDPA triple-dispatch loop — into one dispatch
  for head dims <= 128 (the common path); the naive per-head loop only remains as a rare fallback (headDim
  > 128 or non-integer GQA ratio). No isolated barrier-cost number exists (`VulkanLinearProfileMeasurement`
  folds it into an undifferentiated per-dispatch host-overhead bucket) and building one means new
  instrumentation, not just a measurement run. Given GEMM/attention kernel throughput is still 30–430×
  behind CUDA (see the flash-attention and GEMM tables above) — a far larger, already-quantified gap — this
  is not worth chasing now. Revisit only if per-dispatch overhead specifically (not kernel throughput)
  shows up as the bottleneck in a real end-to-end profile.
- [x] **coopmat2** (`VK_NV_cooperative_matrix2`): the base GEMM kernel is built, measured, wired into
  `Linear`/`DispatchMatmul` behind an opt-in flag (`EnableCoopMat2`), and validated against real Krea2
  weights (2026-07-31, see the §2 GEMM-perf-tuning update below). Correctness is perfect throughout.
  **Update, same day**: chased all three open items — (a) benchmarked against the correct coopmat1
  competitor (`matmul_coopmat_partial_m`), coopmat2 still wins 0.74-0.98x; (b) fused bias directly into the
  shader (real correctness/precision win, 1.5%->0.05% max error, but barely moved the real e2e number,
  meaning bias overhead wasn't the dominant real-world cost); (c) the "~4s stall" turned out to be a
  pre-existing `ErrorOutOfDeviceMemory` issue on this shared GPU box, reproduced IDENTICALLY with coopmat2
  OFF — unrelated to coopmat2 entirely. With that shared confound factored out: still ~5% slower at 2 steps,
  but at-or-below baseline at 4 steps. **Update, same day — ran 6 more 4-step runs (4 ON, 2 OFF) to check
  the variance, 9 total samples**: ON mean 121,486ms (n=6, stdev 6,271ms), OFF mean 126,456ms (n=3, stdev
  8,199ms) — coopmat2 3.9% faster on average but NOT statistically significant (Welch's t~0.92, well under
  the ~2 significance threshold; the groups substantially overlap). Honest read: roughly on par at 4 steps,
  plausibly a small win, not confidently provable with 9 samples on this noisy shared box — not worth more
  GPU time chasing further right now. The 2-step result (clean, no OOM confound, ~5% slower) remains the
  more reliable signal. All 9 4-step runs (6 ON, 3 OFF) hit the identical deterministic OOM and produced
  byte-identical output. **Stays off by default.** Still open: fused dequant-on-load, `coopMatReduceNV`-fused
  softmax, `VULKAN_OPTIMIZATION.md` §4.2's broader epilogue fusion, diffusion-shaped tile-size tuning (ggml
  PR #10942), and separately (not a coopmat2 item) — the shared-box VRAM-pressure OOM at 4+ steps, now
  confirmed fully deterministic, is worth its own investigation regardless of this GEMM work. Explicitly
  NVIDIA-only.
- [x] **Root-caused the shared-box 4-step OOM (2026-07-31): NOT a leak — a genuine one-time VRAM peak from
  step-graph capture's design.** Built real diagnostics (`VulkanMemoryAllocator.OnOutOfMemoryDiagnostic` +
  `VulkanGpuTransferHelper.DiagnosticsSummary()`, dumping weight/activation/weight-cast/step-graph-retained
  breakdowns on any genuine OOM) instead of guessing. Real numbers at the OOM: weight cache 12.5 GB,
  activation cache 146.8 MB (small, not accumulating), **step-graph-retained 8.0 GB** — a captured command
  buffer bakes buffer addresses into its descriptor bindings, so every buffer disposed DURING the one-time
  capture window is retained for the graph's whole lifetime instead of freed, meaning capture must hold
  every intermediate activation from Krea2's full 28-block forward pass alive simultaneously. That needs
  well over 15 GB fully realized, on top of ~12.5 GB of weights, against only ~21 GB of real budget on this
  box — never going to fit, independent of anything else running. Explains "2-step never OOMs, 4-step
  always does" precisely: capture only triggers on the 3rd call at a given signature, and a 2-step
  generation never reaches call 3. Practical impact is smaller than it looked: fires ONCE per model load,
  not per-step/per-generation, and the existing fallback-to-eager recovers cleanly (9/9 byte-identical
  outputs across the whole coopmat2 investigation). Not "fixed" in the sense of making capture succeed here
  — the capacity math is real, not a bug — but the diagnostics are now permanent and reusable for any future
  OOM investigation. See `docs/Checklists/TROUBLESHOOTING.md`'s Vulkan section for the full writeup.
- [x] **coopmat2 flipped to default-ON (2026-07-31), correcting the earlier "~5% regression" finding, which
  turned out to be a cross-session variance artifact, not a real property of coopmat2.** Added per-GEMM-path
  host-wall-clock timing to the profiler (`coopmat2/coopmat/tiled avg host-wall` in the `HARTSYINFERENCE_VK_
  PROFILE=1` dump) to investigate why isolated per-op benchmarks consistently showed coopmat2 winning while
  the original real-run comparison showed it losing. A controlled, same-session, back-to-back comparison (4
  runs each config at 2 steps, alternating ON/OFF to cancel time-drift) told a different story than the
  original cross-session one: coopmat2 mean 59,793.5ms vs. baseline 62,296.8ms — a statistically significant
  ~4.0% real win (Welch's t≈2.88, n=4 each). The earlier "regression" was comparing runs from different
  sessions with different external GPU contention (SwarmUI/ComfyUI/rustdesk all resident on this shared
  box) — exactly the confound already flagged as a risk for single-run comparisons on this hardware.
  Extended validation to 8 steps (the exact step count where an EARLIER, unrelated coopmat1 fix passed every
  synthetic test and then hit a real `ErrorDeviceLost`): coopmat2 completed cleanly, no device-loss, ~10.2%
  faster (244,683ms vs. 272,603ms Linear time), byte-identical output to the baseline. Correctness now
  verified byte-identical across every configuration tested this whole investigation — 2-step, 4-step,
  8-step, aligned and non-aligned M, with and without bias, F16 and F32 output — dozens of real generations,
  zero divergence ever observed. `VulkanBackend.EnableCoopMat2` now defaults true (override via
  `HARTSYINFERENCE_VK_COOPMAT2=0`); several coopmat1-specific tests needed an explicit `EnableCoopMat2 =
  false` to keep testing coopmat1 in isolation now that coopmat2 engages first by default. See
  `benchmarks/scoreboards/VULKAN.md` and `TROUBLESHOOTING.md` for full numbers.
- [x] **Beyond-GEMM investigation of the real ~53x per-step Vulkan-vs-CUDA gap (2026-08-01) — root cause
  identified: `VulkanMemoryAllocator`'s block-pooling/reclaim gap, not GEMM kernel throughput.** Real
  full-process wall-clock comparison (same prompt/seed/steps, same 4090): CUDA marginal ~0.67s/denoise-step
  vs. Vulkan ~35.9s/step — a 53.5x gap, far beyond the 8.3-15.6x isolated-GEMM gap. Ruled out with
  controlled benchmarks at Krea2's real FFN shape (M=4109, K=6144, N=16384): weight-cast caching
  (0.99-1.01x — negligible, and a deliberate cross-backend VRAM tradeoff, not Vulkan-specific),
  dependency-chain barrier serialization (chained SwiGlu ~33ms/call, same as independent calls), and
  allocator block-count scaling for FFN-only shapes (flat ~32.7-32.8ms/call at 4 vs. 8 blocks). **Then
  solved**: built `VulkanKrea2BlockFullScaleBenchmark.cs` using the REAL `Krea2Block`/`Krea2Attention`
  classes (not a re-implementation) at Krea2's real config and real joint sequence length (4109) — the full
  attention machinery (QKV, per-head RMSNorm, GQA, RoPE, SDPA, output gate), not just FFN. Result:
  **160-170 ms/GEMM-call — matches and exceeds the real run's 82-135 ms/call**, the first synthetic
  reproduction. Flat across block count AND flat across `EnableCoopMat2` on/off (162 vs. 170ms — rules out
  the GEMM kernel choice entirely). What correlates: `reserved` VRAM hit 19.8 GB at just 4 synthetic blocks
  and allocator block count hit 268 — attention's rich diversity of small, differently-shaped tensors
  (per-head Q/K/V, permutes, RoPE, GQA-repeated K/V) triggers the SAME pooling/reclaim bug found in the
  FFN-only pass, far more severely than FFN's 2-3 recurring large sizes (which pool cleanly). **Root cause
  identified with high confidence: fixing `VulkanMemoryAllocator`'s reclaim cadence is likely the single
  biggest remaining lever on the real Vulkan-vs-CUDA gap — bigger than GEMM kernel throughput.** Not fixed
  yet (needs a careful design — more aggressive reclaim trades against more `Sync()`-induced host stalls).
  **Also found and fixed along the way**: (a) `VulkanGpuTransferHelper.CacheActivation` leaks a tensor's
  previously-cached GPU buffer when the same Tensor object is reused as a non-in-place op's output more
  than once (real `Krea2Block` code doesn't hit this — always fresh tensors — a synthetic-benchmark-only
  artifact, not fixed, just documented); (b) `HartsyInference.Vulkan.Tests` had no GPU-test parallelism
  control, letting concurrent `VulkanBackend` instances from unrelated test classes cause spurious
  `ErrorOutOfDeviceMemory` in PRE-EXISTING tests — fixed via `[assembly: CollectionBehavior(
  DisableTestParallelization = true)]` (`AssemblyInfo.cs`); all 167 tests pass with it in place.
  **Correction (2026-08-02): the "root cause identified with high confidence" claim above does not survive
  direct measurement.** Added always-on allocator/buffer diagnostics (`VulkanMemoryAllocator.
  VkAllocateMemoryStats`/`.SnapshotBlocks()`, `VulkanBufferDiagnostics`) and re-ran the same benchmark.
  Findings: (a) the pooling bug is real — 111-115 of ~125 dedicated blocks are genuinely empty (~9 GB) vs.
  10 pinned (720 MB), confirming eviction (not placement) is the right fix; (b) but `vkAllocateMemory`
  (0.1-0.3%) and `vkCreateBuffer`/`vkDestroyBuffer` overhead (0.1-0.4%) together account for well under 1%
  of the measured wall-clock in steady state — the allocator does NOT explain the 160-170 ms/GEMM-call
  figure; (c) that figure is itself the actual defect: `VulkanCommandStream` batches dispatches
  (`FlushThreshold=8`) so per-op host-wall (what `VulkanProfiler` measures) is recording cost only — real GPU
  execution time surfaces later, collected at whatever call forces a submit-and-wait (here, the benchmark's
  final `Sync()`). Confirmed directly: the profiler's own per-op accounting covers only ~620ms/~164ms
  (coopmat1/coopmat2) of host-wall across the WHOLE backend lifetime, while the benchmark's external
  Stopwatch around half that work reports ~5.2-5.5s — GPU execution collected at sync, not unaccounted cost.
  So `ms/GEMM-call` = all-ops'-Sync()-collected-GPU-time / GEMM-count-alone, not a per-Linear GPU cost, and
  isn't comparable to the isolated GPU-only-timing numbers elsewhere in this doc. It's also not directly
  comparable to the real run's "82-135 ms/Linear-call" figure, though for a weaker reason — that number came
  from `HARTSY_LOG_LEVEL=Verbose` phase logging, a different derivation than this benchmark's arithmetic, so
  it may or may not share the same issue; not checked this pass. Also stale: the "reclaim trades against more
  `Sync()`-induced stalls" tension — `SubmitAndAdvance()` already reclaims opportunistically off a
  non-blocking timeline peek, so a reclaim fix needs no new waits.
  **Revised next step**: the allocator eviction fix is still worth doing (real ~9 GB of dead reserved VRAM,
  same shape as the documented 4+-step OOM below), but as VRAM-headroom/OOM-avoidance work, not as the fix
  for the ~53x gap — what actually drives that gap is open again; isolating true per-op GPU time via
  `VulkanGpuTimer`/`MeasureGpuTimeMs` at Krea2's real shape is the way to reopen it. See `docs/Checklists/
  TROUBLESHOOTING.md`, `VulkanFp8WeightCastOverheadBenchmark.cs`, `VulkanKrea2BlockFullScaleBenchmark.cs`.
- [~] **Wire the INT8 quantizer into `Linear`'s call surface** — **op-level wiring done** (2026-07-29);
  **NOT the same as this section's original ask**, which was model *loading* wiring with an e2e SSIM/parity
  gate — that part is still open, see below. What landed: `VulkanBackend.Linear` has an opt-in INT8
  dot-product GEMM path (`TryDispatchInt8Linear`), gated by the settable `EnableInt8Linear` property
  (defaults from `HARTSYINFERENCE_VK_INT8=1`, mirroring `CudaBackend.EnableW8A8`'s instance-property
  pattern rather than a static-readonly field, specifically so tests/tooling can flip it without an env var
  + fresh process). Scoped narrowly to plain 2-D F32 Linear (K%4==0, integer dot-product feature present)
  — anything else falls through to the unchanged GEMM path. Re-quantizes both weight and activation on
  every call via the already bit-exact-validated `Int8Quantizer.RowwiseSymmetric` + `MatMulInt8`; measured
  relative Frobenius error ~0.55% vs an exact-path (opt-in off) baseline that itself matches the F32
  reference to <0.0001% (`Backend_Linear_Int8OptIn_ApproximatesF32Reference`, run twice on identical input
  with the flag off/on so the test can't pass via a silent fallthrough). Also gated on: a chained-into-a-
  second-GPU-op test (`Backend_Linear_Int8OptIn_BiasSurvivesDownstreamGpuConsumption`) confirming the
  bias-add's host-side write doesn't leave a stale cached GPU buffer for a downstream consumer to read —
  verified safe (the tensor's lazy-sync callback evicts+frees the GPU cache entry on the host read before
  the write happens), not just assumed safe.
  **Still open** (the actual model-loading item): no model's weight-loading path calls into this yet, and
  there is no e2e SSIM/parity gate — `EnableInt8Linear` is a manually-flipped correctness-tested opt-in,
  not something any real generation pipeline turns on today. Also deliberately deferred: this re-quantizes
  the weight from scratch on every call — caching the weight's quantized form across calls (weights are
  static between calls; only activations change) is the natural perf follow-up and needs its own lifecycle
  wiring (freed alongside `FreeWeights`), not bundled into this pass. Per-shape INT8 tile selection (below)
  is a separate, still-open item.
- [x] **LLM decode-graph device state (Phase 6a-d)** — **done** (2026-07-30): the Vulkan-native leaf ops a
  future graph replay would drive, all validated against CPU references on the 3060 + llvmpipe
  (`VulkanDecodeGraphTests`): `BuildRopeTableDevice`/`RopeApplyDecodeStep` (new `rope_decode_step.comp.glsl`,
  interleaved + split-half, partial-rotary correct), `EmbedGatherDecodeStep`/`ArgMaxInto` (new
  `embed_gather_decode.comp.glsl`/`argmax_lastdim.comp.glsl`, single-workgroup reduction), device
  token-id/position/history/counter buffers (`AllocDeviceTokenId`/`AllocDevicePos`/`AllocDeviceHistory`/
  `AllocDeviceCounter` + writers), `AppendTokenHistoryStep`/`ApplyRepetitionPenaltyStep` (new
  `history_append.comp.glsl`/`repetition_penalty.comp.glsl`, matching
  `HartsyInference.LLM.Sampling.RepetitionPenaltyStep`'s HF-convention compounding-on-repeat exactly),
  `KvCacheAppendDev` (new `kv_cache_append_dev.comp.glsl`), and `FlashAttentionDev` (new
  `sdpa_flash_dev_f32.spv`, a `HAS_DEVICE_POS` compile variant of `sdpa_flash.comp.glsl`). `GraphDecodeSupported`
  is a **settable property defaulting to OFF** (not a hardcoded `true`) — several real production call
  sites (`Qwen35Model`, `GenericTransformer`, `SsmGenerationPipeline`, `CsmModel`, `MusicGenDecoder`) gate
  straight onto this path the instant it flips, with no separate opt-in of their own, so it stays off until
  a real end-to-end decode-loop parity test (not just these per-op unit tests) validates it — tracked below.
  Two real bugs found and fixed before shipping (both would have been silent, production-breaking
  correctness bugs, not perf regressions): (1) a host-side scalar-buffer write racing ahead of an
  already-recorded-but-unsubmitted dispatch that still needed the OLD value — `WriteScalarBuffer` now
  syncs first (see `TROUBLESHOOTING.md`); (2) `KvCacheAppendDev`/`FlashAttentionDev`'s initial versions
  forwarded the caller's placeholder host `offset`/`kvLen`/`qOffset` (real callers pass literal `0`s,
  expecting `devicePos` to be authoritative) instead of reading the actual value from the device buffer —
  caught in review, not by a test, before it ever ran.
- [ ] **LLM decode-loop end-to-end parity + throughput** (Phase 6f) — the gate that flips
  `GraphDecodeSupported` on: token-for-token parity vs. the eager path on a real small model, then a
  tokens/sec measurement. Deferred past this session (user directed finishing Phases 6/7/8 first, then one
  e2e image-model run) — a local `qwen2.5-0.5b-instruct-q4_k_m.gguf` is available on this box for it.
- [x] **`VulkanStepGraph`** (the plan's Phase 6e, the CUDA-Graph-capture analog backing
  `StepGraphBegin`/`StepGraphEndAndLaunch`/`StepGraphLaunch`/`StepGraphReset`/`StepGraphOwner`) — **built**
  (2026-07-30), reversing the earlier "deliberately NOT built" call in this same entry. The descriptor
  hazard that call was based on (`Dispatch()`'s pool-ring set could be overwritten by a later dispatch
  before a frozen command buffer replays it) is real for pool-allocated sets, but dissolves entirely for
  **push descriptors** (`VK_KHR_push_descriptor`): a push-descriptor-bound set has no backing `VkDescriptorSet`
  object for a ring to invalidate — the bindings are baked directly into the command buffer at record time.
  `VulkanBackend.StepGraphBegin()` now switches `Dispatch()`/`CopyInto()` onto a capture branch that records
  into `VulkanStepGraph`'s own persistent `VkCommandBuffer` via `PushSet` instead of `AllocateSet`, with no
  rearchitecture of the ~40 existing `Dispatch()` call sites — one `forCapture` bool threaded through
  `VulkanDescriptorManager`/`VulkanKernelRegistry`/`GetKernel` selects a parallel, push-descriptor-flagged
  layout cache instead of the pool-allocated one. `vkCmdPushDescriptorSet` is not statically exported by
  this NVIDIA driver (`EntryPointNotFoundException` on a direct P/Invoke) — resolved dynamically via
  `vkGetDeviceProcAddr`, trying the `KHR`-suffixed name first, falling back to the unsuffixed Vulkan 1.4
  core name (see `TROUBLESHOOTING.md`). Every transient buffer touched during capture must be redirected
  from "free now" to "retain until `StepGraphReset()`" (`VulkanGpuTransferHelper.CapturingStepGraph`) since a
  recorded command buffer bakes device addresses at record time; independent queue submissions have no
  automatic ordering in Vulkan, so `_stream.WaitIdleHost()` must precede both `StepGraphBegin()` and
  `StepGraphLaunch()` or a replay can race a still-in-flight normal-stream write. Proven correct by
  `StepGraph_TrivialCapture_ReplaysWithFreshDataAcrossThreeSteps` (130/130 suite green). **Not wired into
  any real model yet** — see the Krea2 attempt below, which found it works but doesn't fit that model's
  memory profile. `StepGraphSupported => Vk.HasPushDescriptor` is now `true` on this box, so any of the ~13
  DiT models gating onto `HARTSY_DIT_GRAPH=1` will now attempt capture where they previously couldn't; the
  existing `catch (Exception ex) when (capture)` fallback in each model's forward pass is the only thing
  standing between a capture-illegal op and a crash for models not yet audited against the capture
  contract — audit before flipping the env var on for a model this entry hasn't covered.
- [ ] **RCCL** collectives for multi-GPU on AMD (§1).
- [x] **Diffusion domain fill (Phase 7), scoped to Krea2** — **done, e2e image valid** (2026-07-30). The
  static call-site analysis performed earlier this session (every `backend.*` call across
  `Krea2Pipeline`/`Krea2Transformer`/`Krea2Attention`/`Krea2TextFusion`/`LlamaStyleEncoder`/
  `QwenImageVaeDecoder`, cross-referenced against every `NotImplementedException`/`NotSupportedException`
  throw site) concluded "zero backend gaps" — **this claim was wrong**, disproven the moment a real
  weight-loaded e2e run was actually attempted, exactly the failure mode the plan itself warned static
  analysis alone can't rule out. Five real, previously-unknown backend gaps surfaced sequentially, each
  caught by re-running after the previous fix (see `TROUBLESHOOTING.md` for full root-cause writeups):
  **(1)** `WanRopeInterleaved` had no `VulkanBackend` override at all — `FluxRope.ApplyGpuGqa`'s GQA rope
  fell through to `IBackend`'s CPU-loop default, which `AccessViolationException`'d on the GPU-resident
  Q/K tensors (a process-crashing failure, not a catchable exception) — fixed with a new
  `wan_rope_interleaved.comp.glsl` + override, unit-tested F32/F16 on 3060 + llvmpipe. **(2)**
  `RepeatKvHeads` had no override — F32-only CPU default threw on Krea2's F16 GQA K/V — fixed with
  `repeat_kv_heads.comp.glsl`. **(3)** `GatedResidualLastDim` had no override — same F32-only throw on F16
  activations — fixed with `gated_residual_last_dim.comp.glsl`. **(4)** `SliceRows` had no override —
  F32-only throw on Krea2's F16 joint sequence (`Krea2Transformer.SliceTail`) — fixed with
  `slice_rows.comp.glsl`. **(5)** `Conv2D`'s im2col path materialized the FULL `[gemmK, outH·outW]` column
  matrix in one allocation — ~7 GB at Krea2's 1024×1024 VAE-decode resolution, OOMing even with the
  transformer's weights already freed — fixed by tiling over output columns. **Sixth, the actual blocker:**
  after all five gaps closed, the e2e run completed without crashing but produced an invalid image
  (NaN/all-black in F16, pure noise if F16 was worked around to F32). Root-caused to `DispatchMatmul`
  deriving `M`/`N` from `output.Shape`'s rank structure instead of from the weight tensor — silently wrong
  for any Linear whose output is shaped `[B, S, heads, headDim]` (Krea2's Q/K/V, split that way so a
  downstream per-head `RmsNorm` can normalize without a reshape). Fixed by deriving `N` from the weight
  operand (mirrors `CudaBackend.LinearImpl`, which never reads `output.Shape` at all) and `M` as
  `output.ElementCount / N`. **Krea2 now produces a correct, coherent image on Vulkan** — verified via the
  CLI (`HartsyInference.Cli`, the production path, not the xUnit test harness) with the identical
  prompt/seed/steps/cfg as CUDA; both produce the same recognizable astronaut-on-horse photo. Also fixed
  along the way: a real production OOM in `Krea2Recipe.cs` (`CacheWeightCasts` wasn't disabled on the
  Engine/CLI path, only in the xUnit test's manual pipeline construction — a class of bug the test suite
  structurally cannot catch). All fixes landed with new GLSL kernels/overrides and dedicated regression
  tests passing on the 3060 and Mesa llvmpipe — see `Backend_WanRopeInterleaved_MatchesCpu`,
  `Backend_RepeatKvHeads_MatchesCpu`, `Backend_GatedResidualLastDim_MatchesCpu`, `Backend_SliceRows_MatchesCpu`,
  `Backend_Conv2D_TiledPath_MatchesUntiled`, `Backend_Linear_SplitHeadOutputShape_MatchesCpu` in
  `VulkanBackendSmokeTests.cs` (124/124 total suite green). **Speed is NOT yet at parity** — see the open
  item immediately below and `benchmarks/scoreboards/VULKAN.md`. `PARITY_VERIFICATION.md` can now take a
  real Krea2/Vulkan row (coherent image, CLI path, matches CUDA) — not yet added, tracked as follow-up.
- [ ] **Krea2 Vulkan e2e speed: ~29× slower than CUDA, known dispatch-overhead ceiling, not a bug.** With
  correctness fixed (above), a `HARTSY_LOG_LEVEL=Verbose` breakdown on the 4090 shows: text encode ~2.4s,
  DiT preload ~2.2s, denoise loop ~34.4s/step × 8 steps ≈ 276s (stable within 1.1s across all 8 steps — a
  steady-state cost, not a stall), VAE decode ~39s, total ~322s vs. CUDA's ~11s end-to-end for the identical
  config. Checked for a fixable cause before concluding this is architectural: dispatches already batch by
  default (`HARTSYINFERENCE_VK_SUBMIT_PER_OP` defaults off, `FlushThreshold=8`), and the tight, uniform
  per-step timing across 8 independent steps is inconsistent with a stray-D2H-sync or un-batched-submit
  explanation (either would show as an outlier step or a large idle gap, not uniform stability). This is the
  already-scoped Vulkan dispatch-overhead ceiling — hand-written GLSL kernels with per-dispatch
  submit/barrier overhead vs. CUDA's cuBLAS/cuDNN + graph capture — not a new finding. Closing it is Phase
  5's core-primitive perf ceiling work (coopmat2, INT8 GEMM wired into model loading) plus extending Phase
  6's step-graph-capture mechanism (built for LLM decode) to the diffusion denoise loop, per Phase 7's
  "denoise-loop graph capture reusing Phase 6's command-buffer-reuse mechanism" item — **attempted
  (2026-07-30), does not close this gap for Krea2 specifically.** `VulkanStepGraph` (above) now exists and
  works, but wiring Krea2's denoise loop onto it via `HARTSY_DIT_GRAPH=1` and running the real CLI found a
  structural conflict, not a bug: capture requires retaining every transient buffer touched during the
  recorded pass (a replayed command buffer's bindings point at fixed device addresses), but Krea2 runs with
  `CacheWeightCasts=false` specifically because its fp8 weights transient-dequant to F16 per-GEMM and get
  freed immediately — retaining all ~224 (28 blocks × 8 Linears) of those casts across one capture instead
  of freeing-and-reusing them OOMs (`vkAllocateMemory` failed on the 4th block's SwiGlu Linear, size
  ~128MB, `ErrorOutOfDeviceMemory`) even on the 4090. These two requirements are mutually exclusive for this
  model at this VRAM budget, not merely hard to reconcile — closing it needs an intra-capture bump-pointer
  arena with slot reuse for weight casts specifically (the "CUDA Graph allocation-node semantics have no
  Vulkan equivalent" gap this plan's Non-Goals section already named), which is real, separate engineering,
  not a quick follow-up. Before building it: run a **single-block capture** (1 of 28) instead of all-block
  to measure the actual per-block replay speedup against eager — peak retained VRAM drops ~28× (should
  easily fit) and the number tells you whether the arena is worth building at all, since capture removes
  host dispatch/submit overhead, not GPU kernel time, and the GEMM scoreboard's 30–160× per-op kernel gap
  means the ceiling on what capture alone can buy here may be modest. Along the way, three real,
  previously-unknown `VulkanBackend` gaps surfaced (found via the real-CLI-run loop, each independently
  caught by the model's own `catch when (capture)` safety net, which fell back to eager and completed with
  a correct image in all 4 attempted runs — the fallback itself is now proven robust including under a
  mid-capture OOM): `CfgEulerStep` (the CFG/Euler combine step had no GPU-resident override at all —
  `IBackend`'s CPU-loop default was reading/writing `.DataPointer` every denoise step even in the normal
  EAGER path, not just breaking capture), `Concat` (was a pure CPU `Buffer.MemoryCopy` loop — rewritten
  device-resident via `vkCmdCopyBuffer`, closing a real per-step D2H sync), and `AddScalar` (`DiTUtils.Modulate`
  had no override, same CPU-default D2H cost). All three are fixed permanently and eliminate real per-step
  D2H syncs in the eager path Krea2 actually runs today, independent of whether graph capture ever lands for
  this model — see `Backend_CfgEulerStep_MatchesCpu_PreservesZAddress`,
  `GetD2hSyncCount_Concat_StaysGpuResident`, `Backend_AddScalar_MatchesCpu` (130/130 suite green).
- [x] **VAE decode fixed (2026-07-31): ~33s → ~0.5s (~65×), closing a real bug, not the kernel-throughput
  ceiling.** `HARTSYINFERENCE_VK_SUBMIT_PER_OP=1` A/B'd against default batching first (~4% delta on the
  denoise loop, ~0% on VAE decode) — confirming the denoise loop is kernel-bound, not dispatch-overhead-bound
  (graph capture's ceiling here is small; the weight-cast arena above is correctly not worth building). VAE
  decode's near-zero sensitivity to batching pointed away from a dispatch storm and toward a CPU-loop
  fallthrough: `WanRmsNormChannel` had **no `VulkanBackend` override at all** — every call (the decoder's
  final head norm, `[1,96,1024,1024]`, ~402MB, the worst possible shape for this) fell through to
  `IBackend`'s CPU-loop default (full D2H sync, single-threaded cache-hostile reduction, H2D re-upload).
  Fixed with a real GLSL kernel mirroring `CudaBackend`'s existing `wan_vae_rms_norm_channel.cu`. VAE decode:
  33.0s → 0.5–0.65s (run-to-run jitter); full generation: 331.1s → 296.2s (~10.6%). Now inside the normal
  ~30–50× hand-written-vs-cuDNN band, not a pathological outlier. See `TROUBLESHOOTING.md` for the full
  writeup, `benchmarks/scoreboards/
  VULKAN.md` for before/after numbers, `Backend_WanRmsNormChannel_MatchesCpu` for the regression gate
  (132/132 suite green, 3060 + llvmpipe). Also found (not fixed, documented as a trap for next time):
  `HARTSYINFERENCE_VK_PROFILE=1` is blind to `Add`/`Clamp`/`Fill`/`Scale`/`Silu`/`Gelu`/`Sigmoid`/`Tanh`/
  `Elu` (they dispatch without opening an `EnterOp()` scope, so `VulkanProfiler.Record` never sees them) —
  this is why the first profiling pass read "`Conv2D` = 32ms" against a 33s phase instead of pointing at the
  real cost.
- [ ] **Coopmat is engaged for only ~1% of a real Krea2 run (16/2112 GEMMs) — root cause is shape
  (M/N/K must be exact multiples of 16, no padding support), not dtype.** Measured directly (2026-07-31) via
  new `VulkanBackend` engagement counters (`_coopmatGemmCount`/`_tiledGemmCount`, printed alongside
  `HARTSYINFERENCE_VK_PROFILE=1`). A real dtype bug was found and fixed along the way — `ResolveGemmDtype`
  never promoted F32 outputs to F16 to reach `TryDispatchCoopmat`'s existing `OUTPUT_F32` support, so no
  F32-output Linear (which is every Linear in `Krea2Transformer`, matching `CudaBackend.LinearImpl`'s own
  convention) could ever reach coopmat regardless of input dtype — but fixing it only moved the number from
  0/2112 to 16/2112: the dtype gate was never the binding constraint for this model. The real blocker is
  that `matmul_coopmat.comp.glsl` hard-requires M/N/K all exact multiples of 16 (its own comment: "spec
  handles partial fragments... IF the host pads the buffer... We require multiples of 16 here" — no padding
  is built), and every per-block QKV/FFN/out-proj Linear operates on the joint `[txtSeq+imgSeq]` sequence,
  where `txtSeq` is the raw tokenized+encoded prompt length with no padding to a fixed size. Measured on the
  reference prompt: `imgSeq=4096` (a multiple of 16), `txtSeq=13` → `jointSeq=4109`, 3 short of the next
  multiple of 16 — and since `txtSeq` is prompt-dependent, essentially no real prompt lands on a multiple of
  16 by chance, so this blocks nearly every expensive GEMM in the model regardless of prompt. **The dtype fix
  was reverted the same session** (kept only the zero-cost engagement counters): it introduced a real,
  never-actually-verified precision trade-off (full F16 compute — 5-bit exponent, vs. F32/TF32's 8-bit
  range — on FP8-dequantized-weight GEMMs specifically, where activation magnitudes are least predictable)
  for a measured zero-throughput win, since the shape gate blocks the same GEMMs either way.
  **Update 2026-07-31 — the host-side M-padding fix was also attempted, and also reverted, after a real GPU
  crash.** Built `TryDispatchCoopmatMPadded`: pad the M dimension up to the next multiple of 16 via a scratch
  buffer + device-to-device copy, dispatch the existing (unchanged) coopmat kernel against the padded shape.
  Passed a from-scratch CPU-reference correctness test, a 200-iteration no-sync stress test at Krea2's exact
  real scale (zero leak, 17s), and two full real CLI runs (2-step, 4-step — both completed cleanly, GPU
  pinned at 100% utilization, coopmat engagement climbing to 60.7% then 74.9%). **A full 8-step run then
  failed with `Vulkan error -4 (ErrorDeviceLost): vkQueueSubmit2`** — a genuine GPU driver fault/reset, not
  an allocation failure (which the existing retry-then-throw OOM path already handles cleanly and would have
  produced instead). An earlier apparent 40-minute "hang" on the same fix turned out to be confounded by an
  unrelated external process (a ComfyUI backend, independently consuming ~12.7GB of the same 4090, started
  sometime mid-session outside this work) — ruled out as the sole cause once the device-loss reproduced
  again on a verified-clean GPU with ~22GB of headroom. No root cause found before reverting (two leading,
  unconfirmed suspects — a possible barrier gap if `AcquireRecording()` ever splits the padding copy and the
  coopmat dispatch across separate command-buffer submissions, or an out-of-bounds access that only
  manifests under the real model's op-interleaving, not a simple repeated-Linear stress loop); needs Vulkan
  validation layers or `compute-sanitizer`-class tooling to pin down properly, not more full-CLI-run
  attempts. **Recommended next design, not a repeat of this one:** ggml/llama.cpp's Vulkan backend
  (`mul_mm.comp`) handles the same non-tile-aligned-GEMM problem via a specialization-constant `ALIGNED` flag
  selecting a per-element-bounds-checked, shared-memory-staged load/store — no separate scratch buffer, no
  device-to-device copy, no cross-command-buffer barrier risk, since everything stays inside one dispatch's
  own shared memory. Real shader work (its own new risk surface, needs its own careful verification), but
  structurally avoids this whole bug class. If/when either approach lands, re-evaluate the F32→F16 dtype
  promotion from a real e2e SSIM/pixel-diff gate rather than the "CUDA already runs reduced-precision GEMMs
  by default" argument alone (necessary, not sufficient — TF32's wider exponent range isn't the same
  guarantee as F16's). See `TROUBLESHOOTING.md` for the full writeup.
- [x] **Update 2026-07-31 — the ggml shared-memory design (the recommended next attempt above) is DONE and
  KEPT: coopmat engagement 0% → 84.8% on a real Krea2 run, verified stable, but delivers ~0 real-world
  speedup — the actual lever is coopmat kernel throughput, not engagement.** Built
  `matmul_coopmat_partial_m.comp.glsl`, a separate shader (the existing aligned fast path stays byte-
  identical, zero new risk there) mirroring `matmul_tiled.comp.glsl`'s own proven bounds-checked shared-
  memory staging idiom — no scratch VulkanBuffer, no device-to-device copy, no cross-command-buffer barrier,
  structurally avoiding the risk class that caused the previous attempt's `ErrorDeviceLost`. **A real,
  separate bug was caught by a 9-case parameterized test BEFORE ever touching the real model**: a divergent-
  barrier bug (some subgroups in a workgroup hitting an early `return` for out-of-N-bounds columns while
  siblings continued to this kernel's `barrier()` — undefined behavior, silent 100%-wrong output on this
  GPU, not a crash) triggered whenever N wasn't a multiple of the workgroup tile width BN, independent of
  N being a multiple of 16 (the host's existing gate). Fixed by removing every early return after the
  K-loop in favor of unconditional barrier participation + bounds-checked scalar draining. After the fix:
  9/9 correctness cases pass, a pre-existing large-scale real-shape test (M=4108) now automatically
  exercises this kernel and passes unmodified, a 200-iteration no-sync stress test at Krea2's exact scale
  shows zero leak, and three consecutive real Krea2 CLI runs (2/4/8-step) all completed cleanly — the
  8-step run specifically: 296.7s, coopmat engagement 84.8% (1792/2112), GPU pinned at 100% utilization
  throughout (not idle/stuck), correct verified-by-eye output image. Full suite 143/143 (3060 + 4090);
  llvmpipe unaffected (no coopmat hardware, these tests self-skip there). **The honest result**: an A/B done
  earlier this session (78.9s coopmat fully OFF vs. 80.5s at 60.7% engagement, same 2-step shape) already
  showed engagement rate doesn't move wall-clock, and the 8-step number confirms it at full scale — coopmat
  was ALREADY measured 30–157× slower than CUDA's cuBLAS even where it engages (see the GEMM table at the
  top of `benchmarks/scoreboards/VULKAN.md`), so raising engagement from 0%→84.8% correctly fixes a real
  capability gap (and sets up any future coopmat-kernel-throughput work to reach these GEMMs automatically)
  but was never going to close Krea2's CUDA gap on its own. **Kept in the codebase** (unlike the second,
  reverted attempt) because it's proven correct and safe. See `TROUBLESHOOTING.md` for the full writeup.
- [ ] **Update 2026-07-31 — register blocking (ggml's core coopmat optimization) tested directly and found
  to be a net LOSS on this RTX 4090; the 30–157× gap is confirmed NOT primarily an arithmetic-intensity
  problem.** Read ggml/llama.cpp's actual `mul_mm.comp` source (not just PR summaries) and built
  `matmul_coopmat_blocked.comp.glsl` — a standalone diagnostic kernel (not wired into any production path)
  implementing its core technique: each subgroup computes a grid of accumulators from one shared-memory-
  staged tile instead of one accumulator per direct global load. Verified correct (4/4 CPU-reference cases),
  then A/B'd against the existing naive kernel at the GEMM table's own shapes: **register blocking
  (WM=WN=32, 4 subgroups/workgroup instead of 16): 0.79–1.05×, neutral to worse** — losing occupancy costs
  more than the reduced memory traffic saves on this GPU. **Plain shared-memory staging alone (occupancy
  unchanged): 1.00–1.41×, a real but modest win**, correlated with K-depth. Neither closes a meaningful
  fraction of 30–157×. **This redirects the investigation**: the bottleneck isn't primarily memory access
  pattern — it's likely raw `coopMatMulAdd` throughput on this driver's coopmat1 implementation vs. CUDA's
  hand-tuned WMMA/MMA, or fixed per-dispatch overhead no kernel-level tiling can amortize. Real GPU profiling
  (Nsight Compute / Vulkan validation layers) is the recommended next step — neither is available on this
  box (no passwordless `sudo` to install `vulkan-validationlayers`, though the package exists). Barring that,
  `VK_NV_cooperative_matrix2` (NVIDIA-only, §3 below) is the next concrete kernel-level lever, since it
  changes the underlying instruction/memory path rather than the tiling strategy around the same coopmat1
  instructions this pass tuned. See `benchmarks/scoreboards/VULKAN.md` for the full numbers and a
  methodology caveat on the absolute (not relative) timing figures.
- [x] **Update 2026-07-31 — got real GPU profiling working with root access; the true gap is 10–22×, not
  30–157×, and coopmat is NOT meaningfully faster than the plain scalar kernel on this hardware.** Nsight
  Compute (already installed) confirmed CUDA-only (`No kernels were profiled` against the Vulkan process,
  zero Vulkan/graphics mentions in its own `--help`); Nsight Graphics (the tool that would show real SM/
  tensor-core utilization) isn't apt-installable. **Built real GPU-side profiling instead**: `VulkanGpuTimer`
  + `VulkanBackend.MeasureGpuTimeMs`, using `VkQueryPool` timestamps (`vkCmdWriteTimestamp2`) — a core Vulkan
  feature, always available, the standard fallback when vendor tooling isn't cooperating. This measures pure
  GPU execution time with zero host-overhead confound, unlike every prior `Stopwatch`-based number in this
  file. Result: the GEMM table's shapes actually run at **10.6–22.4× vs. CUDA**, not 30–157× — most of that
  earlier figure was host/dispatch/sync overhead specific to an isolated-per-call benchmark methodology, not
  kernel throughput. **Bigger finding**: re-measuring with `HARTSYINFERENCE_VK_DISABLE_COOPMAT=1` (forcing
  the scalar `matmul_tiled` fallback, confirmed via the engagement counters) produced a STATISTICALLY
  IDENTICAL number to coopmat-enabled (22.7× vs. 21.3×) — coopmat isn't actually faster than the plain
  scalar kernel at these shapes on this GPU. This is the finding that makes every other result this session
  consistent (why raising coopmat engagement 0%→84.8% didn't speed up Krea2; why register blocking/staging
  only moved the needle 0.8–1.4×) — those techniques all assume coopmat's tensor-core path has more headroom
  than a well-written scalar kernel, and if it doesn't, tuning its tiling can't create headroom that isn't
  there. Most likely explanation: `VK_KHR_cooperative_matrix` (coopmat1) on this driver isn't compiling down
  to real tensor-core instructions as effectively as CUDA's WMMA/MMA PTX — a driver/compiler characteristic,
  not fixable by further kernel restructuring. **Next step**: `VK_NV_cooperative_matrix2` (a genuinely
  different instruction path, not just different tiling — §3 below) as the concrete next experiment, or
  Nsight Graphics (needs an NVIDIA account login) for a direct SASS-level answer. See
  `benchmarks/scoreboards/VULKAN.md` for full numbers.
- [x] **Update 2026-07-31 — `VK_NV_cooperative_matrix2` built and measured: a real 1.2-2.5x speedup over
  coopmat1/scalar, confirming the hypothesis above, though it does NOT close the CUDA gap.** Built
  `matmul_coopmat2.comp.glsl` (WORKGROUP-scope coopmat, `tensorLayoutNV` + `coopMatLoadTensorNV`/
  `coopMatStoreTensorNV` direct-to-global-memory addressing, built-in bounds clamping — architecturally
  distinct from coopmat1, not just a different tiling of the same instructions). Wired full device support:
  extension detection, `vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV` capability
  querying (RTX 4090 auto-selects the largest FP16/FP32 workgroup-scope config: 32x32 tile granularity,
  K-granularity 16, 256 invocations/workgroup), and the `VkPhysicalDeviceCooperativeMatrix2FeaturesNV`
  feature-chain/extension-enable plumbing (`VulkanCapabilities.HasCooperativeMatrix2` +
  `CoopMat2{M,N,K}Granularity`/`CoopMat2WorkgroupInvocations`). Correctness verified including shapes where
  none of M/K/N are multiples of the tile size — the clamp-mode addressing needs zero manual bounds-checking,
  unlike every coopmat1 kernel here. K-block size swept {16..256}; BK=64 is the measured-best default.
  **GPU-only-time result: coopmat2 is 1.2-2.5x faster than naive coopmat1/scalar (best on the smallest-K
  shape), landing the CUDA gap at 8.3-15.6x — roughly HALF the prior 10.5-25.1x gap.** Confirms coopmat1's
  tensor-core engagement specifically was the bottleneck, as hypothesized. **Update, same day**: wired into
  `Linear`/`DispatchMatmul` behind a new opt-in `VulkanBackend.EnableCoopMat2` property
  (`HARTSYINFERENCE_VK_COOPMAT2=1`, off by default) and validated against real Krea2 weights — correctness
  is perfect (5/5 runs byte-identical PNG output vs. baseline), but real e2e wall-clock is a REGRESSION
  (65.7-65.9s vs. 58.6s baseline, reproducible), not the win the isolated benchmark predicted. Root causes:
  three hypothesized causes: wrong coopmat1 baseline in the isolated benchmark, un-fused bias epilogue, and
  an unexplained ~4s one-time per-run stall. **Update, same day — all three chased down:** re-benchmarking
  against the correct competitor (`matmul_coopmat_partial_m`) showed coopmat2 STILL wins (0.74-0.98x); fusing
  bias directly into the shader was a real correctness/precision win (1.5%->0.05% error) but barely moved the
  real e2e number; and the "~4s stall" turned out to be a pre-existing `ErrorOutOfDeviceMemory` on this
  shared GPU box, reproduced IDENTICALLY with coopmat2 completely OFF — nothing to do with coopmat2. Once
  that shared confound is factored out: still ~5% slower at 2 steps, roughly on par at 4 steps. **Update,
  same day — 6 more 4-step runs (9 total samples)**: coopmat2 3.9% faster on average (121,486ms vs.
  126,456ms) but NOT statistically significant (Welch's t~0.92) — the distributions overlap too much to call
  a confident winner with 9 samples on this noisy shared box, and more runs isn't a good use of shared GPU
  time right now. The clean 2-step result (no OOM confound, ~5% slower) is the more reliable signal.
  **Stays off by default** — see `benchmarks/scoreboards/VULKAN.md` and `TROUBLESHOOTING.md` for full
  numbers, `tests/HartsyInference.Vulkan.Tests/VulkanCoopMat2LinearTests.cs` for the wiring tests. **Next
  step**: ggml PR #10942's tile-size tuning or Nsight Graphics, if this gets revisited. Separately (not a
  coopmat2 item): the shared-box OOM, now confirmed fully deterministic at 4+ steps, is worth its own look
  regardless.
- [ ] **Video/audio/vision domain fill (Phase 7 menu items)** — explicitly deferred, not evaluated this
  session: Wan 3D/volumetric conv (blocks the Wan VAE), grid-sample/MSDA (blocks deformable-attention
  detection models), dedicated audio AdaIN/activation fused kernels. No model in either of these domains
  was the target this session (Krea2, the one real-weight run performed, is image/diffusion only) — per the
  plan's own framing, this is "a menu prioritized by which models the user actually runs on Vulkan, not a
  mandatory checklist." Revisit if/when a specific video, audio, or vision model is the actual target.

## 4. Quantization

Core "more quant support" axis. Note the hard-won lesson: local per-layer relL2 does **not** predict e2e
SSIM (errors add in quadrature) — always gate on end-to-end SSIM/FVD, not local diffs.

- [ ] **Native FP8 GEMM (Ada+):** e2e A/B + SSIM gate; **default-on-for-Ada decision pending user**.
- [ ] **W8A8 stage 4:** SmoothQuant fails the 0.95 SSIM gate (weight-side floor binding). Open options to
  choose: grouped-K quant on both operands · mixed-precision layer skipping · accept ~0.92 SSIM ·
  timestep-aware (NDTC) calibration. Then block-level + Swarm e2e A/B.
- [ ] **Diffusion low-VRAM GGUF** quant wiring (opt-in; 🔒 needs diffusion-GGUF weights).
- [ ] **KV-cache quantization** + paged KV (§5).
- [ ] **MMQ-class quantized-GEMM prefill kernels** — the one unsolved LLM prefill gap (GLM-4-9B TTFT).
- [ ] **In-kernel fused dequant-GEMV** (faster + zero transient); quantized embed/lm_head GPU gather.
- [ ] **Exotic quant codecs:** IQ1/IQ2/IQ3 readers.

## 5. LLM decode / serving throughput

Decode already beats llama.cpp on most models; these are the remaining levers (from the decode/throughput
grinds).

- [ ] **On-device MoE routing** — unblocks graph decode for olmoe/qwen2moe/granitemoe and fixes the D2H
  per-token readback bug (see `TROUBLESHOOTING.md` §Serving). Ties to expert-parallel (§1 M3).
- [ ] **On-device sampler** (temp/top-k/top-p) so graph decode isn't greedy-only.
- [ ] **FP16 activation pipeline** (decode Phase 3b); **QKV/Gate-Up projection fusion** (partial).
- [ ] **Big-model GEMV memory-access-pattern redesign** (the standing "big lever," partly de-risked).
- [ ] **Speculative decoding.**
- [ ] **PagedKvCache + continuous batching** (`MoeLayer` router/expert dispatch prerequisite).
- [ ] **VRAM leak on model swap** (high-sev, open); SSM decode benchmark harness.

## 6. Diffusion / accel-grind open levers

- [ ] **Step-cache port to Sdxl/Flux/StableDiffusion15/Sd3/Chroma/HiDream pipelines** — not a checkpoint
  gap, corrected 2026-08-10: the blocker is transformer-side plumbing, not missing weights, and the six
  targets are NOT one uniform item — they split by transformer topology (design pass done 2026-08-10,
  zero code changed):
  - **DiT-shaped (`Sd3Transformer`/`ChromaTransformer`/`HiDreamTransformer`/`FluxTransformer`)**: all four
    are a homogeneous iterative block loop, the same shape `ZImageTransformer` already wires
    `DeviceFeatureCache` into (`ForwardPacked`/`PackedCore` — run block 0, gate on drift, skip 1..Depth-1
    on a hit via a cached residual, always run the final norm/proj). `DeviceFeatureCache` itself is
    confirmed generic (backend-op based, no packed-token assumption baked in) — this is a real port here.
    `FluxPipeline`/`ChromaPipeline` additionally already use `DitStepGraph` (graph capture) and need the
    same graph-mode mutual-exclusion gate Z-Image has; `Sd3Pipeline`/`HiDreamPipeline` don't, so those two
    are the cleaner starting point.
  - **UNet-shaped (`SdxlPipeline`/`StableDiffusion15Pipeline`, sharing `UNet.cs`)**: NOT a port. A UNet's
    down-block skip connections feed the up blocks directly, so there is no "skip everything after block
    0" cut point the way a DiT has — the literal `DeviceFeatureCache` pattern applied here means skipping
    the *entire* rest of the network on a cache hit, not the established DeepCache-for-UNets technique
    (cache specific down-block features, shorten the up-path). Needs its own algorithm design, sized
    separately from the DiT group. SDXL's fused batched-CFG loop is also a single in-place device kernel
    with nothing host-visible to gate a cache on (same reason it falls back to the eager loop for
    CFG-Rescale). **Deprioritized 2026-08-10 per user judgment** — SDXL/SD1.5 are old enough that a
    from-scratch caching algorithm may not be worth the design effort; not started, revisit only if asked.
  - **Progress (2026-08-10)**: `Sd3Transformer`/`Sd3Pipeline` and `ChromaTransformer`/`ChromaPipeline` — DONE,
    real-weight verified (two independent `DeviceFeatureCache` instances for the true-CFG cond/uncond streams
    on both; SD3's context stream and Chroma's text stream are both dropped on a cache hit since neither is
    read outside the block loop). Chroma needed a different mechanical approach than SD3/Flux: its
    `ForwardDoubleRange`/`ForwardSingleRange` take `ref` params and unconditionally dispose their input on
    every call (no `ReferenceEquals`-guarded loop to alias), so the cache anchor is an explicit device-copy
    snapshot of block 0's output rather than a reused reference — block 0 runs alone via
    `ForwardDoubleRange`'s own partial-range support (the existing DiT-sharding primitive), then the rest of
    the stack runs or is skipped based on the snapshot's drift. `HiDreamTransformer`/`HiDreamPipeline` — DONE,
    real-weight verified (double-then-single-stream, image-token stream cached across the whole stack; the
    text encoders it needs — Llama-3.1-8B, dual CLIP, T5-XXL — turned out to already be present locally from
    other architectures' prior downloads, only the 17.1GB transformer itself needed fetching, deleted after
    testing). See the calibration-finding entry below for HiDream's own failure mode — its proven-safe profile
    needed the late-window gate, not just a threshold, to avoid a real image-collapse failure. `FluxTransformer`/`FluxPipeline` — DONE, real-weight verified. Wired the graph route (forcing
    it off when a cache is armed, matching `ForwardGraphable`'s own internal exclusion) and the sequential
    drainFree path (`RunPlainForward`, one instance for the guidance-embedded case, two for sequential
    true-CFG). Deliberately left the CFG-parallel branch (dual concurrent backends) unwired — its own
    concurrency hazards deserve dedicated verification, not a bundled change — so a generation using both
    step-cache and CFG-parallel simply runs uncached on that branch (silent, not incorrect). The host-step
    branch (ControlNet/Kontext/regional/masked-inpaint) needed no change: the transformer's own `cacheActive`
    gate already excludes all of it, confirmed by `Flux1RegionalPromptingRealWeightTests` passing unchanged
    after this wiring. Threshold 0.08 is Flux's own proven-safe value — a THIRD distinct number from SD3
    (0.03) and Chroma (0.15).
  - **Calibration finding (2026-08-10, applies fleet-wide, not just the new SD3/Chroma ports)**: step-cache
    profiles are uncalibrated across every pipeline that has this wired — `ZImagePipeline`'s own comment
    already says "no calibrated profile yet." Measured on SD3: the generic fallback threshold (env `"1"`/
    `"true"` → 0.10) reused 13/20 steps and produced a visibly darker, lower-detail image; an explicit
    `HARTSY_STEP_CACHE=0.03` reused 5/20 steps and stayed visually consistent. **Measured on Chroma: neither
    SD3's 0.03 nor the generic 0.10 transferred** — 0.03 produced ZERO reuses in 20 steps (Chroma's block-0
    indicator drifts faster per step than SD3's), 0.3 reused 14/20 but visibly washed out the image (same
    scene-level-failure signature as SD3's bad case), 0.15 reused 11/20 and stayed visually consistent. Two
    architectures, two different proven-safe numbers — this is not a one-off SD3 fluke, every pipeline needs
    its own number. **Measured on Flux: 0.08** (a third distinct value, between SD3's and Chroma's), reused
    4/20 steps, mean abs diff 6.91 — the cleanest result of the three, visually near-identical to cache-off.
    **Measured on HiDream: a qualitatively different, worse failure mode, not just another threshold number.**
    At its native 50-step schedule, threshold 0.08 with the whole schedule eligible didn't just degrade the
    image (SD3/Chroma/Flux's bad cases stayed recognizable) — it collapsed it into a flat, textureless color
    field (mean abs diff 77 vs. cache-off). HiDream's block-0 indicator apparently reads as low-drift EARLY in
    the schedule while the true compositional cost of skipping those blocks is severe — reusing early steps
    destroys the image before it forms, unlike SD3/Chroma/Flux where early reuse merely blurred detail.
    Restricting reuse to the back 60% of the schedule (`HARTSY_STEP_CACHE_LATE=0.6` — the late-window mechanism
    already built for Ideogram 4) fixed it completely: mean abs diff dropped to 3.55, visually near-identical.
    **Implication for the fleet:** late-window isn't just a nice-to-have knob for some pipelines — for at least
    one architecture it's the difference between "works" and "destroys the image." Any calibration pass must
    sweep the late-window dimension too, not just the threshold. Nobody has A/B'd the generic 0.10 default
    against any of the OTHER pipelines it's wired into (Z-Image, Krea2, QwenImage, Flux2, Ideogram4, Wan, LTX)
    — it may be silently over-aggressive, or silently collapsing images the way HiDream's did, on some of them.
    A real calibration pass (per-model `StepCacheProfile`, the mechanism `StepCacheEnv.Resolve` already
    supports) is real, useful, currently-nonexistent work.
  - CFG-interval late-band replicate to HiDream / Wan.
- [ ] F16-ingest / F16-out Sage attention kernel (designed, build next).
- [ ] Wan2.2-Lightning / LTX-distilled loadable accelerators.
- [ ] **TensorRT compile support** — zero TRT code anywhere in the engine today; no design started.
- [ ] **LoRA extraction / checkpoint-diff utility** — zero checkpoint-diff/extraction code anywhere;
  no design started.
- [ ] **Seamless tiling / circular conv padding** — neither cuDNN's convolution graph API nor the im2col
  fallback kernel support anything but zero padding, engine-wide (not per-model). Needs a tensor-level
  wrap-pad utility that copies wrapped edge rows/columns into an explicit padded buffer before calling the
  existing conv paths with zero additional padding requested.
- [ ] **CPU-offloaded activations** — weight offload works because the host `Tensor` is always the
  authoritative copy (`PreloadWeights`/`FreeWeights` just add/drop a GPU-side cache entry); activations have
  no host-authoritative copy today (`FreeActivations` discards the device buffer outright, no D2H anywhere
  in that path). Needs a genuine materialize-to-host-then-reload path, plus interaction with the CUDA
  step-graph invalidation that already exists for the weight-free case (`StepGraphInvalidateForActivationFree`)
  — a captured graph bakes activation pointers, so offloading mid-graph needs the same invalidation treatment.
- [ ] **PAG / SAG attention-hook infrastructure** — no self-attention substitution/introspection mechanism
  exists anywhere (confirmed zero hits for any `AttentionHook`-shaped primitive); `FluxTransformer`'s
  ControlNet residual-injection points are an external-adapter residual-add, not reusable as-is for a
  perturbed/identity-attention branch. Needs its own design pass (where to fork the branch, how to blend it
  back, which pipelines get it first) before implementation — cross-cutting quality infra, not one model's gap.
- [~] **`VaeTiledEncoder`/`VaeTiling`** (pixel-space-tiled VAE encode, for large img2img/inpaint inputs that
  today go through the encoder untiled) — dtype-safety fixed 2026-08-09 (was F32-hardcoded, silently
  corrupts against BF16 weights) and unit-tested, but wiring it into a real pipeline (`SdxlPipeline`'s
  img2img encode path was the first attempt) reproducibly **segfaults inside `libcuda.so`** the moment the
  BF16 cuDNN conv fast path first engages for the VAE encoder's downsample convs (100% reproducible at
  1536x1536 SDXL img2img; crash site is the driver itself, not managed code — this dtype-correctness fix is
  the first caller to ever feed the encoder a dtype-matched, BF16, tile). The wiring was reverted pending a
  dedicated CUDA-driver-level investigation; the class itself has zero production callers today.

## 7. Robotics models (new modality — greenfield)

- [ ] Scope target robotics/action models (VLA-style) and the modality's request/result DTOs.
- [ ] Backend ops + recipe pattern; a `MODEL_STATUS_ROBOTICS.md` once the first model is scaffolded.

## 8. New SwarmUI extensions

- [ ] **3D extension** (image→mesh) — surface the 3D modality as its own Swarm extension.
- [ ] **World-model extension** — interactive/world models as a Swarm extension.
- [ ] Keep each thin over `HartsyInference.Engine` (no re-implemented load/generate orchestration).

## 9. CLI / API

- [ ] **`VideoAudioReference` / `VideoAudioInput` feature gating:** both are typed video-request conditioning with no
  `VideoFeatures` bit — a family that ignores them silently drops the audio (the exact bug class the 2026-08-07
  `ReferenceImages`/`ReferenceVideos`/`ReferenceAudios`/`DrivingVideo` bits closed for the other inputs). Same pattern:
  add bits, wire `VideoService.RequestedFeatures`, declare on WanS2V/H3.

- [ ] **HTTP public API reference doc** + API stability freeze (release gate).
- [ ] Wire the **T5/seq2seq generation loop** in `TextService` (currently unwired).
- [ ] Finish CLI catalog coverage + real-hardware verification runs (LLM ~26 entries, video recipes:
  `kandinsky5-video`, `cosmos-predict1-5b/13b` need `IVideoRecipe` wrappers; HunyuanVideo/LTX-2 verified
  but not surfaced). Assets/sha256 sourcing per entry.
- [ ] `--image` VLM flag, `--thinking`/`--no-thinking` real-weight verification.
- [ ] Per-provider audio unload endpoint.

## 10. Release / production hardening

From `PRODUCTION_RELEASE_CRITERIA` (~70% done) — the 1.0.0 gate.

- [ ] Full ~26-architecture verification sweep (real weights).
- [ ] Graceful shutdown / request draining; per-request timeout + max-size limits.
- [ ] `/metrics` endpoint; multi-hour SOAK test.
- [ ] Decode-round & native-crash fault isolation verified under load (see `TROUBLESHOOTING.md` §Serving).

## 11. NuGet publication

From `RELEASE_NUGET` — mostly unstarted; persist through the 1.0 release.

- [ ] Branding/metadata pass across all packages; license/readme per package.
- [ ] Quality gates (build/test/pack) in CI; symbol packages.
- [ ] Publish alpha → beta → stable progression; verify package-boundary dependency graph.
