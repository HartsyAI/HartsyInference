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

- [ ] **M0 — device topology + placement:** enumerate GPUs, PCIe/NVLink topology, per-device VRAM
  budget; a placement planner that assigns layers/experts to devices.
- [ ] **M1 — layer-split (pipeline) parallel:** shard transformer blocks across devices, micro-batch the
  pipeline; the entry path for "many small cards."
- [ ] **M2 — tensor parallel:** split individual GEMMs (QKV, MLP, lm_head) across devices; all-reduce on
  the seam.
- [ ] **M3 — expert parallel (MoE):** route experts to devices; ties to on-device MoE routing (§4).
- [ ] **M4 — DP-attention / sequence parallel:** for long context and video (Ulysses-style seq-parallel,
  see §2 D3).
- [ ] **M5 — disaggregated serving:** separate prefill and decode pools.
- [ ] **Collectives:** NCCL (NVIDIA) via Driver-API P/Invoke; **RCCL** for the AMD/ROCm path (§3).
- [ ] **Tiers to validate:** consumer multi-card (e.g. 2–8× 8 GB, Pascal+) and enterprise (H200-class).
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
  a `HARTSYINFERENCE_VK_PROFILE=1` check before any further tuning. **A second finding was found AND
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
- [ ] **coopmat2** (`VK_NV_cooperative_matrix2`): workgroup-scope tiles, fused dequant-on-load,
  `coopMatReduceNV`-fused softmax, bias/activation epilogue fusion (`VULKAN_OPTIMIZATION.md` §4.2). All
  capability flags confirmed present on the 3060 — genuine reachable ceiling, but explicitly NVIDIA-only
  and the highest-effort item in this section. Deliberately deferred past this Phase 5 pass (user chose
  "portable items now, skip coopmat2" — 2026-07-29) in favor of INT8 wiring/dequant/flash-attention, which
  land on every vendor exposing the base features; revisit once a coopmat1 vs coopmat2 measured gap on the
  3060 justifies the NVIDIA-only investment.
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
- [ ] **`VulkanStepGraph`** (the plan's Phase 6e, the CUDA-Graph-capture analog backing
  `StepGraphBegin`/`StepGraphEndAndLaunch`/`StepGraphLaunch`/`StepGraphReset`/`StepGraphOwner` — used far
  more broadly than LLM decode: nearly every DiT model in this codebase (Flux/Flux2/Krea2/ZImage/Chroma/
  HunyuanVideoDit/LtxVideo(2)/Kandinsky5/OasisDit/Hunyuan3DDit/F5Dit/MusicGen) gates a full per-step forward
  pass onto it). **Deliberately NOT built** (2026-07-30) — this is a correctness hazard on THIS box's own
  descriptor-management scheme, not just an effort estimate: `VulkanBackend.Dispatch()` allocates a
  descriptor set from a **recycling pool ring** (`_descriptors.AllocateSet`) on every single dispatch. A
  frozen command buffer that `vkCmdBindDescriptorSets` a set from that ring, then gets replayed after the
  ring has cycled back and overwritten that set's bindings for an unrelated later dispatch, is undefined
  behavior — the exact shape of the `WriteScalarBuffer` race just found and fixed above (passed on the
  3060, failed on llvmpipe). Doing this correctly needs (a) a dedicated, never-recycled descriptor pool for
  captured sets, and (b) a bump-pointer arena that transient allocations route through during capture so a
  replayed command buffer's bindings point at stable, deterministically-reproduced offsets instead of
  fresh `VkBuffer` handles every call — a rearchitecture of the ~40 `Dispatch()` call sites' allocation
  path, not a new file dropped alongside them. Not building it breaks nothing today:
  `StepGraphSupported => false` (IBackend's default, unoverridden) means every one of those ~13 DiT models
  runs its normal eager per-op path on Vulkan, correctly, just without the graph-capture speedup — the
  Krea2 e2e comparison this session ends with will honestly reflect "Vulkan without graph capture," which
  is the true current state, not a claim this item quietly closed.
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
  "denoise-loop graph capture reusing Phase 6's command-buffer-reuse mechanism" item — a substantial,
  separate perf-engineering effort, not attempted this session beyond confirming it isn't a quick fix.
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

- [ ] Step-cache replicate to Chroma / HiDream (🔒 no weights); CFG-interval late-band replicate to
  HiDream / Wan.
- [ ] F16-ingest / F16-out Sage attention kernel (designed, build next).
- [ ] Wan2.2-Lightning / LTX-distilled loadable accelerators.

## 7. Robotics models (new modality — greenfield)

- [ ] Scope target robotics/action models (VLA-style) and the modality's request/result DTOs.
- [ ] Backend ops + recipe pattern; a `MODEL_STATUS_ROBOTICS.md` once the first model is scaffolded.

## 8. New SwarmUI extensions

- [ ] **3D extension** (image→mesh) — surface the 3D modality as its own Swarm extension.
- [ ] **World-model extension** — interactive/world models as a Swarm extension.
- [ ] Keep each thin over `HartsyInference.Engine` (no re-implemented load/generate orchestration).

## 9. CLI / API

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
