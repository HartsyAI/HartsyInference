# Engineering roadmap

Cross-cutting open work only. Model-specific work belongs in [model status](MODEL_STATUS.md).
Implementation is not verification: recheck the named code path before starting an inherited item.
Measurements live in [scoreboards](../../benchmarks/scoreboards/); historical investigations are in git.

## 1. Multi-GPU / model sharding

Existing placement, layer/block sharding, CFG/context parallelism, tensor-parallel primitives, and
collectives are described in [MULTI_GPU.md](../MULTI_GPU.md). Remaining work:

- [ ] Tensor parallel: threaded driver and broader real-model/hardware verification; do not rebuild existing primitives.
- [ ] Expert parallel: on-device MoE routing and per-device expert placement.
- [ ] Context parallel: >2 ranks, NCCL exchange, Ulysses/head parallelism, CP×CFG composition, broader video families.
- [ ] Validate datacenter P2P/NVLink and ≥3 physical GPUs; same-device multi-rank tests do not establish that coverage.
- [ ] Disaggregated prefill/decode serving.
- [ ] Longer same-GPU concurrency soak before changing its opt-in default.
- [ ] Placement budgeting: charge lm_head bytes to the final stage; review audio eviction across all shard backends.
- [ ] Cache mllama vision features per stage instead of copying each token; avoid uploading CosyVoice's unused final head.
- [ ] GameCraft BF16-cast/cache policy validation when its checkpoint is available.
- [ ] CrossAttentionBlock cache pinning/teardown if components become shared or mid-loop activation eviction is introduced.

## 2. GPU kernel performance

Benchmark infrastructure already exists in [benchmarks](../../benchmarks/README.md).

- [ ] Refresh matched baselines across consumer, Ada, A100, and Hopper hardware.
- [ ] General/packed variable-length attention; audit existing attention paths before designing another kernel.
- [ ] QKV/gate-up/norm/activation fusion where full-model profiling shows launch or bandwidth cost.
- [ ] Extend memory reuse and graph capture only on measured eligible paths; both mechanisms already exist.
- [ ] F16/BF16 coverage and Hopper FA3/WGMMA experiments with numerical gates.
- [ ] Winograd conv only if VAE/UNet 3×3 stride-1 convolution is a measured bottleneck.
- [ ] Backfill source for legacy handwritten PTX opportunistically; preserve intentional handwritten MMA kernels.
- [ ] GPU marching cubes and DINO fusion. Video-specific primitives belong in the video status checklist.

## 3. AMD / ROCm + cross-vendor (Vulkan) support

- [ ] Real AMD/Intel validation; llvmpipe is useful for small-subgroup correctness, not hardware/performance proof.
- [ ] Subgroup-size pinning, im2col 64-bit indexing, descriptor-pool timeline lifetime checks.
- [ ] Real-model Vulkan decode parity/throughput before enabling GraphDecodeSupported by default.
- [ ] INT8 loading policy, end-to-end quality gate, cached quantized-weight lifetime, and shape-specific tuning.
  The opt-in Linear primitive already exists; that alone does not establish model support.
- [ ] Measure actual GPU time with VulkanGpuTimer before attributing full-model latency to a specific op.
  Host recording time and time collected at Sync are not interchangeable with GPU timestamps.
- [ ] Improve GEMM/attention throughput, fused dequant, epilogues, and tile selection where measured.
  Coopmat2 and partial-M support already exist. Do not restart the superseded engagement investigation.
- [ ] Review empty dedicated-block eviction for VRAM headroom; earlier claims that allocator cost explained
  the throughput gap were retracted. Profile missing EnterOp scopes before trusting an op breakdown.
- [ ] Verify reused-output CacheActivation ownership; historical diagnostic reported a replaced buffer leak.
- [ ] Graph capture: account for retained transient/cast buffers; measure a single-block benefit before
  building a reuse arena. Capture OOM with clean eager fallback is not itself a leak.
- [ ] Barrier scoping only after isolated measurement justifies it.
- [ ] Model-driven domain fill: volumetric conv, grid-sample/MSDA, and audio fused ops.
- [ ] RCCL collectives on AMD hardware.

References: [Vulkan scoreboard](../../benchmarks/scoreboards/VULKAN.md),
[Troubleshooting](TROUBLESHOOTING.md), [Vulkan research](../Research/VULKAN_OPTIMIZATION.md).

## 4. Quantization

- [ ] Broaden native FP8 end-to-end validation and document default policy per GPU/model; existing paths are built.
- [ ] W8A8: meet the existing end-to-end quality gate (SmoothQuant previously failed SSIM 0.95).
  Investigate grouped-K, mixed-precision layer selection, or timestep-aware calibration; do not weaken the gate.
- [ ] Extend low-VRAM GGUF to unsupported formats/models; basic diffusion GGUF wiring already exists.
- [ ] Quantized KV beyond existing F16 storage; distinguish storage precision from paged-cache integration.
- [ ] MMQ-style quantized prefill and missing fused dequant/GEMV variants; several common variants already ship.
- [ ] IQ1/IQ2/IQ3 codec coverage: inspect the registry for the exact missing formats.

## 5. LLM decode / serving throughput

- [ ] On-device MoE routing and non-greedy sampling; remove per-token host synchronization on affected paths.
- [ ] FP16 activation coverage, projection fusion, and large-model GEMV tuning against measured bottlenecks.
- [ ] Integrate/validate continuous batching and paged KV through Engine/API. DynamicBatchScheduler and
  PagedKvCache exist in LLM, but TextService does not currently use the scheduler.
- [ ] Extend and measure existing speculative decoding; it is already exposed by TextService.
- [ ] Reproduce the inherited model-swap VRAM-leak report before assigning a cause; audit all placement backends.
- [ ] Streaming lifetime: cancellation before worker startup, consumer abandonment, bounded buffering,
  and error completion (TextService.StreamAsync is a priority source-review finding).
- [ ] SSM recurrence profiling and long-context/quantized quality checks; per-architecture gaps are in LLM status.

## 6. Diffusion / acceleration

- [ ] Calibrate step-cache profiles per remaining family, including late-window limits; generic thresholds
  can silently collapse output. Preserve already verified per-model profiles.
- [ ] SDXL/SD1.5 UNet step-cache design remains deferred unless requested; extend CFG late-band work where useful.
- [ ] F16 input/output Sage attention and loadable few-step accelerators, subject to quality gates.
- [ ] LoRA extraction/checkpoint-diff utility.
- [ ] PAG/SAG attention hooks and per-pipeline regional-tag handling.
- [ ] Pixel-space tiled VAE encode: reproduce the documented BF16 CUDA crash at 1536² SDXL img2img.
  VaeTiledEncoder exists but production wiring was reverted; a source-only dtype fix is not a working feature.

TensorRT support requires a separate architecture decision: it is not an approved pure-C# inference implementation.
Model-specific SDXL refiner/segment gaps belong in [image status](MODEL_STATUS_IMAGE.md).

## 7. Robotics models

- [ ] Scope a target model and request/result contract before creating a new modality package or checklist.

## 8. SwarmUI extensions

- [ ] 3D and world-model UI extensions, thin over Engine.
- [ ] Validate package/extension compatibility for newly exposed features; H3 gates live in video status.

## 9. CLI / API

- [ ] VideoAudioReference/VideoAudioInput capability bits and rejection on unsupported recipes.
- [ ] HTTP API reference and stability review against actual endpoint/request types.
- [ ] T5/seq2seq generation in TextService; implementation inside a model is not catalog/CLI reachability.
- [ ] Audit catalog assets, hashes, and real consumer-path coverage against per-model evidence; avoid stale entry counts.
- [ ] Per-provider audio unload endpoint.

## 10. Release / production hardening

- [ ] Graceful shutdown/draining and per-request timeout/size policy across modalities. Native video already
  has MaxVideoRequestBodyBytes; fast/long queues and per-caller rate limiting already exist.
- [ ] /metrics and multi-hour soak: memory plateaus, cancellations, OOM recovery, repeated model swaps.
- [ ] Model-level readiness: /ready currently checks BackendDescription, not loaded-model worker health.
- [ ] Verify decode faults and native-process recovery under load through the real Engine/API path.
- [ ] Complete applicable real-weight release gates; use missing-weight-fails campaigns when claiming completion.

## 11. NuGet publication

Publication already exists in publish-nuget.yml. Directory.Build.props owns version/metadata.

- [ ] Fresh-consumer build/pack and dependency-boundary verification before stable release.
- [ ] Alpha→beta→stable criteria and public API compatibility policy.
- [ ] Symbols/SourceLink only when the current closed-alpha policy changes; they are intentionally disabled.

Dedicated CPU/GPU CI was removed by project decision; do not recreate it as routine checklist cleanup.

## 12. Cleanup follow-ups

- [ ] TensorPool: adopt on a demonstrated hot path or retire (no established production usage).
- [ ] Re-run analyzer diagnostics before acting on the historical snapshot; judge config/API mirrors individually.
- [ ] LtGemmExecutor consolidation only after Fp4 block-scale attributes are supported.
- [ ] Vulkan spec/push-constant builders only if parameterization improves the actual differing contracts.
- [ ] QwenImageVaeOps.FlattenGamma: clarify borrowed-vs-owned return behavior across dtypes.
- [ ] VocosConfig.AdaNormNumEmbeddings/Padding: unused options must not imply supported behavior.
- [ ] WeightNormFusion.LoadFused: remove or deterministically dispose redundant outer F32 conversions.
- [ ] Review remaining MixContract/SplitContract, CFG-step, and interleaved-RoPE duplication with numerical gates.
- [ ] CLI repeated settings/runner helpers only where behavior actually matches.
- [ ] VideoRequest.VideoModel/VideoFormat: check extension consumers before altering public DTOs.
