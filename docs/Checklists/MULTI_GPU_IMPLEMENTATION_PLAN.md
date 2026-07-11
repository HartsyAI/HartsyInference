# Multi-GPU Sharding — Implementation Plan

## Context

HartsyInference today is **single-GPU only**: each `CudaBackend` wraps one device (`CudaBackend(int deviceOrdinal)`), a model holds one `IBackend`, and every op routes through it. That caps us at models which fit one card. The biggest LLMs we want (Kimi-K2 1T, DeepSeek-V3 671B, Mixtral, large MoE) are marked **Build-defer** in `docs/Checklists/LLM_MODEL_COVERAGE.md` precisely because they exceed one GPU. We also want to serve two very different audiences with one codebase:

- **Consumer "string together cheap GPUs"** — e.g. several GTX 1080 Tis. Floor = **compute capability 6.1 (Pascal)**, PCIe-only (no NVLink), F32/F16 (no tensor cores/bf16/fp8). Pin **CUDA 12.x** (Pascal offline codegen dropped in 13.0).
- **Enterprise "a building of H200s"** — NVLink/NVSwitch intra-node + InfiniBand/RoCE inter-node; full parallelism + disaggregated serving.

Full external research lives in `docs/Research/MULTI_GPU_PARALLELISM.md`. **Two decisions are locked there:** (1) target **both** tiers with one capability-gated codebase; (2) use **NCCL** as the collective backend (BSD-3 licensed, the same category as the **cuBLAS/cuBLASLt the engine already P/Invokes** — per the library-exception policy, don't reinvent collectives). RCCL is a symbol-compatible drop-in for AMD via a library swap. NCCL needs no InfiniBand/NVLink/MPI: it auto-uses NVLink→PCIe P2P→SHM intra-node and IB/RoCE→**TCP sockets** inter-node, bootstrapped by a 128-byte `ncclUniqueId` we broadcast ourselves over a tiny C# TCP rendezvous.

The intended outcome: a model 2-to-N× too large for one GPU runs across multiple GPUs (and, later, multiple nodes), fast on NVLink and at least *functional* on cheap PCIe rigs, with the parallel plan auto-selected from the detected hardware topology.

## Scope & sequencing

Built as independent, separately-shippable milestones. **M1 (layer split) is the first deliverable** — it needs no collective at all and is verifiable on a 2-GPU consumer rig. Later milestones add NCCL and the high-performance modes. Each milestone is gated honestly by hardware (slice-test loaders/collectives against a CPU reference like Phase 8a did for MLA; defer e2e throughput claims to real multi-GPU access, per the existing Build-defer policy).

**Non-goals (this plan):** training, Vulkan multi-GPU (CUDA first; RCCL/AMD is a later lib-swap), and reaching parity with vLLM/SGLang's most advanced overlap tricks (two-batch overlap, EPLB rebalancing) in the first cut.

---

## Architecture

A new package **`HartsyInference.Distributed`** holds device/topology/parallel-plan abstractions and the orchestration that sits *above* the per-device `IBackend`s. CUDA-specific additions (peer bindings, `NcclApi`, a `NcclCommunicator`) go in `HartsyInference.Cuda`. The `GenericTransformer` spine gains a `ParallelConfig` and a multi-backend execution path, mirroring the engine's existing "preset + knob, not a new class" philosophy.

### New CUDA-layer pieces (`src/HartsyInference.Cuda/`)
- **Peer-access P/Invoke** in `CudaDriverApi.cs`: `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, `cuCtxDisablePeerAccess`, `cuMemcpyPeerAsync`/`cuMemcpyPeer` (the layer-split boundary copy + topology probe). Currently absent (only `cuMemcpyDtoD` exists).
- **`NcclApi.cs`** — new file, follows `CublasApi.cs` verbatim (`[LibraryImport(LibName="nccl")]`, `internal static partial`). Bind: `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommInitAll`, `ncclCommDestroy`/`Abort`, `ncclGetVersion`/`ErrorString`; collectives `ncclAllReduce`, `ncclAllGather`, `ncclReduceScatter`, `ncclBroadcast`; P2P `ncclSend`/`ncclRecv` + `ncclGroupStart`/`ncclGroupEnd` (grouped send/recv *is* all-to-all — there is no public `ncclAllToAll`).
- **`CudaLibraryResolver.cs`** — add an `nccl` branch (`libnccl.so.2` / `nccl.dll`), exactly like the existing `cublas` version-probe.
- **`NcclCommunicator.cs`** — managed wrapper owning one `ncclComm_t` per local rank, exposing `AllReduce/AllGather/ReduceScatter/Broadcast/AllToAll(Tensor…, CudaStream)`; built either single-process via `ncclCommInitAll` (single node) or per-process via `ncclCommInitRank` after a unique-id broadcast.
- **Fix single-device static state** so N backends coexist live: `GpuTransferHelper._context` and `CudaMemory` compute-stream/pool statics must become per-device (keyed by ordinal, or instance state threaded through). This is the one real refactor in the CUDA package.

### New `HartsyInference.Distributed` package
- **`GpuTopology`** — startup probe: `cuDeviceGetCount`, per-device compute capability + free VRAM, and a P2P-reachability matrix via `cuDeviceCanAccessPeer` (probe + fall back to SHM/host; never assume — 1080 Ti P2P is BIOS/IOMMU-fragile). Detects NVLink vs PCIe.
- **`ParallelConfig`** — `{ int TpSize, PpSize, EpSize, DpSize; int[] DeviceOrdinals; TransportKind }`, plus a `Plan(GpuTopology, modelBytes)` factory that picks a sane plan: layer-split when no fast link, add TP/EP when NVLink/RDMA present, "TP = GPUs per node, PP = nodes."
- **`DeviceMesh`** — maps (tp, pp, ep, dp) coordinates → device ordinal / global rank; the bridge the transformer queries to know where a layer/expert/shard lives.
- **`TcpRendezvous`** — pure-C# `System.Net.Sockets` service: rank 0 generates the `ncclUniqueId`, broadcasts the 128 bytes, assigns ranks/world-size. Replaces MPI; single-node skips it (uses `ncclCommInitAll`).

### `GenericTransformer` spine changes (`src/HartsyInference.LLM/Transformer/`)
- `TransformerConfig` (or a sibling) carries an optional `ParallelConfig`.
- A new `MultiGpuExecutor` (or extend the model holder) owns **`IBackend[] backends`** (one per local device) instead of a single backend, and drives the per-milestone execution path below. The existing `ForwardEmbeds(startLayer,endLayer)` parameters are the layer-split seam — no core math changes for M1.

---

## Milestones

### M0 — Foundation (no model behavior change yet)
- Add peer-access bindings + `GpuTopology` probe + `ParallelConfig`/`DeviceMesh` + the `GpuTransferHelper`/`CudaMemory` per-device-state fix.
- Unit-test: enumerate devices, build a P2P matrix, instantiate 2 `CudaBackend`s and round-trip a tensor device→device via `cuMemcpyPeerAsync` (host-staged fallback when P2P is false).
- **Verifiable on any 2-GPU box (incl. 2× cheap PCIe cards).**

### M1 — Layer split / pipeline (the first real capability) — collective: **none**
Partition `L` layers into `S` contiguous stages by per-GPU free VRAM (or explicit ratios, llama.cpp `--tensor-split` semantics). Each stage's `CudaBackend` preloads only *its* layers' weights (`GenericTransformer.LoadWeights` already supports a layer range; `EnumerateWeights` filtered per stage) and owns the KV cache for its layers (`FixedKvCache` is already per-layer arrays — allocate per stage). Forward = run `ForwardEmbeds(startLayer=stage.lo, endLayer=stage.hi)` on each backend in order, copying the single residual activation `[b,s,h]` (or `[b,1,h]` decode) across the boundary with `cuMemcpyPeerAsync` + a `cuEvent`/`cuStreamWaitEvent` handoff (reuse the `CudaStreamingWeightCache` event pattern). Embed on device 0, `ProjectLogits` on the last device.
- **Enables:** run a model 2-N× too big for one GPU. **Memory scales, latency does not** (one stage active at a time; single-request decode has the pipeline bubble — that's expected and fine).
- **Composes with block-swap** (`MEMORY_SCHEDULING_SERVING.md`): layer-split across GPUs + block-swap on each = model bigger than the *sum* of VRAM.
- **Verify:** a 20-24 GB model split across 2× consumer cards (e.g. 2×12 GB) produces token-identical output vs the same model on one big GPU (greedy). No NVLink needed.

### M2 — Tensor parallel — collective: **NCCL all-reduce** (`NcclApi` + `NcclCommunicator` land here)
Column/row-parallel loaders: shard each weight `[out,in]` along `out` (column, QKV + gate/up) or `in` (row, attn-out + down), splitting attention by head. Insert **2 `ncclAllReduce`/layer** (after attn-out, after MLP-down). Vocab-parallel embedding/lm_head with an all-reduce. KV cache holds only each rank's KV-head subset (requires head count divisible by `tp`).
- **Enables:** real **single-request latency** speedup — the "fast" mode.
- **Reality check:** pays off on **NVLink/NVSwitch**; on PCIe-only >2 GPUs the tiny all-reduces are latency-bound and lose (so consumer rigs stay on M1). NCCL handles transport selection + the P2P-fragility internally.
- **Verify:** slice-test the sharded linear + all-reduce against the single-GPU result (bit-exact within F32 tol) on a CPU/2-GPU reference; e2e throughput claims deferred to NVLink hardware.

### M3 — Expert parallel (MoE: Kimi-K2 / DeepSeek-V3 / Mixtral) — collective: **NCCL grouped send/recv (all-to-all)**
Distribute experts across devices (reuse the existing `MoeFeedForward` + `GgufLanguageModel.SplitStackedExperts` per-expert split; assign expert ranges via `DeviceMesh`). MoE layer = **dispatch all-to-all** (route tokens to expert-owner) → local expert GEMM → **combine all-to-all**. The current host-side top-k routing in `MoeFeedForward.Forward` is the natural place to compute the per-rank dispatch lists. DeepSeek node-limited routing already exists (`RouteGroupLimited`).
- **Enables:** scale MoE expert weight past one GPU — unblocks the Build-defer giants.
- **Verify:** expert-parallel MoE output matches single-GPU MoE on a small MoE (OLMoE/Qwen-MoE already verified single-GPU) split across 2 GPUs.

### M4 — DP-attention + EP hybrid (the efficient DeepSeek/Kimi recipe) — collective: **all-gather + all-to-all**
For MLA models, replicate attention per DP rank (each stores only its small latent KV — avoids the TP duplication of MLA's single KV head), `ncclAllGather` hidden states before the MoE layer, EP the experts, redistribute back. Builds on M2+M3.
- **Enables:** the memory-efficient way to serve DeepSeek-V3/Kimi-K2 at scale.

### M5 — Disaggregated prefill/decode (datacenter throughput) — collective: **KV transfer**
Separate prefill and decode worker pools; a `KVConnector` abstraction (vLLM's term) moves KV cache prefill→decode. Start with NCCL/TCP transfer; RDMA/NIXL-style later. Pairs with multi-node `TcpRendezvous`.
- **Enables:** datacenter goodput/SLO at many concurrent requests. Highest effort, datacenter-only.

---

## Image & video diffusion multi-GPU (answering "is it possible/recommended?")

**Short answer: yes, and worthwhile — but the methods differ from LLMs, and plain LLM tensor-parallel is the *wrong* tool here.** (Sources in `docs/Research/MULTI_GPU_PARALLELISM.md`; xDiT/PipeFusion/DistriFusion are the references.)

Key facts that shape it:
- For **images**, multi-GPU is mostly about **throughput**, not fit (SD1.5 ~2 GB, SDXL ~7 GB, Flux 12B ~24 GB all fit one big card). For **video** (Wan/LTX/Hunyuan — the engine has 8+ video pipelines), it's about **fit *and* single-video latency**, because activation memory scales with the latent sequence (frames×H×W). This is the higher-value case for us.
- The denoise loop is sequential like LLM decode, so a **naive layer-split pipeline is a poor fit** (one bubble per denoising step). Sequence/patch parallelism is the right shape.

Recommended diffusion track (separate from the LLM milestones, lower urgency, but reuses M0 foundation + `NcclCommunicator`):
- **D1 — Batch/data parallel (throughput):** replicate the pipeline per GPU, one image/video per GPU. Near-linear, zero numerical risk, trivial to validate. Best ROI for a generation *service*. Mostly orchestration above existing pipelines.
- **D2 — CFG parallel (latency, ~2× cap):** the codebase already runs **two separate transformer forwards per step** (`FluxPipeline`/`Sd3Pipeline` do `vCond`/`vUncond` separately) — put the two branches on two GPUs, one tiny all-gather/step. Low effort, exact. (A cheaper single-GPU win — batching the two into one forward — is also worth doing regardless.)
- **D3 — Sequence/context parallel (video latency, the real win):** split the patch/latent sequence across GPUs with Ulysses (all-to-all around attention; needs `num_heads % N == 0`) then Ring for small-head models. This is exactly what the official Wan/Hunyuan/CogVideoX repos use, near-linear, **mathematically exact** (validate attention output bit-for-bit). Requires a backend-level sequence-parallel attention path (touches `ScaledDotProductAttention`/`FlashAttention`), so it's the highest-effort diffusion item.
- **Defer:** PipeFusion/DistriFusion (stale-activation patch parallel) — approximate, needs per-model FID validation that collides with our strict reference-parity rule; revisit only for *fitting* oversized DiTs. **Skip diffusion tensor-parallel entirely** (xDiT rejects it — scales badly on long sequences).
- **Already done / reuse:** spatial VAE tiling (`VaeTiledDecoder`/`VaeTiledEncoder`) and component eviction (T5 freed after encode) exist; VAE patch-decode can run data-parallel.

---

## Key files

**Create:**
- `src/HartsyInference.Cuda/NcclApi.cs`, `src/HartsyInference.Cuda/NcclCommunicator.cs`
- `src/HartsyInference.Distributed/` (new package): `GpuTopology.cs`, `ParallelConfig.cs`, `DeviceMesh.cs`, `TcpRendezvous.cs`, `MultiGpuExecutor.cs`
- Tests: `tests/HartsyInference.Cuda.Tests/PeerAccessTests.cs`, `tests/HartsyInference.LLM.Tests/LayerSplitTests.cs`, `MultiGpuMoeTests.cs`

**Modify:**
- `src/HartsyInference.Cuda/CudaDriverApi.cs` (peer bindings), `CudaLibraryResolver.cs` (nccl branch), `GpuTransferHelper.cs` + `CudaMemory.cs` (per-device static state).
- `src/HartsyInference.LLM/Transformer/TransformerConfig.cs` (carry `ParallelConfig`), `GenericTransformer.cs` (multi-backend layer-split path via existing `startLayer/endLayer`), `MoeFeedForward.cs` (expert-parallel dispatch, M3).
- `src/HartsyInference.LLM/Generation/GgufLanguageModel.cs` (per-stage/per-rank weight load), `TextGenerationPipeline.cs` (drive multi-backend), `samples/HartsyInference.TextGen.Cli/Program.cs` (multi-GPU CLI flags: `--gpu-split`, `--tp`, `--ep`).

**Reuse (do not rebuild):** `CudaStreamingWeightCache` event/overlap pattern (cross-device handoff), `FixedKvCache` per-layer arrays, `SplitStackedExperts`/`SplitFusedPhi`, `RouteGroupLimited`, `VaeTiledDecoder`, block-swap (`MEMORY_SCHEDULING_SERVING.md`).

## Verification

- **M0:** xUnit — device enumeration, P2P matrix, 2-backend tensor round-trip (peer + host-staged fallback paths).
- **M1 (primary milestone gate):** split a 20-24 GB GGUF across 2 GPUs; assert **token-identical greedy output** vs single-GPU (env-gated test like `GgufEndToEndTests`, `HARTSY_TEST_MULTIGPU`). Runs on cheap PCIe cards.
- **M2/M3:** slice-test sharded-linear+all-reduce and expert-parallel-MoE against the single-GPU result within F32 tolerance (CPU/2-GPU reference, no NVLink needed); e2e throughput deferred to NVLink/datacenter hardware and logged honestly in `LLM_MODEL_COVERAGE.md` / `PARITY_VERIFICATION.md`.
- **Diffusion D1-D3:** D1/D2 assert image-identical vs single-GPU; D3 asserts attention output bit-exact (sequence parallel is exact).
- Build per the repo convention (`dotnet build -m:1` where the audio/CUDA notes require it); keep the no-managed-array hot-path + reference-parity rules.

## Risks / open items
- **Single-device static state** (`GpuTransferHelper`, `CudaMemory`) is the one non-trivial refactor; must make N live backends safe (M0).
- **1080 Ti P2P fragility** — always probe `cuDeviceCanAccessPeer`, fall back to SHM/host; NCCL handles this internally for M2+.
- **CUDA-graph capture across devices** — unverified whether `CudaGraph` spans a multi-device step or must be per-device.
- **Process- vs thread-per-GPU** — threads single-node (`ncclCommInitAll`, no GIL in .NET), processes multi-node (`ncclCommInitRank` + `TcpRendezvous`).
- **Verification hardware** — most TP/EP/disaggregation throughput can only be honestly claimed on NVLink/datacenter access; the dev box is one 12 GB 3060. M1 + all slice tests are verifiable on cheap consumer multi-GPU.
- **NuGet:** new `HartsyInference.Distributed` package + version bump (per `nuget-version-bump` memory); NCCL is a runtime native dep resolved like cuBLAS (not bundled).
