# Memory, Scheduling & Serving — Research Notes

Memory management, transfer overlap, VRAM offload, and execution-scheduling techniques for HartsyInference, the pure-C#/.NET Driver-API engine. Dev box is a 12 GB RTX 3060 on PCIe 3.0 x16 (~12-13 GB/s effective pinned H2D), which must run models far larger than VRAM (Flux 12B ~24 GB bf16, Wan/LTX/Hunyuan video). Everything here maps to CUDA Driver-API P/Invoke against `nvcuda`/`libcuda`, consistent with the no-native-libs pillar.

This complements the GPU-resident caching already described in [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md) (weight cache, lazy-sync activation cache, `cuMemFreeAsync`, UNet eviction before VAE decode). The current engine has no real memory pool, no pinned host memory, no async H2D/D2H, no multi-stream overlap, and no block-swap offload. Those five are the gaps this doc targets.

---

## 1. CUDA stream-ordered allocator (`cuMemAllocAsync` / `cuMemFreeAsync` + pools)

**Mechanism.** `cuMemAllocAsync` inserts an allocation into a stream and returns a pointer immediately (do not touch the memory until the stream reaches that op); `cuMemFreeAsync` schedules a stream-ordered free. Freed blocks return to a per-device **memory pool** rather than the OS, so a later `cuMemAllocAsync` reuses them without a real driver allocation. Cross-stream reuse is allowed via `CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES` and `CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES` (default on).

**Release threshold (the key knob).** `cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &bytes)`. Default **0** means the pool returns pages to the OS at the next sync. Set it high (e.g. `UINT64_MAX`) so the pool never releases during sync, keeping pages cached for reuse. This is the single most important setting for steady-state inference with a stable per-step working set. On a 12 GB card you must still **cap** retained cache or you OOM; expose a trim hook (`cuMemPoolTrimTo`).

**What it saves.** Eliminates repeated synchronizing `cuMemAlloc`/`cuMemFree` driver calls and their implicit syncs. NVIDIA's RAPIDS gpu-bdb (SF1000) reported **2-5x** end-to-end switching `cudaMalloc` to `cudaMallocAsync`, but that is a churny multi-GPU data workload; a diffusion loop with a stable working set benefits mainly from killing per-step alloc latency and sync, not necessarily 2-5x.

**Driver-API calls:** `cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemAllocFromPoolAsync`, `cuDeviceGetDefaultMemPool`, `cuMemPoolCreate`, `cuMemPoolSetAttribute` / `cuMemPoolGetAttribute`, `cuMemPoolTrimTo`. C# implementability: trivial; same `[DllImport]` pattern as the existing `cuMemAlloc`; ManagedCuda mirrors all signatures.

- *Stream-Ordered Memory Allocation (Programming Guide)* — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html
- *Stream-Ordered Allocator devblog Part 1 / Part 2* — https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1 , https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2
- *Driver API CUDA_MALLOC_ASYNC* — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html

## 2. Low-level GPU virtual memory management (VMM) API

**Mechanism.** Decouples virtual-address reservation from physical backing (like OS `mmap`/`mprotect`): `cuMemGetAllocationGranularity` (2 MiB page granularity on current NVIDIA GPUs, incl. the 3060) then `cuMemAddressReserve` (reserve a contiguous VA range, possibly larger than backed) then `cuMemCreate` (physical chunk, returns `CUmemGenericAllocationHandle`) then `cuMemMap` then `cuMemSetAccess`. Teardown: `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`.

**What it saves.** Growable buffers with no realloc/copy (reserve a large VA, back only what you use, grow by mapping more 2 MiB pages onto the already-reserved contiguous VA; pointers stay valid). Defragmentation: map non-contiguous physical pages into a contiguous VA window, stitching scattered free memory into one usable buffer (this is how GMLake defrags large DNN allocations). Cost is the 2 MiB granularity (sub-2-MiB tail waste) plus heavier per-call bookkeeping, so it is for coarse, long-lived, growable arenas, not per-tensor churn.

**C#:** doable; `CUmemGenericAllocationHandle` is a `ulong`, `CUmemAllocationProp`/`CUmemAccessDesc` are small blittable structs. ManagedCuda binds the full VA group as a layout reference.

- *Virtual Memory Management (Programming Guide)* — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
- *Introducing Low-Level GPU VMM* — https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
- *GMLake (VMM defrag paper)* — https://arxiv.org/pdf/2401.08156

## 3. PyTorch-style caching allocator (pattern to copy, not an API)

Two-level design: *segments* (contiguous regions from CUDA or VMM) and *blocks* (sub-regions tracking tensors). Two pools, small and large. Rounding happens twice: request rounds up to a 512 B multiple; pulls from CUDA in 2 MiB-minimum multiples expecting to split. Blocks split to serve smaller requests and coalesce with free neighbors on return. Classic failure: lots of free memory but no single contiguous block large enough, because blocks cannot coalesce across two separate `cudaMalloc` segments.

**`expandable_segments`** uses the §2 VMM API: one big reserved contiguous VA range backed by fixed-size pages (2 MiB small / 20 MiB large) mapped on demand, so blocks always merge with neighbors and the cross-segment barrier disappears (the fragmentation fix). Tradeoff: page-size rounding can waste tail (16 MiB request from a 20 MiB page leaves 4 MiB); track `requested_bytes` vs `allocated_bytes` to detect when rounding overhead exceeds the fragmentation it saves.

**C#:** a design to copy, not an API to bind. The base caching allocator needs only `cuMemAlloc`/`cuMemAllocAsync` plus your own block/segment free-lists with split/coalesce. `expandable_segments` mode needs the §2 VMM calls. `CUDACachingAllocator.cpp` is the canonical reference.

- *A guide to PyTorch's CUDA Caching Allocator (zdevito)* — https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
- *PyTorch DevLog: fragmentation in the caching allocator* — https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/

## 4. Pinned host memory + async copies + copy/compute overlap

**Pinned (page-locked) host memory.** GPU copy engines cannot DMA out of pageable host memory; for a pageable transfer the driver stages through a temporary pinned buffer (an extra CPU-bound copy). Allocating the host buffer page-locked removes that staging copy. NVIDIA: pinned transfers are ">2x" pageable; on PCIe 3.0 x16 roughly ~6 GB/s pageable vs ~12 GB/s pinned. Get it via `cuMemHostAlloc(&p, bytes, flags)` (fresh page-locked, flags `PORTABLE`/`DEVICEMAP`/`WRITECOMBINED`) or `cuMemHostRegister(p, bytes, flags)` (page-lock an existing malloc'd / mmap'd region in place). `cuMemHostGetDevicePointer` for `DEVICEMAP` zero-copy. Free with `cuMemFreeHost` / `cuMemHostUnregister`. Warning: pinned memory is a scarce OS resource; pin only the staging buffers you stream, not everything.

**Async copies.** `cuMemcpyHtoDAsync_v2(dst, srcHost, n, stream)` / `cuMemcpyDtoHAsync_v2`. **Critical pitfall:** async copies overlap *only* if the host buffer is pinned. With pageable memory there is no error, but the driver stages and synchronizes, so the "async" copy behaves like a blocking copy and overlaps with nothing. This is the single biggest trap in the whole effort: perfect stream wiring shows zero overlap purely because the buffer was not pinned.

**Copy/compute overlap.** Ops in different non-default streams may run concurrently; the GPU DMAs stream[i+1]'s chunk while computing stream[i]'s chunk. Three hard requirements: pinned host memory, copy and kernel in different non-default streams (the NULL stream serializes everything), and concurrent-copy-and-execute support (every modern GPU; the 3060 additionally has independent H2D and D2H copy engines so both directions overlap). NVIDIA's canonical C2050 benchmark: sequential 9.98 ms; per-stream copy/kernel/copy 5.74 ms (~1.7x); batched-copies-then-kernels 7.60 ms (D2H does not overlap). Issue per-chunk copy-kernel-copy with 2-3 streams and double-buffered staging; more than ~3-4 streams gives diminishing returns.

**Events for cross-stream deps.** `cuEventCreate` (with `CU_EVENT_DISABLE_TIMING`), `cuEventRecord(evt, stream)`, `cuStreamWaitEvent(stream, evt, flags)` (device-side wait, no CPU spin). Double-buffered weight upload: copy stream records `evt_N` after the H2D for layer N; compute stream calls `cuStreamWaitEvent(compute, evt_N)` before layer N; compute records `evt_done_N` after consuming buffer N; copy stream waits on `evt_done_N` before reusing that staging buffer for layer N+2. Two staging buffers + two events = continuous overlap, no CPU sync.

**Streaming a 24 GB model over PCIe 3.0.** ~24 GB / 12 GB/s pinned ~= **2.0 s** to upload once; pageable ~6 GB/s ~= **4 s** plus staging overhead with no overlap. PCIe is packetized so peak needs large transfers (>= 8 MB); batch many small weight tensors into big contiguous uploads. For a 12 GB card running a 24 GB model this PCIe time is recurring per step, which is exactly why overlap (below) makes streaming viable.

- *How to Optimize Data Transfers in CUDA C/C++* — https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- *How to Overlap Data Transfers in CUDA C/C++* — https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- *Driver API Memory / Stream / Event management* — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html

## 5. VRAM offload / weight streaming / block swap (the highest-value item for 12 GB)

**Whole-component model offload.** Keep one major component (text encoder, UNet/transformer, VAE) on GPU at a time; the busy one stays resident for its full run (no per-step ping-pong). Near-zero added latency. Frees the other components but does not by itself fit a 24 GB transformer on 12 GB. You nearly have this already via the weight cache + UNet-eviction-before-VAE.

**Sequential (leaf-level) offload.** On-load each leaf submodule immediately before its forward, off-load after, every step. Peak can reach <3 GB, but it is "extremely slow" with synchronous transfers and no overlap (thousands of blocking round-trips for a 50-step DiT). This is the naive baseline to *beat* with prefetch.

**Group / block-level offload with stream prefetch (the reference design to port).** Offload a *group* of N transformer blocks at a time on a dedicated transfer stream, pinned CPU staging, `non_blocking` H2D, with a hook that traces execution order on the first forward and wires each group's `next_group` so block i's pre-forward kicks off block i+1's onload; synchronize the transfer stream before the current group's params are used. Diffusers Flux-transformer numbers (5 forward passes): leaf-level + stream prefetch took GPU VRAM to **~0.53 GB at ~baseline speed (9.48 s vs 9.75 s non-offloaded)** when pinned memory is pre-allocated; the penalty (13.72 s) only appears when pinning on-the-fly to save CPU RAM. Key data point: **stream-overlapped per-block offload hides nearly 100% of PCIe transfer latency** when per-block compute exceeds per-block transfer time.

**kijai WanVideoWrapper block swap (production reference for DiT video on small cards).** `blocks_to_swap` (14B model has 40 blocks; swap the last K), `prefetch_blocks` (1 is usually enough to offset the speed loss), `use_non_blocking`, plus `offload_img_emb`/`offload_txt_emb`. Each swapped Wan-14B block is ~400-700 MB; swapping 20 frees ~12 GB. Per-block telemetry: transfer ~0.05 s, compute ~0.12 s, swap-back ~0.04 s, so transfer (~50 ms) is smaller than compute (~120 ms), which is why prefetch hides it almost entirely. Warning: naive `non_blocking=True` without proper stream/pin discipline leaks host+GPU RAM that never recovers; keep the pinned source buffer alive until the consuming event fires and free pinned host memory deterministically.

**Driver-API recipe (slots onto the existing weight cache + `cuMemFreeAsync`):**
1. `cuMemHostAlloc(&pinnedA/B, blockBytes, PORTABLE)` — two pinned host buffers sized to the largest block.
2. One extra `cuStreamCreate` for transfers (compute stays on the existing stream).
3. Per block: `cuMemcpyHtoDAsync(dNext, pinnedNext, n, xferStream)` issued at the start of the current block's compute, then `cuEventRecord(xferDone, xferStream)`.
4. Before launching block i+1's kernels: `cuStreamWaitEvent(computeStream, xferDone, 0)`.
5. Free the consumed device block with `cuMemFreeAsync` once its post-forward event fires; keep pinned buffers alive for the whole generation (do not re-pin per block).

**Quantized weights shrink the streamed bytes.** GGUF/NF4 blocks transfer ~1/4 (Q4) the bytes per block over the bus, multiplying both block-swap headroom and per-block transfer speed, widening the latency-hiding margin. Pairs with block-swap; needs dequant PTX kernels (see [`QUANTIZATION_LOW_PRECISION_INFERENCE.md`](QUANTIZATION_LOW_PRECISION_INFERENCE.md)). DisTorch2's byte/ratio allocation (`cuda:0,4gb;cpu,*`) is a clean config model for resident-vs-streamed once block-swap exists.

- *diffusers memory guide* — https://huggingface.co/docs/diffusers/optimization/memory
- *group_offloading source* — https://github.com/huggingface/diffusers/blob/main/src/diffusers/hooks/group_offloading.py ; *stream PR #10516*, *record_stream PR #11081*, *Flux offload benchmark PR #11106*
- *kijai block swap* — https://deepwiki.com/kijai/ComfyUI-WanVideoWrapper/6.2-block-swapping-and-device-management ; *non-blocking RAM leak* — https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/396
- *ComfyUI core memory management* — https://deepwiki.com/Comfy-Org/ComfyUI/2.6-memory-and-device-management

## 6. Attention / activation memory reduction (VAE tiling is the real high-res win)

With flash/memory-efficient attention already present (it never materializes the N-squared score matrix; O(N) memory), the UNet/DiT attention peak is already solved. That makes **VAE decode the binding activation peak** at high resolution: the decoder upsamples the small latent grid back to full pixel resolution, holding full-res multi-channel activations at once. Concrete: SD-VAE peak activation reaches ~60.41 GB at 4096px; a 512 to 2048 decode hits ~13 GB peak on a 5090.

- **Spatial VAE tiling (image)** — divide the latent into overlapping spatial tiles, decode each with the existing decode kernel, blend overlaps with a linear ramp (`blend_h`/`blend_v`, weight = position/blend_extent). Makes decode peak **constant in output resolution** (the original implementation generates 4k images in 8 GB). The single biggest high-res win, mostly a host tile loop plus a small alpha-blend pass.
- **Temporal+spatial VAE tiling (video) with causal feature cache** — temporal outer loop, spatial inner loop. HunyuanVideo 3D VAE defaults: spatial tile 256 / stride 192 (blend band 64 px), temporal 16 frames / stride 12, with `blend_t` for the temporal overlap. The 3D VAE is causal, so decode in temporal chunks while caching the tail frames for the next chunk to keep the conv sliding window continuous (Wan-VAE does the same to decode unlimited-length video without holding the full clip). Makes peak scale with chunk frames instead of total frames. The only nontrivial part is the tail-frame feature cache, and the engine already has CausalConv3d cache machinery. Failure mode: temporal-seam ghosting at chunk boundaries if the overlap/blend is wrong.
- **VAE slicing (batch)** — decode one image at a time from a batch; helps only when batch > 1, useless for a single high-res image/video.
- **Attention slicing** — the pre-flash fix (compute attention one head/query-slice at a time). Given flash attention is present, this adds ~10% latency for near-zero benefit. Implement only for a non-flash fallback path.

- *diffusers memory guide / tiled VAE PR #1441* — https://github.com/huggingface/diffusers/pull/1441
- *HunyuanVideo VAE source* — https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py
- *Asymmetric VAE (peak-memory numbers)* — https://arxiv.org/pdf/2509.24142

## 7. Multi-stream concurrency + CFG batching

**Streams.** Ops in different non-default streams may run concurrently, but only if the GPU has spare SMs/registers/SMEM after the first kernel is scheduled. A single large denoiser kernel already fills the 3060, so a second concurrent kernel finds no room. Stream **priorities** (`cuStreamCreateWithPriority`, lower number = higher priority) are a non-preemptive scheduling hint with no effect on copies and no denoise-loop benefit.

**Where overlap actually exists for a single request.** The denoise loop is inherently sequential (step N+1 consumes step N's latent), so multi-stream gives **no single-request denoise speedup**. The real wins: (a) next-block weight H2D prefetch overlapped with current-block compute (§4/§5, the high-value one on 12 GB), and (b) concurrent/prefetched text encoders (SDXL's two independent encoders, but a small one-time cost). VAE decode is last, so nothing follows it to overlap for a single request.

**CFG batching.** Standard batched CFG concatenates uncond+cond into one batch=2 forward (`eps = eps_uncond + scale*(eps_cond - eps_uncond)`): same FLOPs as two passes but one launch sequence and better occupancy, at **2x activation memory** held simultaneously. On a 12 GB card that 2x peak is a real constraint, so batch=2 (lower latency) vs two sequential passes (lower peak) is a direct latency-vs-VRAM lever you control.

**Guidance-distilled models (the big CFG win).** Flux.1-dev folds guidance into a learned guidance-embedding input (single forward, no uncond branch); SDXL-Turbo/LCM use guidance_scale ~0. Detecting a guidance-distilled checkpoint and skipping the uncond batch gives **~2x speed and half the activation VRAM**, purely a pipeline control-flow choice with no kernel work. See [`CFG_AND_GUIDANCE.md`](CFG_AND_GUIDANCE.md) and [`STEP_ACCELERATION.md`](STEP_ACCELERATION.md).

**Serving.** Continuous batching is LLM-centric (diffusion is fixed-length, so the ragged-completion problem barely exists); vLLM-Omni's step-wise variant admits requests between steps but cost still scales with batch and is fragmented by resolution. On a 12 GB card, multi-request batching of a large model is VRAM-bound, so practical serving batch size is small. Treat multi-stream as plumbing for weight prefetch + optional small-batch serving, not a single-request speedup.

- *How to Overlap Data Transfers* — https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
- *CUDA Driver API Stream Management* — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html
- *Flux guidance-distilled discussion* — https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions/17 ; *Adapter Guidance Distillation (arXiv:2503.07274)* — https://arxiv.org/pdf/2503.07274
- *vLLM-Omni diffusion continuous batching* — https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/diffusion_continuous_batching/

## 8. Feature caching across steps + KV-cache analog for video AR world models

**Within-request feature caching across denoise steps (training-free, high value).** Adjacent denoise steps produce near-identical deep features, so compute the expensive part every N steps and reuse it between. Detailed in [`STEP_ACCELERATION.md`](STEP_ACCELERATION.md); the memory-side note is that each is just a per-request scalar threshold plus one or two cached GPU tensors held between steps, no new kernels. TeaCache (DiT, reuse full output when a timestep-modulated proxy delta stays under threshold, 1.5-4.4x) and First-Block-Cache (run block 0, skip the rest if its residual delta is small, 1.5-3x) fit the DiT image/video backbones; DeepCache (2.3-4.1x) is UNet-only.

**KV-cache analog for video AR / causal world models (the engine's real KV-cache).** In a block-causal AR video model each new frame/chunk attends to past frames, so caching past-frame K/V makes the next chunk cost O(new x cached) instead of re-encoding history, exactly the LLM decode KV-cache.
- **CausVid** — block-causal mask (bidirectional within a 5-latent-frame chunk, causal across chunks); after each chunk, one forward computes its K/V and appends to the cache; distilled to 4 steps. Cache = per-layer append buffer of K,V shaped [chunk_tokens, heads, head_dim].
- **Self-Forcing** — adds a fixed-size **rolling** KV cache holding only the most recent L frames (evict oldest when full, the LLM sliding-window pattern). Complexity O(TL) vs O(L^2 + TL). 16.1 FPS, 4-step, single H100, stable long rollout.
- **Diffusion Forcing** — clean past frames (noise level 0) have stable reusable K/V; only the frame being denoised has changing K/V across its few steps. This is what makes the rolling cache valid. Oasis (the CI smoke model that fits the 3060) is built on it.

C# implementability (KV cache): the same engineering an LLM KV-cache needs. Per attention layer keep two unmanaged ring buffers (K, V) of shape [window_frames x tokens_per_frame, heads, head_dim] in `NativeMemory.AlignedAlloc`; append after each chunk, evict the oldest frame when full; the attention kernel needs a block-causal mask and reads the concatenated [cached_KV ; current_KV]. Match the window length L the checkpoint was trained with (Matrix-Game 2.0 = 6 frames; Self-Forcing trains a specific L). See [`INTERACTIVE_INFERENCE.md`](INTERACTIVE_INFERENCE.md).

- *DeepCache (arXiv:2312.00858)*, *TeaCache (arXiv:2411.19108)*, *FBCache (ParaAttention)* — https://github.com/chengzeyi/ParaAttention
- *CausVid (arXiv:2412.07772)* , *Self-Forcing (arXiv:2506.08009)* , *Diffusion Forcing (arXiv:2407.01392)*

## 9. CPU-offloaded activations — what exists, and why paging usually loses to fixing the peak

**The materialize→reload mechanism already exists per-tensor.** `GpuTransferHelper.CacheActivation` plants a lazy sync callback (`GpuTransferHelper.cs`) that does stream-sync → `Tensor.EnsureHostBuffer()` → D2H → free-device; reload is simply the next `CopyToDevice` cache miss re-uploading from host data. Touching `DataPointer` fires it, which is why several transformers "host-materialize" a cross-step tensor with a bare `_ = t.DataPointer` so a later `FreeActivations` cannot revert them to stale host memory (`BooguImageTransformer`, `ErnieImageTransformer`, `WanVideoTransformer`). What the engine lacked was a *named, bulk, policy-level* entry point — `IBackend.OffloadActivation` / `OffloadActivations(targetBytes)` — not the transfer path. `FreeActivations` itself does no D2H by design: it reclaims buffers that are provably dead.

**Why bulk paging is not a general lowvram lever here.** Under eager execution intermediates are already freed at their point of last use, so there is no large pool of idle-but-resident activations to page. The activations that *are* long-lived are long-lived because something reads them repeatedly — and the read frequency decides everything:

- **Per-block-per-step tensors: offload always loses.** The host path is *pageable* (`EnsureHostBuffer` is `NativeMemory.AlignedAlloc`, not `cuMemHostAlloc`) and *synchronous* (blocking `cuStreamSynchronize` then a blocking copy, nothing overlapped), so real throughput is ~3-6 GB/s, well under the ~12-13 GB/s a pinned async path reaches on PCIe 3.0 x16. Paging one full-sequence tensor per block per step across a 50-block DiT is hundreds of GB of round trips per step. Making it competitive would need pinned staging rings + async D2H on a separate stream + prefetch (§4), and it would *still* lose to simply not holding the tensor (below).
- **Per-step tensors: offload is defensible as a last resort.** A step-cache residual (`DeviceFeatureCache`) is latent-sized and read at most once per step, so paging costs one round trip per step to free its full size. That is a real latency-vs-VRAM lever, unlike the per-block case.

**Case study — MiniMax-H3 chunked attention (the engine's one measured activation OOM).** `MiniMaxH3Transformer.AttentionChunked` runs two passes: pass 1 projects q/k/v per row-chunk, scatters k/v into full `[1, heads, seq, hd]` buffers, and holds every q chunk for pass 2 — three full-sequence F32 buffers live at once. Paging q would be the per-block-per-step case above. Two structural fixes remove the same bytes at no transfer cost:

- **Split the projection across the passes.** Pass 1 projects k+v only (2/3 of the fused GEMM); pass 2 re-projects q per chunk (1/3). Total FLOPs are unchanged — it is a redistribution, not a recompute — and the pass-1 peak drops from 3x to 2x `seq·inner·4`. The fused qkv weight is packed `[q | k | v]` contiguously, so both slices are borrowed row ranges. Not bit-identical: q's GEMM narrows, so cuBLASLt picks a different algorithm (the same divergence class already documented for chunked-vs-unchunked).
- **Scatter outputs into their destination instead of listing + `Concat`.** Both `AttentionChunked` pass 2 and `MlpChunked` accumulated every output chunk in a list, keeping a full `[seq, hidden]` live alongside the concatenated result. Writing each chunk straight into the destination at its row offset (`ScatterRowsGeneric`, the byte-offset inverse of `SliceRowsGeneric`) removes that copy and is bit-identical. Same argument as the existing `ScatterSeqHeadMajor`, which fixed the identical mistake for k/v.

Together these take the per-block floor from `residual + 3·fullSeqInner + chunkScratch` to `residual + 2·fullSeqInner + chunkScratch`, i.e. slope `107520·seq` → `78848·seq` bytes for this config — a **+36% sequence-length ceiling** at fixed VRAM. Measured on the 4090 (resident DiT 19,988 MB, 22,634 MB free → 2,646 MB for activations): 56 frames @ 768x768 (seq 10,490) needed 2,771 MB and was refused by 125 MB before the change.

**Invariants for any bulk offload path** (learned from the sibling free paths, all enforced in `GpuTransferHelper`):

- **Safe points only, on the owning thread.** `State`'s caches are non-concurrent dictionaries, and nothing distinguishes an idle cached activation from a live kernel argument — which is why `FreeActivations` is documented "called between pipeline stages, never mid-op". In particular **do not hook offload into the allocator's OOM retry**: that fires mid-op by construction. The typed `OutOfVramException` exists so stage-boundary callers can degrade instead.
- **Skip arena-backed pointers** — graph-arena allocations are never freed individually.
- **Invalidate the step graph**, exactly as `FreeActivations` does: a captured graph bakes activation pointers, and a reload returns a *new* device pointer even though host data is intact. Callers must also skip offload while a step graph is live (the video pipelines already suppress their per-step `FreeActivations` for this reason).
- **Clear the pin.** `PinActivation` exists because the host copy was never materialized; after a real D2H the host *is* authoritative, so a pinned tensor becomes offloadable.
- **Block weight auto-promotion for anything offloaded**, or the lever silently undoes itself: the reload path re-uploads unchanged host data every read, and `TryAutoPromote` promotes a >=1 MB tensor into the *weight* cache on its second upload — quietly making it device-resident again.

## Ranked impact vs effort (12 GB RTX 3060)

| # | Technique | Impact | Effort | Notes |
|---|---|---|---|---|
| 1 | Block-swap with stream prefetch + pinned double-buffer | Very high | Medium | The only thing that runs unquantized Flux 12B / video on 12 GB while hiding ~all PCIe latency. Port the diffusers group-offload / kijai design. Do first. |
| 2 | Pinned staging buffers (`cuMemHostAlloc`) + async copies | High | Low | ~2x H2D bandwidth and prerequisite for all overlap. Watch the pageable to silent-sync pitfall. |
| 3 | Spatial VAE tiling (image) + temporal/spatial+causal cache (video) | Very high (high-res/video) | Low-medium | Targets the true activation peak; host tile loops over existing kernels + linear blend + tail-frame cache. |
| 4 | Stream-ordered pool (`cuMemAllocAsync` + capped `RELEASE_THRESHOLD`) | High | Low | Near drop-in over current calls; kills per-step alloc latency/sync. Cap retained cache to avoid OOM at 12 GB. |
| 5 | Within-request feature caching (TeaCache / FBCache) | High | Low | ~1 day; scalar threshold + cached residual tensor; training-free. (Detailed in STEP_ACCELERATION.) |
| 6 | Rolling/block-causal KV cache for AR world models | Very high (interactive) | Medium | Mandatory for Matrix-Game/Oasis/CausVid; ring buffers + masked attention. |
| 7 | Whole-component model offload as baseline policy | Medium | Low | Nearly have it; cheap, zero-latency, ship alongside #1. |
| 8 | Guidance-distilled path (skip uncond) | High | Low | ~2x speed + half activation VRAM; pipeline control flow, no kernels. |
| 9 | Custom size-class caching suballocator over the pool | High | Medium | The "real pool" the engine lacks; predictable steady-state VRAM for repetitive shapes. |
| 10 | VMM-backed expandable segments (defrag / growable) | Medium-high | High | Only when fragmentation OOMs appear (free-but-not-contiguous) near 12 GB. |
| 11 | Batched-vs-two-pass CFG toggle | Medium | Low | Latency-vs-peak-VRAM lever. |
| 12 | Multi-stream concurrency for kernels / serving | Low (single req) | Medium-high | No denoise-loop speedup (GPU already saturated); serving-only, VRAM-bound on 3060. |
| 13 | Attention slicing | ~Zero w/ flash attn | Low | Only for a non-flash fallback path; otherwise skip. |
| 14 | CPU-offloaded activations (bulk paging) | Low, situational | Low (mechanism exists) | §9. Per-tensor D2H/reload already exists; only a policy entry point was missing. Loses to removing the peak for per-block tensors; defensible only for per-step ones (step-cache residual). |

## Top recommendations for a 12 GB RTX 3060 running 12B+ models

1. **Block-swap with stream prefetch** (pinned double-buffer + transfer stream + `cuEventRecord`/`cuStreamWaitEvent` over the existing weight cache). The enabler for large unquantized models; hides ~all PCIe latency when per-block compute > per-block transfer.
2. **Pinned staging + async copies** first as the foundation for #1; ~2x bandwidth and the prerequisite for any overlap.
3. **Stream-ordered pool** with a capped release threshold to remove per-step alloc/sync, then a **size-class caching suballocator** on top once instrumented.
4. **VAE tiling** (spatial for image, temporal+spatial+causal-cache for video) to remove the binding high-res/video activation peak.
5. **Guidance-distilled path + within-request feature caching (TeaCache/FBCache)** for cheap, training-free per-step compute cuts.
6. **Rolling KV cache** for the interactive AR world models (match each checkpoint's trained window).
7. Reach for **VMM expandable segments** only when fragmentation data justifies it; treat multi-stream kernel concurrency and attention slicing as low-priority for this single-GPU, flash-attention setup.

Caveats relayed from research: the 2-5x stream-allocator figure is a multi-GPU data-churn workload (upper bound vs a stable diffusion working set); DisTorch2's "~10% faster" is an n=1 author claim with no rigorous table; block-swap latency is only fully hidden when per-block compute exceeds per-block transfer (fails for tiny/over-quantized blocks or a slow bus or non-pinned memory).
