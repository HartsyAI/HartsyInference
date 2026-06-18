# Deep Kernel / GPU Optimization — Research Notes

Optimization techniques *beyond* the kernels HartsyInference already documents in [`CUDA_AND_PTX.md`](CUDA_AND_PTX.md) (Conv2D im2col+GEMM, warp-shuffle GroupNorm/LayerNorm, materialize-S SDPA, elementwise, `f16x2` packed arithmetic) and beyond the roadmap items already named in [`CUDA_PERFORMANCE_PLAN.md`](CUDA_PERFORMANCE_PLAN.md) (FA2, fusion, CUDA Graphs). Everything here maps to hand-written PTX (loaded via `cuModuleLoadData`, JIT by `ptxas`) or to Driver-API / cuBLASLt P/Invoke. No CUDA C++. Primary target SM 8.0/8.6/8.9 (dev box RTX 3060 = SM 8.6), Hopper SM 9.0 as a cloud upside.

This is a survey companion to the master plan; it does not replace [`FLASH_ATTENTION.md`](FLASH_ATTENTION.md) (algorithm reference) or [`CONV2D_CUDA.md`](CONV2D_CUDA.md).

---

## 1. Tensor-core programming from raw PTX

### `mma.sync` vs `wmma` (use `mma.sync`)

Both are warp-level (32 threads cooperate) and lower to the same SASS `HMMA`, so `mma.sync` buys *control*, not new throughput. `wmma` duplicates each A/B element across two threads and hides the fragment layout; `mma.sync.aligned` packs fragments with no duplication, exposes every shape/dtype, and pairs with `ldmatrix`. A hand-written `mma.sync` HGEMM reaches >= 95% of cuBLAS, and the Spatters SM89 kernel hit 100% of cuBLAS / 93% of 4090 peak.

Workhorse instruction (broadest SM80+ reach, largest FP16 shape on Ada):
```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3};
```
Documented shapes: m8n8k4 (Volta legacy, avoid), m16n8k8 (Turing+), m16n8k16 (Ampere+). `.row.col` means A row-major, B col-major across registers.

- *Nvidia Tensor Core — Getting Started with MMA PTX Programming (Bruce-Lee-LY)* — https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d
- *Implementing a fast Tensor Core matmul on Ada (spatters.ca)* — https://www.spatters.ca/mma-matmul
- *PTX ISA §9.7.15.5 (mma)* — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

### `ldmatrix` — feed fragments from shared memory

Warp-collective load of 8x8 `.b16` tiles directly into the per-thread MMA layout, eliminating manual cross-lane shuffles. One `ldmatrix.x4.m8n8` exactly fills the 4-register A fragment of `m16n8k16`. The `.trans` variant transposes in-flight (how you feed row-major SMEM into the col-major B operand).
```
ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];
ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {...}, [addr];   // for V / col-major B
```
- *CuTe ldmatrix (Lei Mao)* — https://leimao.github.io/blog/CuTe-ldmatrix/
- *cuda_hgemm/src/common/ptx.h (Bruce-Lee-LY)* — https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/common/ptx.h

### Register fragment layout (m16n8k16, f32 accumulate) — the load-bearing detail

Per lane, with `groupID = laneid >> 2` and `tg = laneid % 4`:
- A elem i in 0..7: `row = groupID + 8*((i/2)%2)`, `col = 2*tg + (i%2) + 8*(i/4)`
- B elem i in 0..3: `row = tg*2 + (i%2) + 8*(i/2)`, `col = laneid >> 2`
- C/D elem i in 0..3: `row = groupID + 8*(i/2)`, `col = 2*tg + (i%2)`

A/B are packed two-fp16-per-32-bit register; C/D are one-fp32-per-register. This index math is exactly why `ldmatrix` exists.
- *A gentle introduction to GEMM using MMA tensor cores (Aman)* — https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/

### Data-type / SM capability tiers

| Type | Shape | Accum | First SM |
|---|---|---|---|
| f16 | m16n8k16 | f16/f32 | sm_80 (m16n8k8 sm_75) |
| bf16 | m16n8k16 | f32 | sm_80 |
| tf32 | m16n8k8 | f32 | sm_80 |
| int8 | m16n8k32 | s32 | sm_75/80 |
| fp8 e4m3/e5m2 | m16n8k32 | f16/f32 | sm_89 (warp `mma.sync`); sm_90 adds `wgmma` |
| fp4 e2m1 (+fp6) | m16n8k32/64 | f32 | sm_100 / sm_120 |

A100 throughput vs FP32 CUDA cores: TF32 ~8x, FP16/BF16 ~16x. Ada SM89 has the warp-level FP8 path. Consumer Blackwell (sm_120) has FP4 via *synchronous* `mma.sync.m16n8k32`, no TMEM/tcgen05.
- *Benchmarking NVIDIA Tensor Core MMA Peak Performances (Lei Mao)* — https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/
- *NVIDIA A100 Ampere Whitepaper* — https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

### `cp.async` pipelining (SM80+)

Copies global to shared without going through registers, asynchronously; the warp keeps doing `mma` on the current buffer while the next fills.
```
cp.async.cg.shared.global.L2::128B [smem],[gmem],16;   // .cg bypasses L1, 16B copies
cp.async.commit_group;  cp.async.wait_group N;
```
Multistage (N=3 optimal at 4096-cubed on a 4090): the prelude issues N-1 K-tile copies; the steady state is `wait_group N-2` then `ldmatrix` current, then issue next `cp.async`+commit, then `mma.sync`. This is the same pipeline FA2 uses.
- *CUTLASS Tutorial: GEMM Pipelining (Colfax)* — https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/

### Open MMA-in-PTX references to copy

- Bruce-Lee-LY/cuda_hgemm (full ldmatrix+mma PTX) — https://github.com/Bruce-Lee-LY/cuda_hgemm
- Spatters SM89 walkthrough (naive to swizzle to cp.async, all PTX strings + numbers) — https://www.spatters.ca/mma-matmul
- Hopper/Blackwell MMA layouts — https://vjkrish.com/2026/01/19/Mma_Layouts.html

CuTe concepts worth *borrowing* (not as a dependency): represent each PTX mma variant as an **MMA Atom** descriptor (shape + register counts + fragment-mapping formulas); model `ldmatrix`/`cp.async`/TMA as **Copy Atoms**; compose atom-to-tile via a **TiledMMA** layout instead of copy-pasted loops; apply an XOR **swizzle** on SMEM writes. — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html

---

## 2. Flash attention evolution

### FA2 online-softmax recurrence (what you implement on Ampere)

Per query block i, looping KV block j:
```
S = Q_i · K_j^T
m_new = max(m_old, rowmax(S))
P~    = exp(S - m_new)
l     = exp(m_old - m_new)*l_old + rowsum(P~)
O     = diag(exp(m_old - m_new))*O_old + P~ · V_j
```
Normalize **once** after the loop: `O = diag(l)^-1 · O`. Deferring the divide and storing only `L = m + log(l)` removes non-matmul FLOPs (A100: matmul 312 vs non-matmul 19.5 TFLOP/s, ~16x gap). FA2's central change is **split-Q**: split Q across the 4 warps, keep K,V shared, so warps need no inter-warp reduction (25-40% to 50-73% of peak). Grid = batch x heads x query-blocks.
- *FlashAttention-2 (Tri Dao, arXiv:2307.08691)* — https://arxiv.org/abs/2307.08691

### FA3 needs Hopper hardware Ampere lacks

FA3 = (1) warp-specialized producer/consumer with **TMA** loads, (2) ping-pong **WGMMA**/softmax overlap (H100 matmul 989 vs exp 3.9 TFLOP/s), (3) FP8 with block-quant plus incoherent (Hadamard) processing. Hard dependencies absent on Ampere/Ada: **TMA, `wgmma.mma_async`, `setmaxnreg` register reallocation, Hopper async mbarriers**. Result: 1.5-2.0x FA2, up to 740 TFLOP/s. Keep FA3 as a *separate* SM90 path; do not back-port.
- *FlashAttention-3 (arXiv:2407.08608)* — https://arxiv.org/abs/2407.08608

### Best practical FA2 tiling for Ampere SM86 (3060)

- head_dim 128: `BLOCK_Q=128, BLOCK_KV=64`, 4 warps, each warp owns 32 persistent Q rows in registers, ~40 KB SMEM (K,V double-buffered). head_dim 64: 64x64 tile.
- Keep tile + double-buffer under ~48-64 KB (SM86 SMEM caps at 100 KB opt-in) to keep >= 2 blocks/SM.
- `mma.sync.m16n8k16.f32.f16.f16.f32` for **both** QK^T and PV; `ldmatrix` (`.trans` for V); `cp.async.cg` 2-3-stage pipeline.
- Softmax **in registers** (never round-trip full S through SMEM): rowmax/rowsum via `__shfl_xor_sync` butterfly over the 4 threads sharing an MMA output row; compute `exp(S - m_new)` before writing P. XOR-swizzle SMEM.
- Reachable ceiling: 67-94% of speed-of-light without WGMMA/TMA.
- Closest hand-PTX reference: **Tugbars/Flash-Attention-PTX-CUDA** — https://github.com/Tugbars/Flash-Attention-PTX-CUDA ; tiling/swizzle numbers: gau-nernst fa-5090 — https://gau-nernst.github.io/fa-5090/ ; fragment/PTX detail: Sonny lubits.ch Part 2 — https://lubits.ch/flash/Part-2

### Diffusion simplifies the kernel (drop the mask)

Diffusion attention is full **bidirectional**, fixed seqlen, no KV cache. Relative to a causal LLM kernel you can drop the entire causal mask + diagonal special-case, the triangular load-imbalance scheduling, and all variable-length (`cu_seqlens`) plumbing. You keep online softmax (KV does not fit SMEM for big token counts) and must handle cross-attention with `seqlen_kv != seqlen_q` (just a different inner-loop bound). **One non-causal FA2 kernel parameterized by `(seqlen_q, seqlen_kv, head_dim in {64,128})` covers diffusion self- and cross-attention** and is materially simpler to hand-write than a causal kernel.
- *Attention in Diffusion Model: A Survey* — https://arxiv.org/html/2504.03738v1

### Flash-Decoding / FlashDecoding++ (NOT for diffusion)

Adds a KV-split grid dimension for query-len-1 long-KV decode (batch 1: 50x faster attention). Relevant only to your *autoregressive audio/token* models (ACE-Step/MusicGen decode), not the diffusion denoiser, which already saturates the GPU via query-block parallelism.
- *Flash-Decoding (PyTorch blog)* — https://pytorch.org/blog/flash-decoding/ ; *FlashDecoding++ (arXiv:2311.01282)* — https://arxiv.org/pdf/2311.01282

### Memory-efficient attention + fused QKV

Rabe & Staats is the online-softmax precursor (O(sqrt n) memory). **Fused QKV**: one GEMM against concatenated `[3*d, d]` weights for self-attention (Q,K,V share input); for cross-attention fuse only KV and project Q separately. This lives in your GEMM/projection layer, orthogonal to the flash kernel.
- *Self-attention Does Not Need O(n^2) Memory (arXiv:2112.05682)* — https://arxiv.org/abs/2112.05682

---

## 3. Hopper / Blackwell features (cloud upside)

**TMA** — `cp.async.bulk.tensor.{1d..5d}` single-thread-issued multi-dim async copy; descriptor `CUtensorMap` built **host-side via `cuTensorMapEncodeTiled`** and passed **by value as `__grid_constant__`** (PyTorch measured pointer-passing at ~4 ms launch vs by-value ~10 us, ~3330x). Completion via `mbarrier.arrive.expect_tx` / `mbarrier.try_wait.parity`. FP8 GEMM: 910 GB/s to 1.45 TB/s. SM90a.
- *CUTLASS Tutorial: Mastering the TMA (Colfax)* — https://research.colfax-intl.com/tutorial-hopper-tma/ ; *Deep Dive on the Hopper TMA Unit (PyTorch)* — https://pytorch.org/blog/hopper-tma-unit/

**wgmma** — `wgmma.mma_async` async warpgroup (128-thread) MMA, M fixed 64, N 8..256, K=16 (16-bit)/8 (tf32)/32 (fp8). B always SMEM, accumulator always registers, SMEM operands via 64-bit descriptor (start/LBO/SBO/swizzle). Ordering: `wgmma.fence` then `mma_async` then `commit_group` then `wait_group N`. SM90a only.
- *CUTLASS Tutorial: WGMMA on Hopper (Colfax)* — https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/

**Clusters / DSMEM** — co-scheduled CTAs share a unified SMEM address space; `mapa.shared::cluster` to address remote SMEM, `cluster.sync` barrier, TMA-multicast feeds many CTAs from one load. Launch via **`cuLaunchKernelEx` + `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`**. SM90.

**Persistent + warp specialization** — 1 CTA/SM (Stream-K), producer warps do only TMA, consumer warps only wgmma+softmax; a ping-pong scheduler hides softmax behind tensor-core work. **`setmaxnreg.inc/.dec`** reallocates registers (producers few, consumers many), mandatory to avoid spills.
- *Deep Dive on CUTLASS Ping-Pong GEMM (PyTorch)* — https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/

**tcgen05 (Blackwell SM100, forward-looking, NOT H100/H200)** — single-thread-issued MMA reading from **TMEM** (256 KB/SM, explicitly `tcgen05.alloc`/`dealloc`/`ld`/`st`), 2-SM cooperative MMA, MX microscaling (mxfp4/nvf4), 2-4x wgmma. Consumer Blackwell sm_120 lacks this surface.
- *tcgen05 for dummies* — https://gau-nernst.github.io/tcgen05/

**To exploit H100/H200 concretely:** JIT to **`sm_90a`**; build `CUtensorMap` host-side; pass by value as grid-constant; opt into ~228 KB SMEM via `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`; emit TMA + wgmma + mbarrier + `setmaxnreg` PTX; adopt persistent warp-specialized producer/consumer. Plan the MMA layer swappable: `mma.sync` (SM80) to `wgmma` (SM90a) to `tcgen05.mma` (SM100a), since accumulator location differs (regs to regs-async to TMEM).
- *SM90 Hopper Features (CUTLASS DeepWiki)* — https://deepwiki.com/NVIDIA/cutlass/7.1-sm90-hopper-architecture

---

## 4. CUDA graphs via the Driver API

All names below are the `cu*` Driver API; the dev blogs show the runtime `cuda*` equivalents. The 4000+ kernels/step with identical topology is the textbook best case.

- **Capture**: `cuStreamBeginCapture` then your normal `cuLaunchKernel` calls then `cuStreamEndCapture(&graph)`. Or explicit: `cuGraphCreate`, `cuGraphAddKernelNode`/`AddMemcpyNode`/`AddNode`, `cuGraphAddDependencies`.
- **Instantiate/launch**: `cuGraphInstantiate(&exec, graph, flags)` (expensive, once) then `cuGraphUpload` (optional) then `cuGraphLaunch(exec, stream)` (one CPU call for the whole graph).
- **Launch-overhead win**: V100, 2.9 us kernel: sync launch 9.6 us, async 3.8 us, **graph 3.4 us (0.5 us overhead)**. Across 4000 launches/step this removes ~ms of CPU overhead per step plus GPU timeline bubbles.
- **Hot-path param update (never re-instantiate per step)**: store every `CUgraphNode` at build; patch with `cuGraphExecKernelNodeSetParams` (updates kernel args + grid/block at minimal cost). Whole-graph diff via `cuGraphExecUpdate` (topology must be identical, else it fails and you re-instantiate). Update-based reached 1.63x vs recapture 1.22x. Keep pointers/dims stable so most nodes need zero updates.
- **Conditional nodes** (data-dependent control flow): `cuGraphConditionalHandleCreate` before the node, add via `cuGraphAddNode` + `CU_GRAPH_NODE_TYPE_CONDITIONAL`, set value device-side via `cudaGraphSetConditional`. IF/WHILE is CUDA 12.4+; ELSE/SWITCH is 12.8+.
- **Dynamic-shape pitfalls**: node params + topology are frozen at instantiate; different shapes that change node/edge count is a topology change and forces re-instantiate. **Allocate all device buffers OUTSIDE capture** (allocs inside capture become fixed-address graph-mem nodes). Engines **bucket shapes**: one cached `CUgraphExec` per resolution/batch/seqlen bucket; same-topology scalar changes use `cuGraphExecKernelNodeSetParams`.
- **`cuLaunchKernelEx` + programmatic dependent launch (Hopper 9.0+ only)**: the secondary kernel sets `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`; the primary calls `cudaTriggerProgrammaticLaunchCompletion()`, the secondary `cudaGridDependencySynchronize()`, overlapping the secondary's prologue with the primary's tail. **No-op on the 3060** (graphs + node-update are the Ampere wins).
- *Getting Started with CUDA Graphs* — https://developer.nvidia.com/blog/cuda-graphs/ ; *Constructing CUDA Graphs with Dynamic Parameters* — https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/ ; *Dynamic Control Flow with Conditional Nodes* — https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/ ; *Graph Management (Driver API ref)* — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html

---

## 5. Megakernel / fusion-at-scale + cuBLASLt epilogue

### cuBLASLt epilogue fusion (do this FIRST, lowest effort, no new PTX)

Set `CUBLASLT_MATMUL_DESC_EPILOGUE` to fuse bias+activation into the GEMM epilogue, removing a separate elementwise kernel plus an activation-sized HBM round-trip per GEMM. Inference-relevant values: `BIAS(4)`, `RELU_BIAS(6)`, `GELU(32)`, `GELU_BIAS(36)`. P/Invoke flow: `cublasLtMatmulDescCreate` then set TRANSA/TRANSB then set EPILOGUE then set `CUBLASLT_MATMUL_DESC_BIAS_POINTER` then layouts then `cublasLtMatmulPreferenceCreate` (`MAX_WORKSPACE_BYTES`) then `cublasLtMatmulAlgoGetHeuristic` then `cublasLtMatmul`. CUDA 12.0+; 256-byte A/B/C/D alignment, aligned bias pointer (else fall back to DEFAULT + separate add). cuBLASLt is a C API, P/Invoke directly. This engine already binds cuBLASLt for the FP8 path ([`Fp8GemmExecutor.cs`](../../src/HartsyInference.Cuda/Fp8GemmExecutor.cs)).
- *New cuBLAS 12.0 Features (NVIDIA)* — https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/ ; *cuBLASLt Epilogue enum* — https://docs.nvidia.com/cuda/nvmath-python/0.1.0/bindings/generated/nvmath.bindings.cublasLt.Epilogue.html ; *llm.c bias+GELU fusion (#54)* — https://github.com/karpathy/llm.c/issues/54

### Fusions to hand-roll next

Residual/scale/SiLU/GeGLU gating into the GEMM/conv epilogue (one read, one store); norm fused with the adjacent op (RMSNorm+QKV+RoPE, RMSNorm+up/gate+SiLU); and the **fused AdaLN/modulation kernel for DiT** (`(1+scale) (x) LN(x) + shift` plus gated residual in one pass), the highest-value diffusion-specific fusion, removing redundant HBM round-trips of the modulation tensors.

### Megakernel (Phase 2, big batch-1 latency wins)

Graphs remove CPU launch overhead but **preserve kernel boundaries**: each kernel still materializes outputs to HBM and incurs the implicit device-wide sync. A megakernel fuses the whole forward into one persistent kernel: data stays on-chip, sync drops to per-data-unit counters.
- **Mirage/MPK** (CMU 2025): compiler to SM-level task graph + in-kernel scheduler, no global barrier (event counters). 1.0-1.7x end-to-end; Qwen3-8B A100 14.5 to 12.5 ms/token. — https://arxiv.org/pdf/2512.22219
- **Hazy Research Llama-1B megakernel** (the design to copy for hand-PTX): an on-GPU **interpreter**, 7 fused instruction types, persistent kernel, warp specialization, TMA async I/O, explicit 16 KiB SMEM paging, **global-memory counter sync** (not a grid barrier). H100 <1 ms/forward, 78% bandwidth, ~2.5x vs vLLM. — https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

Hard parts: no free global barrier (use `cuLaunchCooperativeKernel` + `grid.sync()`, or a counter spin-wait with `.acquire`/`.release`/`membar`); cooperative grid capped by `cuOccupancyMaxActiveBlocksPerMultiprocessor`; register/SMEM budget; you own the scheduler. Wins only in latency-bound small-batch decode where HBM round-trips + barriers dominate.

---

## 6. Occupancy / latency-hiding micro-opts

- **Register/thread tiling**: each thread computes a `TM x TN` microtile from register-resident strips (outer product), turning `TM+TN` SMEM loads into `TM*TN` FMAs. siboehm SGEMM: 2D blocktile 8x8 to 68.7% cuBLAS, +vectorized loads to 78.4%, +autotune to 84.8%, +warptiling to 93.7%. — https://siboehm.com/articles/22/CUDA-MMM
- **ILP > occupancy (Volkov)**: a register-blocked GEMM runs *faster at lower occupancy*; supply Little's-Law parallelism via ILP (deep register tiles + multistage buffering), not just more warps. — https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf
- **128-bit vectorized access**: `ld.global.v4.f32 {%0,%1,%2,%3},[%4];` (or `uint4` = 8 halfs); one 512 B/warp transaction, 4x fewer load instructions, needs 16-byte alignment. PTX 8.8 adds `.v8` stores.
- **L2 persistence (SM80+, Driver-only, no PTX)**: `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, bytes)` + `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` (base/num_bytes/hitRatio/hitProp=Persisting) + `cuCtxResetPersistingL2Cache`. Anti-thrash: `num_bytes*hitRatio <= setAside`. ~20% win when hot data fits (3090: 3.071 to 2.443 ms). Ideal for pinning weights / text-encoder embeddings re-read across denoise steps. — https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/
- **SMEM swizzle (mandatory for any MMA kernel)**: 32 banks x 4 B; XOR-swizzle (`swz = col ^ row` at 16-byte granularity) spreads `ldmatrix` 8x16B accesses across all banks for conflict-free access, no wasted SMEM (unlike padding). — https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/ , https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html
- **PTX tuning directives**: `__launch_bounds__` becomes `.maxntid` + `.minnctapersm` (caps registers to dial occupancy; overrides `maxrregcount`); `.reqntid` for exact block size. `__grid_constant__` (`.param` by-value) for large structs, **required for Hopper TMA descriptors**. — https://docs.nvidia.com/cuda/parallel-thread-execution/

---

## 7. Convolution algorithms

- **Implicit (precomputed) GEMM** forms GEMM tiles on-the-fly in SMEM by gathering from NHWC with stride/dilation/pad, never materializing im2col (saves the replication-factor memory + bandwidth your current im2col pays). Reuses your existing tiled/tensor-core GEMM mainloop with a different address generator. The safe general default; cuDNN's most-used path. — https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/implicit_gemm_convolution.md
- **Winograd F(2x2,3x3)** — `Y = A^T[(GgG^T) (x) (B^T d B)]A`; 16 vs 36 multiplies = **2.25x** fewer. Wins for 3x3 stride-1 small-batch (UNet / diffusion decoder layers). Precompute `GgG^T` at load; needs fp32 accumulate; larger tiles (F(6x6,3x3)+) grow numerical error (often too inaccurate for fp16); does not generalize to stride>1/dilation. — https://arxiv.org/abs/1509.09308
- **FFT conv** wins only for large kernels / large spatial; zero-pads the filter to map size (wasteful for 3x3), large workspace. Skip unless 7x7+.
- **Decision guide**: 3x3 s1 small-batch to Winograd (fp32 accum); 1x1/strided/dilated/odd to implicit GEMM; large kernels to FFT; first layer C=3 pad C to 4 (TF32)/8 (FP16). For tensor cores make C,K divisible by 8 (FP16)/4 (TF32), ideally 64. — https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
- **cuDNN v9 graph API** fuses conv+bias+act(+BN) via a runtime-compiled fusion engine, but it is a closed native lib (violates the "pure C#, no native shared libs" pillar). Recommendation: hand-roll implicit-GEMM + Winograd in PTX; use cuDNN only as an optional parity oracle, not a runtime dependency.

---

## Ranked table — impact vs PTX-implementation effort (Ampere-first)

| # | Technique | Impact | PTX/Driver effort | Min SM |
|---|---|---|---|---|
| 1 | cuBLASLt epilogue fusion (bias+GELU/ReLU) | High | Very low (descriptor P/Invoke) | 80 |
| 2 | CUDA Graphs + node-param update (4000 to 1 launch) | High | Low (Driver `cuGraph*`) | 80 (all) |
| 3 | L2 persistence for re-read weights/embeddings | Medium | Very low (Driver, no PTX) | 80 |
| 4 | `mma.sync` + `ldmatrix` + `cp.async` GEMM (replace im2col core) | Very high | Medium-high | 80 |
| 5 | SMEM XOR-swizzle (enables 4) | High (enables TC) | Low (address math) | 80 |
| 6 | Non-causal FA2 (mma+cp.async+register softmax) for diffusion | Very high | High | 80 |
| 7 | 128-bit `ld.global.v4` on hot paths | Medium | Very low | 80 |
| 8 | Register/thread tiling + ILP>occupancy + `__launch_bounds__` | High | Medium | 80 |
| 9 | Implicit-GEMM conv (drop im2col materialization) | High | Medium | 80 |
| 10 | Winograd F(2x2,3x3) for 3x3 s1 layers | Medium-high | High (separate kernel) | 80 |
| 11 | Hand-rolled epilogue fusion (residual/scale/gate, fused AdaLN) | Medium-high | Medium | 80 |
| 12 | Megakernel (interpreter, counter-sync, persistent) | High (batch-1 latency) | Very high | 80 |
| 13 | FP8 `mma.sync` (m16n8k32) tier | Medium | Medium | 89 |
| — | **Hopper:** TMA + wgmma + clusters + warp-spec + FA3 | Very high (cloud) | Very high | 90a |

## Top recommendations for an Ampere-first PTX engine

1. **cuBLASLt epilogue fusion now** — bias+GELU/ReLU into every Linear/QKV/O GEMM. Near-zero effort, immediate launch + HBM-round-trip savings.
2. **CUDA Graphs + `cuGraphExecKernelNodeSetParams`** — collapse 4000+ launches/step into one `cuGraphLaunch`; bucket by shape; allocate outside capture; patch only timestep/sigma/CFG nodes.
3. **L2 persistence** — pin re-read weights / text-encoder embeddings across denoise steps; ~20% where hot data fits. Pure Driver call.
4. **Tensor-core GEMM core** = `mma.sync.m16n8k16.f32.f16.f16.f32` + `ldmatrix` (`.trans`) + 3-stage `cp.async.cg` + **XOR-swizzled SMEM** + register tiling. This replaces the im2col+GEMM heart and is the dependency for everything fast.
5. **One non-causal FA2 kernel** parameterized by `(seqlen_q, seqlen_kv, head_dim in {64,128})`, split-Q, register-resident online softmax with `__shfl_xor_sync`. Drop all mask/variable-length machinery, since diffusion allows it. Copy Tugbars (hand-PTX) + gau-nernst (tiling numbers).
6. **Implicit-GEMM conv** to kill the im2col materialization; add **Winograd F(2x2,3x3)** for 3x3 stride-1 decoder layers (fp32 accumulate).
7. Apply **128-bit vectorized loads** and `__launch_bounds__` (`.maxntid`/`.minnctapersm`) tuning across all hot kernels; prefer ILP over raw occupancy.
8. Megakernel (Hazy interpreter model) only later, for batch-1 latency-bound sampling.

## Hopper cloud upside list (SM90a)

- JIT to **`sm_90a`**; opt into ~228 KB SMEM (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`).
- **TMA** (`cp.async.bulk.tensor`) with a host-built `CUtensorMap` (`cuTensorMapEncodeTiled`) passed **by value as `__grid_constant__`**, replacing `cp.async`, freeing threads, ~3330x lower descriptor-launch overhead vs pointer-passing.
- **wgmma** async warpgroup MMA (M=64, accumulate in registers, SMEM descriptors), replacing `mma.sync` for ~2-3x attention.
- **Persistent warp-specialized producer/consumer** + `setmaxnreg` register reallocation + ping-pong scheduler give **FA3** (1.5-2x FA2, up to 740 TFLOP/s; FP8 ~1.2 PFLOP/s).
- **Thread-block clusters / DSMEM** (`cuLaunchKernelEx` + cluster-dim attribute, `mapa`, `cluster.sync`, TMA-multicast) for cluster-scope fusion/reductions.
- **Programmatic dependent launch** to overlap consecutive small kernels.
- Keep the MMA layer swappable for a future Blackwell `tcgen05.mma`/TMEM path (sm_100a). Note H100/H200 are Hopper, so tcgen05 does *not* apply to them.

Consistency note: programmatic dependent launch, clusters, TMA, wgmma, and FA3 are all **no-ops on the 3060 (SM86)**. They are strictly the cloud-Hopper upside; the Ampere wins are items 1-11 above.
