# Multi-GPU Parallelism & Distributed Serving — Research Notes

> Status: Complete | Last Updated: 2026-06-28 | Needed Before: `HartsyInference.LLM` multi-GPU sharding (run Kimi-K2 1T / DeepSeek-V3 671B / large MoE across 2-N GPUs and across nodes)

## Summary

To run a model that does not fit in one GPU's VRAM (Kimi-K2 1.04T, DeepSeek-V3 671B, Llama-405B, Mixtral, large MoE), the parameters, KV cache, and activations must be **partitioned across multiple GPUs and often multiple nodes**. There is no single "multi-GPU" switch; production systems compose up to five orthogonal partitioning dimensions: **tensor parallelism (TP)**, **pipeline/layer parallelism (PP)**, **expert parallelism (EP, for MoE)**, **data parallelism (DP, including DeepSeek's DP-attention)**, and **sequence/context parallelism (SP/CP, for long context)**. Each dimension has a precise place in the transformer forward pass where it inserts a specific collective communication primitive (all-reduce, all-gather, reduce-scatter, all-to-all, or point-to-point send/recv), and each trades latency against throughput against per-GPU memory differently. The datacenters serving the largest models (DeepSeek, Moonshot/Kimi, Meta) additionally **disaggregate prefill from decode** onto separate GPU pools that exchange KV cache over RDMA, and overlay **expert-parallel load balancing** to keep MoE experts evenly loaded.

The single most important decision for HartsyInference: **use NCCL as the collective backend, do not hand-roll all-reduce.** NCCL is a native shared library, but it is **the exact same category as cuBLAS / cuBLASLt, which the engine already P/Invokes** ([`CublasApi.cs`](../../src/HartsyInference.Cuda/CublasApi.cs), [`CublasLtApi.cs`](../../src/HartsyInference.Cuda/CublasLtApi.cs)): a well-maintained, permissively-licensed (BSD-3-Clause) NVIDIA library that would cost enormous effort to reimplement. Per the engine's library policy (no-native-libs is the default, **but well-maintained open-source libs that save large effort are an explicit exception — do not reinvent the wheel**), NCCL is in. It handles every collective we need (all-reduce, all-gather, reduce-scatter, send/recv for all-to-all) across NVLink, PCIe P2P, shared host memory, InfiniBand/RoCE, and **plain TCP sockets**, auto-selecting the fastest available — so it covers *both* hardware tiers we target (a box of old PCIe GPUs and a building of H200s) with one dependency. **RCCL is a symbol-compatible drop-in for AMD ROCm**, so a single NCCL-API binding targets NVIDIA *and* AMD via a runtime library swap. (A hand-rolled custom all-reduce remains an *optional* later micro-optimization for tiny decode messages, the way vLLM keeps one alongside NCCL — not a milestone.)

**The natural first target is still llama.cpp's "layer split" (pipeline / naive model parallelism): each GPU owns a contiguous range of layers, KV cache lives with its layer, one activation tensor is copied GPU→GPU at each stage boundary.** It needs no collective at all (just a point-to-point copy, which NCCL `ncclSend`/`ncclRecv` or a direct `cuMemcpyPeerAsync` both provide), works over plain PCIe, and is the only mode that reliably wins on consumer rigs without NVLink. It scales *memory* (fit a big model) without scaling *latency* (one GPU active at a time). Tensor parallelism (real single-request speedup, NCCL all-reduce, wants NVLink/P2P) and expert parallelism (NCCL grouped send/recv all-to-all, for MoE) are milestones 2 and 3.

This complements [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) (single-GPU VRAM offload / block-swap, the orthogonal "model bigger than one GPU but only one GPU available" answer) and [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md). Block-swap streams a big model through one small GPU slowly; multi-GPU sharding spreads it across several GPUs to run it fast. They stack (shard across GPUs, block-swap on each).

---

# Part 1 — The five parallelism dimensions

Every dimension below is defined by **(a) what gets split, (b) where in the layer the collective fires, (c) the exact collective and its data volume, and (d) the latency/throughput/memory tradeoff.** `b`=batch, `s`=sequence length, `h`=hidden dim, `L`=num layers, `p`=pipeline stages, `m`=microbatches.

## 1.1 Tensor Parallelism (TP) — Megatron-LM style

**What splits:** every weight matrix inside a layer, across `tp` GPUs. The canonical pattern is a *conjugate pair* of sharded linears (Megatron-LM, arXiv 1909.08053):

- **Column-parallel linear** splits weight `A` along its **output** dim: `A = [A₁, A₂]`. Each rank computes `X·Aᵢ` from a **replicated** input → produces a sharded output with **no forward communication**. Used for QKV projection and the first MLP linear.
- **Row-parallel linear** splits weight `B` along its **input** dim: `B = [B₁; B₂]`. Each rank consumes its input shard and produces a **partial sum** → needs an **all-reduce** to combine. Used for the attention output projection and the second MLP linear.

**Exact collective sequence in one transformer layer (inference forward, no sequence-parallel):**
1. Input `X` replicated on every rank (no comm).
2. QKV column-parallel (split **by attention head**, no comm) → per-head local attention (no comm) → output projection row-parallel → **ALL-REDUCE #1**.
3. Residual + norm (replicated).
4. MLP gate/up column-parallel + activation applied locally (no comm) → down-projection row-parallel → **ALL-REDUCE #2**.
5. Residual + norm → next layer.

**→ exactly 2 all-reduces per layer in inference.** Llama-3-70B (80 layers) = 160 all-reduces per forward pass. (Training fwd+bwd = 4/layer.)

**Communication volume per all-reduce (16-bit):** decode `2bh` bytes; prefill `2bsh` bytes. At batch=1 decode ≈ `2h` bytes per all-reduce, `4h` per layer — tiny messages, **latency-bound**, which is why TP is kept **inside one NVLink node**. H100 NVLink 4.0 = 900 GB/s/GPU bidirectional vs inter-node InfiniBand ~50 GB/s/port (≈18× slower); crossing nodes for TP makes the per-token all-reduce the bottleneck.

**Constraints:** `tp` must evenly divide the **attention head count**, the **KV-head count** (GQA/MQA), and `hidden`/`intermediate` dims. Vocab embedding and `lm_head` are **vocab-parallel** (split `E` along vocabulary; each rank looks up its shard, out-of-range masked to 0, then all-reduce to sum). Typical degree: 2/4/8 (8 = one DGX/HGX NVLink island).

**Sequence Parallelism (SP) add-on** (Korthikanti 2205.05198): the otherwise-replicated norm/dropout regions are split along the **sequence** dim, replacing each all-reduce with an **all-gather + reduce-scatter** pair. Zero extra comm volume (a ring all-reduce *is* reduce-scatter + all-gather), but ~5× less activation memory. DeepSeek-V3 serves with TP4 **+SP**.

**Tradeoff:** TP is the **only dimension that lowers single-request decode latency** (all GPUs work the same layer simultaneously). Cost: heavy fine-grained communication; needs NVLink/P2P; bounded by head count.

## 1.2 Pipeline / Layer Parallelism (PP) — the portable one

**What splits:** contiguous **groups of whole layers** ("stages"), one group per GPU. Only **activations** cross a stage boundary, via point-to-point **send/recv** of `b·s·h` bytes — *no collective.* This is llama.cpp's `--split-mode layer` (the default) and HF Accelerate's `device_map="auto"`.

**Schedules** (matter for throughput training; for single-stream decode they collapse):
- **GPipe** (1811.06965): all-forward-then-all-backward + flush; activation memory ∝ microbatches.
- **1F1B / PipeDream** (SOSP 2019): steady-state one-forward-one-backward, bounded activation memory.
- **Interleaved / virtual pipeline** (Megatron 2104.04473): each GPU owns `v` non-contiguous chunks → bubble shrinks by `v×` at `v×` comm.

**Pipeline bubble** (idle time): Megatron form `(p−1)/m`; GPipe/SGLang form `(p−1)/(p−1+m)`. Negligible when `m ≥ 4p`. **At `m=1` (single-request decode) the bubble is `(p−1)/1` — only one stage active at a time.** So PP/layer-split gives **no single-request speedup**; it scales *memory* and (with many concurrent microbatches) *throughput*.

**llama.cpp benchmark, 4× RTX 4090** (PR #4766): prefill — layer 2415 t/s vs row(TP) 250 t/s (layer ~9.6× faster); on slow PCIe (3× P40) layer ran 0.72–0.99× of row. Layer split wins prefill broadly and wins overall on PCIe-class links.

**Tradeoff:** simplest, most compatible, tolerant of slow/inter-node links (P2P, not all-reduce), no collective library. The natural **first** multi-GPU mode.

## 1.3 Expert Parallelism (EP) — for MoE (Kimi-K2, DeepSeek-V3)

**What splits:** the **experts** of each MoE layer across GPUs (each GPU owns a subset). Tokens are routed to whichever GPU holds their selected expert.

**Collective: two all-to-alls per MoE layer** — **dispatch** (send each token to its expert's GPU) then **combine** (send results back). DeepSeek uses **node-limited routing**: each token goes to ≤4 nodes (256 experts = 8 groups of 32, one group per node) to respect the ~4:1 intra:inter-node bandwidth ratio.

**Model configs:** DeepSeek-V3 = 1 shared + **256 routed** experts, **8 active**/token, sigmoid+top-K gating, 671B total / 37B active. Kimi-K2 = 1 shared + **384 routed**, 8 active, 1.04T / 32B active.

**Load balancing (EPLB):** hot experts get **replicated** onto extra GPUs. SGLang's DeepSeek run used 256 base + 32 redundant = **288-expert** pool → 1.49× prefill / 2.54× decode. Production decode places **1 expert per GPU** at EP320 (40 nodes), with extra GPUs for redundant + shared experts.

**Supporting kernels (all native DeepSeek libs, study-not-link for us):** **DeepEP** (all-to-all dispatch/combine; "normal" NVLink kernels for prefill, "low-latency" pure-RDMA kernels for decode), **DeepGEMM** (FP8 grouped GEMM, groups only the M-axis since experts share shape), **EPLB**/**LPLB** (placement planners).

**Tradeoff:** the *only* way to scale MoE expert weight beyond one node; all-to-all is bandwidth-heavy and latency-sensitive. Composes orthogonally with TP/PP/DP.

## 1.4 Data Parallelism (DP) + the DeepSeek DP-attention hybrid

Pure DP = replicate the whole model, split the request batch — only relevant when the model already fits one GPU (not our case). **FSDP/ZeRO** (1910.02054, 2304.11277) shard optimizer/gradient/param state — **training only** (no gradients at inference).

**The inference-relevant case is DeepSeek/SGLang DP-attention + EP.** DeepSeek's MLA has effectively **one KV head**, so TP would *duplicate* the (already tiny) latent KV cache on every rank, wasting memory. Instead:
1. Each DP worker runs MLA + dense layers on its **own** batch, storing only its own small latent KV (compression dim `d_c=512` vs full 16384).
2. **All-gather** hidden states across DP workers before the MoE layer.
3. MoE runs **expert-parallel** (all-to-all dispatch/combine).
4. Results **redistributed** back.

Flags: SGLang `--enable-dp-attention --tp 8 --dp 8`; vLLM `--data-parallel-size 8 --enable-expert-parallel` (effective EP = TP×DP). Reported ~1.9× decode throughput vs vanilla TP. **Official DeepSeek-V3 deployment:** prefill attention TP4+SP + **DP8**, MoE **EP32**; decode attention TP4+SP + **DP80**, MoE **EP320**.

## 1.5 Sequence / Context Parallelism (SP/CP) — long context only

Splits the **sequence** dimension for prefill of very long prompts (compute-bound → cuts TTFT; decode of 1 token benefits little).
- **Ring Attention** (2310.01889): Q blocks stationary, K/V blocks passed neighbor-to-neighbor around a ring with online softmax; KV comm fully overlapped with compute, exact. **Striped/ZigZag** (2311.09431) fixes the causal-mask imbalance (1.45–1.65× measured).
- **DeepSpeed-Ulysses** (2309.14509): all-to-all re-partitions QKV so each GPU gets the **full sequence for a subset of heads**; per-link comm `O(N/P)`. Limited to `P ≤ num_heads`.
- vLLM splits these into **PCP** (prefill CP, cuts TTFT) and **DCP** (decode CP, cuts KV duplication). Snowflake "Arctic Ulysses": up to 3.71× shorter TTFT (Llama-70B FP8, 8×H100).

**Lowest priority for us** — only matters at >32k context.

## 1.6 Composing them — 3D/5D parallelism

`Total GPUs = TP × PP × CP × EP × DP`. Device-mesh ordering heuristic (NVIDIA): **TP innermost** (one NVLink node), **PP middle** (across nodes, P2P), **DP outermost**; EP overlays the MoE FFN orthogonally. Canonical shape: **TP=8 in a node, PP=N across nodes, DP replicating the stack.** Rule: "set TP = GPUs per node, PP = number of nodes."

What the big labs actually serve:
- **DeepSeek-V3 671B:** PD-disaggregated. Prefill 4 nodes/32 GPUs (TP4+SP, DP8, EP32, +32 redundant experts). Decode 40 nodes/320 GPUs (TP4+SP, DP80, EP320, 1 expert/GPU).
- **Kimi-K2 1T:** single node = 8×H200 TP8 + expert-parallel (INT4). Large-scale = 128 H200, PD-disaggregated, 4 prefill + 12 decode nodes, 96 redundant decode experts → 224k tok/s prefill, 288k decode.
- **Llama-3.1-405B:** BF16 ≈810 GB > one 8×80GB node → MP=16 (TP16/2 nodes); FP8 ≈405 GB fits one node at TP8.

---

# Part 2 — How production serving systems shard (and which mode is portable)

| System | Simplest multi-GPU mode | High-perf mode | TP collective | Cross-node |
|---|---|---|---|---|
| **llama.cpp** | **layer split** (`-sm layer`, default; P2P activation copy, **no collective**) | `-sm row`/`tensor` (NCCL all-reduce + CUDA P2P) | NCCL (row/tensor only) | RPC backend (TCP/RDMA) |
| **vLLM** | `-tp N` (single node) | `-dp N --enable-expert-parallel` + EPLB + DBO + DeepEP | NCCL **+ own custom all-reduce** | Ray |
| **SGLang** | `--tp N` | `--ep --moe-a2a-backend deepep --enable-dp-attention --enable-two-batch-overlap` + PD-disagg | NCCL + DeepEP | mooncake/nixl |
| **TensorRT-LLM** | `--pp_size` (send/recv) | `--tp_size` + custom all-reduce plugin | NCCL plugin (+ custom AR) | MPI / mpirun |
| **ExLlamaV2/V3** | `gpu_split` layer-wise (V2) / layer-parallel (V3) | `load_tp()` (V2) / tensor-parallel default (V3) | **own CUDA P2P / pinned-host copies (not NCCL)** | — |
| **MLC-LLM** | (TP only) | `tensor_parallel_shards=N` | **NCCL all-reduce** (TVM Relax `IPCAllReduceRewrite`) | — |
| **DeepSpeed** | ZeRO-Inference offload | `tensor_parallel={"tp_size":N}` (auto-TP) | NCCL all-reduce | torch.distributed |
| **HF Accelerate / Transformers** | `device_map="auto"` (**naive layer-wise, one GPU at a time** — confirmed in docs) | `tp_plan="auto"` + torchrun (DTensor/DeviceMesh) | NCCL | torch.distributed |

**Two takeaways for a from-scratch C# engine:**

1. **Layer split is the universally-supported, no-collective baseline.** llama.cpp default, HF `device_map="auto"`, ExLlamaV3 layer-parallel, the `pp_size` path in TRT-LLM. It is "naive model parallelism" — memory scales, latency does not — and it is *exactly* what's portable: one activation tensor crosses each stage boundary as a plain device→device copy. **No NCCL, works over PCIe.** This is milestone 1.

2. **The custom-all-reduce precedent matters.** vLLM (`csrc/custom_all_reduce.cuh`), ExLlamaV2/V3, and MLC do NOT always rely on NCCL — they ship their own peer-to-peer reduction kernels. vLLM's: supported world sizes {2,4,6,8}, requires GPU P2P (NVLink/NVSwitch for >2 GPUs; disabled on >2 PCIe-only GPUs), message-size gated (≤8 MB → custom, else NCCL), and dispatches **one-shot** (each GPU reads all peers and reduces locally — best for tiny decode messages: <512 KB @4-GPU, <256 KB @8-GPU) vs **two-shot** (reduce-scatter then all-gather, better as size grows). **This is the blueprint for our TP path without NCCL** (milestone 2).

### vLLM / SGLang detail worth keeping

- **PagedAttention under TP:** KV cache stored in fixed-size non-contiguous **blocks**; each GPU holds KV only for its **subset of KV heads** → TP requires head count divisible by `tp`. Increasing TP both shrinks per-GPU weights and frees room for a larger aggregate KV cache.
- **Executor:** vLLM auto-picks `mp` (multiprocessing, one worker process per GPU, shared-memory message queue) for single-node, `ray` for multi-node. SGLang the same shape.
- **DeepSeek serving knobs:** `--deepep-mode normal` (prefill, throughput) vs `low_latency` (decode, CUDA-graph compatible); `--enable-two-batch-overlap` splits a step into two micro-batches so one's GEMMs run while the other's all-to-all is on the wire (+27–35% prefill, ~35% decode @ bs256); `--enable-eplb` for periodic hot-expert rebalancing.
- **Published throughput:** SGLang 96×H100 DeepSeek-V3 → 52.3k input tok/s/node, 22.3k output tok/s/node, ~$0.20/1M output tokens (≈1/5 official API). vLLM Dec-2025 → sustained 2.2k tok/s/H200 for DeepSeek (DBO+EPLB+DeepEP+MLA).

---

# Part 3 — Datacenter-scale serving: disaggregated prefill/decode

The largest deployments split inference into two GPU pools because the two phases have opposite bottlenecks:
- **Prefill** = process whole prompt in one pass, **compute-bound** (GEMM), dominates **TTFT**.
- **Decode** = one token at a time reading full KV cache, **memory-bandwidth-bound**, dominates **TPOT/ITL**.

Colocating them (continuous batching) causes **prefill/decode interference**: a long prefill stalls latency-sensitive decode steps. **Disaggregation** runs each phase on its own pool, each tuned independently (different TP degree, batch policy, even GPU SKU), and transfers the prefill-computed KV cache to the decode pool over the network.

- **DistServe** (OSDI '24, 2401.09670): foundational; **7.4× more requests / 12.6× tighter SLO** vs colocated, optimizing **goodput** (requests within SLO).
- **Mooncake** (2407.00079, the system behind **Kimi**): KVCache-centric. Separate prefill/decode clusters + a disaggregated **KVCache pool** built from idle cluster CPU/DRAM/SSD; **Conductor** scheduler; topology-aware **Transfer Engine** (RDMA/TCP/NVLink/NVMe-oF/CXL, multi-NIC striping → 87 GB/s on 4×200 Gbps RoCE, 190 GB/s on 8×400 Gbps). Up to **525% throughput** (simulated long-context), **75% more requests** (real Kimi workload).
- **NVIDIA Dynamo:** reconfigurable **xPyD** prefill/decode topology; KV transfer **VRAM→VRAM via NIXL** (GPUDirect RDMA, NIC→GPU with no host staging), non-blocking; KV-aware **Smart Router** (route to workers holding relevant cached prefix); tiered **KVBM** cache manager.
- **NIXL** (NVIDIA Inference Xfer Library): point-to-point transfer abstraction over RDMA/IB, RoCE, GPUDirect RDMA, NVLink, NVMe-oF, TCP, S3; agent + descriptor model, async READ/WRITE, UCX default transport. Used by Dynamo, TRT-LLM, vLLM, LMCache.
- **KV transfer granularity:** layer-by-layer streaming (KV for layer L ships as soon as prefill finishes layer L, overlapping with remaining prefill). Heterogeneous-TP transfer uses a contiguous **GPU staging buffer** + one bulk RDMA + scatter (2–5× over per-token).

**Relevance to us:** this is the *server* tier ([`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) §7 noted continuous batching is single-sequence today). Disaggregation is a milestone-4+ concern, only meaningful once we serve many concurrent requests across many GPUs. The portable takeaway: KV cache is a transferable, independently-managed object; a `KVConnector`-style abstraction (vLLM's term) cleanly separates "who computes KV" from "who consumes it."

---

# Part 4 — The communication layer (what we'd actually call)

## 4.1 Collectives, defined

- **all-reduce:** every rank ends with the element-wise sum (or max) of all ranks' input. The TP workhorse (2/layer). Ring algorithm = reduce-scatter + all-gather.
- **all-gather:** every rank ends with the concatenation of all ranks' shards. (DP-attention pre-MoE; SP.)
- **reduce-scatter:** sum across ranks, each keeps a different shard of the result. (SP; ring all-reduce half.)
- **all-to-all:** each rank sends a distinct chunk to every other rank (transpose). The EP/MoE dispatch+combine workhorse, and Ulysses SP.
- **point-to-point send/recv:** one rank → one rank. The PP/layer-split primitive. **No reduction → no kernel → just a copy.**

## 4.2 Hardware fabric

| Link | Bandwidth (per GPU, bidirectional) | Scope |
|---|---|---|
| NVLink 4.0 (H100) / NVSwitch | ~900 GB/s | intra-node, all-to-all GPU mesh |
| PCIe 4.0 x16 | ~32 GB/s (~26 effective) | intra-node, host-mediated or P2P |
| PCIe 3.0 x16 (our 3060 box) | ~16 GB/s (~12–13 effective pinned) | intra-node |
| InfiniBand / RoCE (per port) | ~25–50 GB/s (200–400 Gbps) | inter-node |

This bandwidth hierarchy is **why** TP stays in a node (tiny latency-bound all-reduces need NVLink), PP/EP cross nodes (bigger, latency-tolerant transfers tolerate IB), and node-limited expert routing exists (~4:1 NVLink:IB ratio).

## 4.3 NCCL — the chosen backend (library-policy exception, like cuBLAS)

NCCL (NVIDIA Collective Communications Library) is the collective library every framework above uses. It is a native shared library (`libnccl.so.2` / `nccl64_*.dll`), but it is **the same category as cuBLAS/cuBLASLt, which the engine already depends on and P/Invokes** — a well-maintained, BSD-3-Clause NVIDIA library. The library policy permits such exceptions, so **NCCL is the collective backend.** Concrete facts that make it fit both hardware tiers:

- **Minimum GPU: compute capability 3.5+** (NCCL's own floor). So Maxwell (5.x), **Pascal incl. GTX 1080 Ti (6.1)**, and everything newer is supported. Requires CUDA 10.0+; for Pascal specifically, **pin the CUDA 12.x toolchain** (offline Pascal codegen was removed in CUDA 13.0; PTX-JIT of `sm_61` still works on a ≤580-series driver).
- **No InfiniBand, no NVLink, no MPI required.** NCCL auto-selects transports, fastest→fallback: NVLink GPUDirect → **PCIe P2P** → **SHM (shared host memory)** intra-node; IB/RoCE RDMA → **TCP sockets** (`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=…`) inter-node. A pair of 1080 Tis in one box uses PCIe P2P/SHM; cross-box uses TCP. Knobs: `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_NET_*`.
- **Bootstrap needs only a 128-byte `ncclUniqueId` broadcast** from rank 0 to all ranks by *any* CPU channel — MPI is the convenient option, but **a small built-in C# TCP rendezvous (broadcast the ID + assign ranks) fully replaces MPI**, keeping the engine self-contained.
- **No off-the-shelf .NET binding** (ManagedCuda covers cuFFT/cuRAND/cuSPARSE/cuBLAS/cuSOLVER/NPP/NVRTC but **not** NCCL) → we hand-roll the P/Invoke. It is a clean C ABI (opaque comm pointers, device pointers + a CUDA stream).
- **Cross-vendor:** RCCL (AMD, BSD-3) keeps the `ncclXxx` symbol names → a library swap targets AMD ROCm. Intel oneCCL is a different API (separate path). NVSHMEM (what DeepEP uses for GPU-initiated async all-to-all) is **NVIDIA-proprietary SLA, not open source → avoid a hard dependency**; grouped `ncclSend`/`ncclRecv` does MoE all-to-all adequately and stays license-clean.

**Core NCCL C API to bind:** `ncclGetUniqueId`, `ncclCommInitRank`, `ncclCommInitAll`, `ncclCommDestroy`, `ncclCommAbort`, `ncclGetVersion`, `ncclGetErrorString`; collectives `ncclAllReduce`, `ncclBroadcast`/`ncclBcast`, `ncclReduce`, `ncclAllGather`, `ncclReduceScatter`; point-to-point `ncclSend`/`ncclRecv` wrapped in `ncclGroupStart`/`ncclGroupEnd` (there is **no** public `ncclAllToAll` — grouped send/recv *is* the all-to-all). License: NCCL/RCCL/UCX/OpenMPI all BSD-3, NIXL Apache-2.0, all MIT-compatible; NVSHMEM is the only non-OSS outlier.

## 4.4 CUDA Driver API peer-to-peer (the layer-split boundary + optional custom AR)

For the layer-split boundary copy (milestone 1) and any future hand-rolled tiny-message all-reduce, these Driver-API entry points are needed (same `[LibraryImport]` pattern as our existing `cuMemcpyDtoD`). A direct `cuMemcpyPeerAsync` avoids even an NCCL dependency for the pure point-to-point case:

**Multi-device context management (have the building blocks):**
- `cuDeviceGetCount` ✅ (already bound), `cuDeviceGet` ✅, `cuDevicePrimaryCtxRetain` ✅, `cuCtxSetCurrent` ✅ — our `CudaContext` already wraps one primary context per device. Multi-GPU = **one `CudaContext` per device**, made current per-thread via `cuCtxSetCurrent` before issuing that device's work.

**Peer access — NOT yet bound (must add):**
- `cuDeviceCanAccessPeer(out int, dev, peerDev)` — query P2P capability.
- `cuCtxEnablePeerAccess(peerCtx, flags)` — open direct peer addressing between two contexts.
- `cuMemcpyPeerAsync(dstPtr, dstCtx, srcPtr, srcCtx, bytes, stream)` — the direct GPU→GPU copy (the layer-split boundary primitive, and the building block of a P2P all-reduce). Falls back to staging through host if P2P unavailable.
- `cuMemcpyPeer` (sync variant).

**Multi-process IPC (only if we use a process-per-GPU model like vLLM's `mp`):**
- `cuIpcGetMemHandle` / `cuIpcOpenMemHandle` / `cuIpcCloseMemHandle` (legacy), or the VMM path `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle`.

**Custom all-reduce recipe (no NCCL, the vLLM/ExLlama approach):**
- *Small messages (decode, ≤~512 KB):* **one-shot** — each rank enables peer access to all others, a PTX kernel reads all peers' buffers directly (P2P load) and writes the local sum. Needs full NVLink/NVSwitch mesh to be fast; on PCIe-only >2 GPUs, fall back to host-staged.
- *Larger messages (prefill):* **two-shot** ring = reduce-scatter (each rank sums one shard from P2P-read neighbors) then all-gather (P2P broadcast the shards). PTX add kernels + `cuMemcpyPeerAsync`.
- Cross-device sync via `cuEventRecord`/`cuStreamWaitEvent` (already used by the streaming weight cache).

**Single-process multi-thread vs multi-process:** the simplest model is **one host thread per GPU**, each owning its device's `CudaContext` (set current on that thread), copies between them via `cuMemcpyPeerAsync` — no IPC handles needed, all pointers live in one address space. (Process-per-GPU is only needed to dodge a global interpreter lock, which .NET does not have.)

---

# Part 5 — Model configs & GPU-count sizing

## 5.1 Target model configs (exact)

| Model | Total params | Active/token | Experts (routed/shared) | Active/token experts | Attention | Notes |
|---|---|---|---|---|---|---|
| **Kimi-K2** | 1.04T | 32B | 384 / 1 | 8 | MLA | DeepSeek-V3 lineage, bigger expert pool |
| **DeepSeek-V3** | 671B | 37B | 256 / 1 | 8 | MLA (1 KV head, `d_c`=512) | sigmoid+top-K gating, node-limited routing (≤4 nodes), leading dense layers |
| **Mixtral 8×7B** | 47B | ~13B | 8 / 0 | 2 | GQA | llama arch + experts, softmax renorm |
| **Llama-3.1-405B** | 405B | 405B (dense) | — | — | GQA 128/8 | BF16 810 GB / FP8 405 GB |

## 5.2 Minimum GPU spec (the two hardware tiers)

The feature targets **both** ends of the spectrum with one codebase:
- **Consumer "string together cheap GPUs":** e.g. several GTX 1080 Tis. Floor = **compute capability 6.1 (Pascal)** — NCCL's real floor is CC 3.5, so Maxwell works too, but Pascal is the sane minimum to officially support (Maxwell is end-of-driver-life). Pascal = no tensor cores, no bf16, no FP8 → **F32/F16 inference only** (dequantize quantized weights to F16, which we already do). PCIe-only, **no NVLink**; P2P works on clean bare-metal but is BIOS/IOMMU/topology-fragile → **always probe `cuDeviceCanAccessPeer` and fall back to SHM/host-staged**. Pin **CUDA 12.x + ≤580-series driver** for Pascal. These rigs realistically run **layer split only** (TP's tiny all-reduces are PCIe-latency-bound and lose without NVLink).
- **Enterprise "a building of H200s":** NVLink/NVSwitch intra-node + IB/RoCE inter-node. Full TP×PP×EP×DP, NCCL auto-uses NVLink + RDMA, and disaggregated prefill/decode pays off. Ampere (8.0) / Hopper (9.0) bring bf16/FP8/tensor cores the consumer floor lacks.
- **One abstraction, capability-gated:** detect per-GPU compute capability + interconnect (`cuDeviceCanAccessPeer`, NVLink query) at startup and pick the parallel plan — layer split when there's no fast link, add TP/EP when NVLink/RDMA is present. Never assume; measure the topology.

## 5.3 Minimum-GPU memory math

- **Weights** = params × bytes/param (FP32 4 / FP16 2 / FP8 1 / INT4 0.5). Rule of thumb ≈ 2 GB/B at FP16, ≈ 0.5 GB/B at 4-bit. DeepSeek-V3 FP8 = 671 GB. Kimi-K2 INT4 ≈ 520 GB.
- **KV cache/token** = `2 × L × n_kv_heads × head_dim × bytes` (the 2 = K and V; use **KV** heads under GQA). Llama-3-70B FP16 = 2×80×8×128×2 = **327,680 B ≈ 320 KB/token** → 8k ctx ≈ 2.56 GB/seq. **MLA collapses this** (~70 KB/token) — the reason DeepSeek uses DP-attention not TP-attention.
- **Activations + workspace** ≈ 1.15–1.2× overhead factor.
- **Min GPUs = ceil(total_bytes / per_GPU_VRAM)** with 15–20% headroom (servers reserve 40–60% for KV cache). H100=80, H200=141, B200=192, A100=80, RTX 3060=12 GB.
- **Arithmetic-min ≠ production sizing:** DeepSeek-V3 FP8 fits on ~5×H200 to *load*, but production decode uses **320 GPUs** for throughput/SLO. Minimum-GPU math answers "does it fit"; production answers "does it hit the latency target at scale."

---

## Key Numbers / Constants

- TP: **2 all-reduces per transformer layer** (inference forward). Comm/all-reduce = `2bh` (decode) / `2bsh` (prefill) bytes at 16-bit.
- Pipeline bubble = `(p−1)/m` (Megatron) or `(p−1)/(p−1+m)` (GPipe); negligible at `m ≥ 4p`; `(p−1)/1` at single-request decode (no speedup).
- EP: **2 all-to-alls per MoE layer** (dispatch + combine). DeepSeek node-limit ≤4 nodes/token.
- vLLM custom all-reduce: world sizes **{2,4,6,8}**; ≤**8 MB** → custom else NCCL; one-shot threshold **<512 KB @4-GPU**, **<256 KB @8-GPU**.
- DeepSeek-V3: 256 routed + 1 shared experts, 8 active, 37B/671B. Kimi-K2: 384 routed + 1 shared, 8 active, 32B/1.04T. EPLB redundancy: +32 experts → 288 pool.
- Bandwidth: NVLink4 ~900 GB/s; PCIe4 ~32; PCIe3 ~16 (~12 effective pinned); IB/RoCE ~25–50 GB/s/port. NVLink:IB ≈ 18:1; intra:inter-node ≈ 4:1 effective (drives node-limited routing).
- KV/token: GQA Llama-70B ≈ 320 KB; MLA DeepSeek ≈ 70 KB.
- Published: SGLang 96×H100 DeepSeek-V3 → 52.3k in / 22.3k out tok/s/node; Kimi-K2 128×H200 → 224k prefill / 288k decode tok/s; vLLM ~2.2k tok/s/H200.
- llama.cpp 4×4090 prefill: layer split 2415 t/s vs row(TP) 250 t/s.

## Data Layouts / Formats

- **Column-parallel weight** `[out, in]` → shard along `out` (rows of the `[out,in]` row-major store): rank `r` holds rows `[r·out/tp : (r+1)·out/tp]`. Input replicated, output sharded, no comm.
- **Row-parallel weight** `[out, in]` → shard along `in` (columns): rank `r` holds cols `[r·in/tp : (r+1)·in/tp]`. Input sharded, output partial → all-reduce.
- **QKV head split:** contiguous head ranges per rank; GQA KV heads distributed so each rank's Q heads find their KV group locally.
- **Vocab-parallel embedding/lm_head:** shard `[vocab, h]` along `vocab`; out-of-range token ids masked to 0, all-reduce sums partial embeddings.
- **Layer-split boundary:** the only cross-device datum is the residual-stream activation `[b, s, h]` (or `[b, 1, h]` decode) copied stage→stage.
- **Stacked MoE experts** (GGUF `ffn_*_exps` `[E,·,·]`): already split per-expert by `GgufLanguageModel.SplitStackedExperts`; EP just assigns expert ranges to devices.
- **KV cache:** per-layer, co-located with its layer (layer-split) or its KV-head subset (TP). PagedAttention = fixed-size non-contiguous blocks.

## Algorithm Steps

**Layer-split (milestone 1), one host thread per device:**
```
partition L layers into S contiguous stages by per-GPU free VRAM (or explicit ratios)
load stage s's weights + per-layer KV cache onto device s
forward(tokens):
  x = embed(tokens)                       # on device 0
  for s in 0..S-1:
    cuCtxSetCurrent(ctx[s])
    x = run_layers(stage[s], x)           # all layers this stage own, KV local
    if s < S-1:
      cuMemcpyPeerAsync(buf[s+1], ctx[s+1], x, ctx[s], bytes, xferStream)
      cuEventRecord(done[s], xferStream); cuStreamWaitEvent(compute[s+1], done[s])
      x = buf[s+1]
  return lm_head(final_norm(x))           # on device S-1
```
No collective. Decode reuses the same path (x is `[b,1,h]`). Throughput (not latency) improves by pipelining microbatches across stages.

**Tensor-parallel all-reduce (milestone 2), custom one-shot (decode, NVLink mesh):**
```
once: for each pair (i,j): if cuDeviceCanAccessPeer: cuCtxEnablePeerAccess
all_reduce(local[r], bytes):
  barrier(all ranks)                       # peer-memory flag spin or event
  ptx_kernel: out[r] = sum over k of peer_buffer[k]   # P2P loads from every rank
  barrier(all ranks)
# fallback (PCIe-only >2 GPUs): host-staged ring reduce-scatter + all-gather
```

## Reference Implementations

- **llama.cpp layer/row/tensor split** — `docs/multi-gpu.md`, PR #4766. *The* portable reference; layer split is milestone 1 verbatim. https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md , https://github.com/ggml-org/llama.cpp/pull/4766 ; RPC (multi-node) https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- **vLLM custom all-reduce (no-NCCL TP)** — `csrc/custom_all_reduce.cuh`, `vllm/distributed/device_communicators/custom_all_reduce.py`. One/two-shot, world-size gating, P2P checks. The blueprint for milestone 2. https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh
- **Megatron-LM** (TP math) — arXiv 1909.08053 (ar5iv mirror); SP — 2205.05198; interleaved PP — 2104.04473.
- **ExLlamaV2/V3** — own CUDA P2P / pinned-host TP on consumer GPUs (no NCCL), closest analog to our constraints. https://github.com/turboderp-org/exllamav3
- **DeepSeek** — DeepEP (all-to-all) https://github.com/deepseek-ai/DeepEP ; EPLB https://github.com/deepseek-ai/EPLB ; DeepGEMM https://github.com/deepseek-ai/DeepGEMM ; V3 paper §3.4 deployment https://arxiv.org/abs/2412.19437 ; Kimi-K2 https://arxiv.org/pdf/2507.20534
- **Serving systems** — vLLM parallelism https://docs.vllm.ai/en/stable/serving/parallelism_scaling/ ; SGLang EP/DP-attention https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ , Kimi-K2 EP https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep/ ; TensorRT-LLM https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html
- **Disaggregation** — DistServe (OSDI'24) https://arxiv.org/abs/2401.09670 ; Mooncake https://arxiv.org/abs/2407.00079 + https://github.com/kvcache-ai/Mooncake ; NVIDIA Dynamo https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving ; NIXL https://github.com/ai-dynamo/nixl
- **Long context** — Ring Attention 2310.01889 ; Striped 2311.09431 ; DeepSpeed-Ulysses 2309.14509 ; USP https://github.com/feifeibear/long-context-attention
- **PP/sizing** — GPipe 1811.06965 ; PipeDream SOSP'19 ; ZeRO 1910.02054 ; FSDP 2304.11277 ; sizing https://bentoml.com/llm/getting-started/calculating-gpu-memory-for-llms

## Differences Between Implementations

- **Collective backend:** llama.cpp/vLLM/SGLang/TRT-LLM/MLC/DeepSpeed/HF all default to **NCCL** for TP; vLLM/ExLlama/MLC additionally ship a custom path. **We use NCCL** (library-policy exception, same as cuBLAS); a custom one-shot AR is an optional later micro-opt, not required.
- **"Layer split" naming:** llama.cpp `layer`, HF `device_map="auto"`, ExLlamaV3 layer-parallel, TRT-LLM `pp_size` — all the same pipeline-shaped, one-GPU-at-a-time mechanism.
- **MoE attention strategy:** TP-attention (duplicates MLA KV) vs DeepSeek DP-attention (replicate attention, EP the experts). For MLA models DP-attention is strongly preferred; for GQA models (Mixtral) plain TP is fine.
- **DeepEP number sets:** original-release README (V3-era µs latencies) vs rewritten live README (different GB/s table) disagree — cite both.
- **Disaggregation throughput claims:** vLLM's own docs say its PD impl "does NOT improve throughput" (latency-only); Mooncake/SGLang/Dynamo *do* report goodput gains — implementation-maturity difference, not a contradiction.
- **vLLM custom-AR env var:** there is **no** `VLLM_DISABLE_CUSTOM_ALL_REDUCE` env var (the real control is the CLI flag `--disable-custom-all-reduce`); related real env var is `VLLM_SKIP_P2P_CHECK`.

## Open Questions

- **~~Hardware target~~ (RESOLVED):** support **both** tiers with one capability-gated codebase — consumer "string together cheap GPUs" (1080 Ti class, CC 6.1 floor, PCIe, layer-split) *and* enterprise H200 buildings (NVLink + RDMA, full TP/EP/PP/DP + disaggregation). See §5.2.
- **~~Collective library~~ (RESOLVED):** **NCCL** (BSD-3, same category as the cuBLAS we already use). RCCL for AMD via lib swap. No NVSHMEM (proprietary). See §4.3.
- **~~Multi-node transport~~ (RESOLVED):** NCCL itself carries inter-node traffic over **TCP sockets** when there's no IB (`NCCL_IB_DISABLE=1`), and over IB/RoCE RDMA when present — no separate transport lib, no MPI. Bootstrap via a small built-in **C# TCP rendezvous** that broadcasts the 128-byte `ncclUniqueId` and assigns ranks.
- **Process-per-GPU vs thread-per-GPU:** NCCL's common model is one process/rank per GPU (`ncclCommInitRank` after the ID broadcast); a single-process/multi-thread model via `ncclCommInitAll` is simpler for single-node and .NET has no GIL. Decide per tier (threads single-node, processes multi-node). Confirm no Driver-API context-thread-affinity gotcha.
- **1080 Ti P2P fragility:** PCIe P2P is real on clean bare-metal but breaks under IOMMU/ACS/VM/cross-NUMA. Must runtime-probe `cuDeviceCanAccessPeer` and fall back to SHM/host-staged — never assume. NCCL handles this internally, which is another reason to lean on it.
- **CUDA-graph capture across devices:** vLLM/SGLang capture per-step graphs; whether our `CudaGraph` can span a multi-device step or must be per-device is unverified.
- **Optional custom-AR win window:** if we later add a hand-rolled one-shot AR for tiny decode messages, how close to / better than NCCL on our hardware? Narrow window (≤8 MB, ≤8 GPUs, full NVLink). Microbenchmark before investing. Not on the critical path.
- **Exact MLA + EP interplay** when wiring DeepSeek-V3 routing (slice-verified, Phase 8a) onto real multi-GPU EP — node-limited-routing all-to-all dispatch is non-trivial.

## Implementation Notes for HartsyInference

**Backend:** **NCCL** (library-policy exception, same category as the cuBLAS/cuBLASLt the engine already P/Invokes) for all collectives; RCCL for AMD via lib swap. The layer-split boundary can use NCCL `ncclSend`/`ncclRecv` or a direct `cuMemcpyPeerAsync` (no NCCL needed for pure point-to-point).

**Current state (updated 2026-08-02 — milestone 1 below has shipped, see `ROADMAP.md` §1 for the live
tracker; this section now describes what's built vs what NCCL/TP still needs):** `CudaContext` wraps
**one primary context per device** (`cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`), `GetDeviceCount`
exists, `cuMemcpyDtoD` is bound, and the engine already has `CublasApi`/`CublasLtApi` native-lib bindings
+ a `CudaLibraryResolver` for runtime lib-name resolution (the pattern an `NcclApi` binding follows).
`cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, and `cuMemcpyPeer`/`cuMemcpyPeerAsync` are now bound
(`CudaDriverApi.cs`) and wired into `IBackend.CopyFromPeer` (P2P path + host-staged fallback, per-pair
probe/enable memo in `CudaPeerAccess`) — layer-split (milestone 1) uses this and topology probing is
`CudaTopology.Probe()`. **Still missing:** a new `NcclApi` (resolve `nccl`→`libnccl.so.2`/`nccl64_*.dll`
via `CudaLibraryResolver`) for milestone 2+ (tensor/expert parallel need real collectives; layer-split
needed only point-to-point copy, which is why it shipped first). The streaming weight cache already has
the `cuEvent`/`cuStreamWaitEvent` sync milestone 2 would reuse cross-device.

**Diffusion-side status (2026-08-05, hardware-verified on the 4090+3060 box):** the diffusion multi-GPU
surface built on the milestone-1 machinery is now live in three shapes. (1) **TE/VAE component placement**
(`PlacementConfig.TextEncoderDevice`/`VaeDevice`, extension `TextEncoderGpuId`/`VaeGpuId`, CLI
`--te-gpu`/`--vae-gpu`) is wired fleet-wide — verified end-to-end for Wan TE (`WanComponentPlacementEngineTests`,
SSIM 0.7665), Wan VAE (`WanVaeComponentPlacementEngineTests`, SSIM 0.9999 — **Wan had zero VAE placement**
until this pass wired `WanVideoPipeline.VaeBackend`, which the base class already exposed but no call site
used), Flux (`FluxComponentPlacementEngineTests`, SSIM 0.8126 from fp8-T5 cross-SM drift on the mismatched
pair; matched cards are expected bit-identical), SDXL (`SdxlComponentPlacementEngineTests`, SSIM 0.9998),
and LTX-1 (`LtxVideoComponentPlacementEngineTests`, TE **and** VAE, 16.4 s → 10.2 s, SSIM 0.9943 — TE was
already wired via `LtxVideoRecipePipeline._textBackend`, but **VAE was not**; both are wired and verified
now). LTX-2's code was already fully wired (both `TextEncoderBackend` and `VaeBackend`, including the
separate audio-VAE+vocoder path) before this pass — `LtxVideo2ComponentPlacementEngineTests` now exists but
has not run for real on this box (the ~22 GB checkpoint split isn't downloaded here; disk-constrained, not a
code gap). Qwen-Image, Chroma, and HunyuanImage are wired but **UNVERIFIED** — no
`ComponentPlacementEngineTests` class exists for any of these three as of 2026-08-05; do not cite them as
verified until a matching engine test lands (tracked in the multi-GPU finish-out plan, Phase 3.4).
(2) **DiT block-range sharding** (`DitShardGpuId` / `--dit-shard-gpu` — VRAM
pooling, not latency, i.e. milestone 1's memory-scales contract applied to DiTs) is verified for six
models: Krea2 (e2e SSIM 0.8787), Qwen-Image 20B (`QwenImageDitSharding{,Vram,Engine}Tests`: 19.6 GB
pooled 13.4+6.2 at the live 41/60 split, SSIM 0.9734, drift 0.00 GB), MiniMax-H3 fp8
(`MiniMaxH3DitSharding{,Vram}Tests`: 19.76 GB pooled at 34/50, finite video+audio; the 66 GB bf16 build
is excluded — it exceeds any 2-consumer-card pool), Flux.1 plain path (same-device split bit-exact over
262k values, cross-device 30/57 pooling 7.7+3.7 GB, engine SSIM 0.9075; ControlNet/Kontext/inpaint/
regional fall back unsharded with a log), Chroma (bit-exact both regimes; real fp8 e2e SSIM 0.8797), and
HunyuanImage (synthetic bit-exact; engine e2e written, awaiting checkpoint). Sharding is mutually
exclusive with step-graph/step-cache/streaming by design. (3) **CFG-parallel** decisions are observable
(`DiffusionPipelineBase.LastCfgParallelDecision` + the `[CfgParallel]` log line: active /
fell-back(reason) / inapplicable(no-true-cfg)), with `FluxCfgParallelFallbackTests` green and Wan's
preload-OOM→sequential fallback in place. Verification runs through `tests/run-multigpu-campaign.sh`
(per-class isolation + `HARTSY_REQUIRE_REAL_WEIGHTS=1`; conventions in `PROFILING_METHODOLOGY.md` §15).
The live tracker stays `ROADMAP.md` §1; the placement pattern's detailed authority is
`MULTI_GPU_COMPONENT_PLACEMENT.md`.

**Recommended build order (each independently shippable):**

| # | Milestone | What it enables | Collective | Effort | Hardware to verify |
|---|---|---|---|---|---|
| 1 | **Layer split (pipeline)** — one `CudaContext` per device, contiguous layer ranges sized by free VRAM/ratios, KV co-located, `cuMemcpyPeerAsync` (host-staged fallback) at stage boundaries — **✅ shipped 2026-08-02**, verified on 4090+3060 (Llama-3.2-1B, exact token parity, VRAM genuinely pooled) | Run a model 2-N× too big for one GPU; **memory scales** (not latency) | **none** (P2P copy) | Medium | 2-N cheap GPUs (1080 Ti class), PCIe — verifiable without NVLink |
| 2 | **NCCL binding + tensor parallel** — `NcclApi` P/Invoke + TCP rendezvous bootstrap; column/row-parallel linear loaders; 2 `ncclAllReduce`/layer | **Latency** speedup for a single request; the real "fast" mode | NCCL all-reduce | High | needs NVLink/NVSwitch to pay off; PCIe-only >2 GPUs ≈ no benefit |
| 3 | **Expert parallel** — distribute experts across devices, all-to-all dispatch/combine via grouped `ncclSend`/`ncclRecv` (reuse `SplitStackedExperts`) | Scale MoE expert weight (Kimi-K2/DeepSeek/Mixtral) past one GPU | NCCL grouped send/recv | High | multi-GPU node |
| 4 | **DP-attention + EP hybrid** — replicate MLA attention per DP rank, `ncclAllGather` before MoE, EP the experts | The efficient DeepSeek/Kimi serving recipe (no MLA KV duplication) | all-gather + all-to-all | High | datacenter |
| 5 | **Disaggregated prefill/decode** + `KVConnector` abstraction + KV transfer | Datacenter throughput/SLO at many concurrent requests | KV transfer (NCCL/TCP first; RDMA later) | Very high | multi-node cluster |
| — | **Sequence/context parallel** | >32k-context TTFT | ring/all-to-all | High | low priority |

**Design recommendations:**
- **Build the parallelism plan as config, not new model classes** — mirror `MODEL_STATUS_LLM.md`'s "preset + knob" philosophy. A `ParallelConfig { TpSize, PpSize, EpSize, DpSize }` consumed by `GenericTransformer`, with the layer loop dispatching to the right device(s). TP is column/row loader variants of the existing Linear; EP is a device assignment over the existing expert split; PP is a stage-partition over the existing layer loop.
- **Start single-node, thread-per-GPU, in-process.** No IPC handles, no MPI, no Ray; one host thread owns each device's context. Defer multi-node (and its native-transport question) entirely.
- **Layer split first and possibly forever for consumer hardware** — without NVLink, TP's small all-reduces are PCIe-latency-bound and lose (the 3× P40 data: layer ≥ row except on token-gen). On a 2-GPU consumer rig, layer split is the only mode that reliably wins.
- **Reuse the existing P2P/event machinery** from `CudaStreamingWeightCache` (pinned staging, `cuMemcpyHtoDAsync`, `cuEventRecord`/`cuStreamWaitEvent`); the layer-split boundary is structurally identical to the block-swap transfer, just device→device instead of host→device.
- **Stack with block-swap:** layer split across GPUs + block-swap on each GPU = run a model larger than the *sum* of GPU VRAM. The two are orthogonal and compose.
- **Gate verification by hardware honestly** (per the Build-defer policy): layer split is verifiable on any 2-GPU box; TP/EP speedups are only meaningful (and only honestly claimable) on NVLink hardware. Slice-test the loaders/collectives against a CPU reference (as Phase 8a did for MLA), defer e2e throughput claims to real multi-GPU access.
