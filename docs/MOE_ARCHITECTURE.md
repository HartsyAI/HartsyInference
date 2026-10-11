# Sparse (MoE) execution architecture

Describes what a model declares to the runtime and what the runtime may decide. Implementation status lives in
[MODEL_STATUS](Checklists/MODEL_STATUS.md) and [ROADMAP](Checklists/ROADMAP.md); research and provenance live in
[Research/HETEROGENEOUS_MOE.md](Research/HETEROGENEOUS_MOE.md).

## Layering

```
Model code (LLM / DeepSeekV41 / Qwen35 …)  → builds SparseModelTopology; owns router math, expert program, state
Core/Moe (topology contracts, CPU-safe)     → descriptors, capabilities, fingerprint
Runtime (placement, cache, scheduler)       → reads capabilities only; never branches on an architecture name
Backends (CPU / CUDA / Vulkan)              → routing, dispatch, combine, expert kernels, transfers
```

The dependency runs one way: a model declares, the runtime executes. A model never names a device, a cache slot, or a
transfer.

## Topology

`SparseModelTopology` is an ordered list of `SparseLayerDescriptor`. Each layer is dense (`Moe == null`) or sparse
(`MoeLayerDescriptor`), keeps one `SequenceStateKind`, and is either a target layer or a draft (speculative) layer.

A sparse layer is:

- **`RouterDescriptor`** — routed-expert count, top-k per phase (`TopKDecode`, `TopKPrefill`), scoring, group limiting,
  renormalization, scale, logit divisor, and whether a selection bias or a token-kind bias applies. It lowers to
  `MoeRouteArgs` when the width fits `MoeRouteArgs.MaxExperts`; otherwise routing stays on the host
  (`CanLowerToBackend == false`, reported as `RequiresHostRouting`).
- **`ExpertGroupDescriptor`** — routed experts, and shared experts. A group has a default `ExpertDescriptor` plus
  optional per-index overrides, so experts of different width or dtype in one layer are representable.
- **`ExpertDescriptor`** — hidden width, intermediate width, storage `DType`, and layout (split or fused gate/up).
  Payload bytes come from `DType` block geometry, so a block-quantized dtype counts correctly.
- **`ExpertProgram`** — `down(act(clamp_gate(gate(x))) * clamp_up(up(x)))` as data: activation plus independent gate
  upper bound and up lower/upper bounds. Plain SwiGLU and DeepSeek-V4.1's clamped SwiGLU are two instances. The runtime
  executes programs; it does not hard-code SwiGLU.

A layer's shared group, if present, runs for every token; `SharedIsGated` marks a sigmoid-gated shared output
(Qwen2-MoE). Whether the gate tensor exists is a fact about the checkpoint, so the caller supplies it.

## Device-resident routed stage (CUDA)

A sparse layer's routed experts run on the device without a host round trip when the backend declares the ops and the
layer is eligible: flat softmax or sigmoid routing (no expert groups, no logit-space bias), and experts stored as one
quantized type per projection that the backend can read. Everything else keeps the host-routed per-expert loop in
`MoeFeedForward.Forward`.

**Small batches (at most 16 tokens: decode, speculative verify).**
`router GEMV → MoeRoute → MoeExpertGateUp → MoeExpertDown → MoeCombineSlots`. The expert ids stay on the device. The
gate/up kernel reads the expert id of every (token, slot) pair from device memory, dots both projection rows against the
token's Q8_1-quantized activation with `dp4a` and writes `act(gate) * up` in the same launch; the down kernel does the same
with the quantized product. The combine folds in the shared expert (sigmoid-gated when the checkpoint has the gate). The
experts of a projection are one resident allocation, back to back (`PreloadWeightGroups`); the backend resolves the base
address and stride and refuses a group whose members were re-placed. Nothing in the stage reads the host, so it is
captured inside the CUDA-graph decode step. `IBackend.SupportsMoeExpertIndexed` reports the quant types (Q4_K, Q6_K and
Q8_0 today).

**Large batches (prefill).**
`router GEMV → MoeRoute → MoeBuildDispatch` orders the (token, slot) pairs by expert. The layer reads the `E + 1`
offsets once and passes them to `MoeExpertsGrouped`, which runs each active expert's gate, up and down GEMMs on its
contiguous rows. When most experts are active the layer's three stacks are dequantized to BF16 in one launch each (a
stacked group is one flat run of quant blocks), cached on the layer's first expert while free memory stays above the
quantized-weight headroom, and every batch of experts takes three `cublasGemmGroupedBatchedEx` calls. cuBLAS accepts 16-bit
grouped operands only with a 16-bit result, so gate, up and the down rows are BF16 there and the down rows are widened to the
F32 expert-major buffer. A refused grouped call falls back to one GEMM per expert. `MoeCombinePairs` sums each token's rows
through the pair map and adds the shared expert.

Two measured choices sit in that path. A quantized stack becomes BF16 by dequantizing to F16 straight into the destination
buffer and converting in place (`moe_cast_f16_to_bf16_inplace`); the generic route staged the whole stack through F16 and F32
temporaries, three passes and 4x the memory, which cost about a fifth of a Mixtral prefill. And when the active experts average
384 rows or more (Mixtral: about 1000), each expert takes a dense `cublasGemmEx` instead of the grouped call, which reached about a
third of the tensor-core rate on those shapes. Many small experts (Qwen3-30B-A3B: about 60 rows each) stay on the grouped call.

**Attention** is not part of the MoE contract but decides the end-to-end result for these models. Prefill of 16 or more query
rows runs `flash_attn_causal_f16` (mma.sync FlashAttention-2, grouped-query, window and offset aware); one decode row per
sequence runs `flash_attn_decode_gqa`, which reads each KV head's cache once for all of its query heads and writes the split-K
partials the existing combine merges.

The decode attention kernel prefetches the next key/value tile into registers while it computes the current one, keeps F16 tiles
F16 in shared memory, and sizes its split count to one wave of resident blocks (four per SM with an F16 cache, two with F32); a
half-empty second wave cost about a sixth of the kernel at 32K context.

**Sampling in the graph.** A sampled decode step draws on the device (`lm_topk_f32` then `lm_sample_from_topk`, top-k up to 64).
Over a vocabulary the top-k runs as two stages: `lm_topk_slices_f32` takes each slice's k largest on its own block (indices already
global), the one-block kernel merges at most 2048 survivors, and the sampler maps merged positions back to token ids. Ties keep the
lowest index through both stages. The device draws with splitmix64, the host chain with its own generator, so one seed gives different
tokens on the two paths (the distributions match, which `CudaDeviceSamplerTests` checks); a request is not reproducible across them. One block over 152K logits took 550 us per step; the two stages take about 55 us.

**Graph capture cost.** Capturing a decode graph walks every weight tensor once to decide which fit the preload budget. That walk
was quadratic in the tensor count (a MoE checkpoint has thousands), about 330 ms per request on Qwen3-30B-A3B; it is one pass now,
and capture takes about 17 ms.

Knobs (all default on): `numerics.moeIndexed`, `numerics.moeGroupedGemm`, `numerics.fa2Prefill`, `numerics.flashDecodeGqa`,
`numerics.lmNormFast`. `vram.kvF16` works with graph decode.

## Capabilities

`SparseCapabilities.From(topology)` derives what the runtime may rely on: sparse/shared/dense presence, variable experts
per layer, heterogeneous shapes, variable or phase-dependent top-k, group routing, token-kind routing, draft layers,
clamped programs, host-routing requirement, the set of sequence-state kinds, the set of expert dtypes, and the scoring
kinds. Generic code checks these flags. It never checks a model name.

## Fingerprint

`SparseModelTopology.Fingerprint` is a SHA-256 over shapes, routing, programs, state kinds, and draft flags. It excludes
the `Family` label. Profiles and packs must record it and refuse to apply to a different fingerprint. The fingerprint
covers only the topology. Weight identity (checksums, source hashes) is a separate concern owned by the pack format.

## Correctness contract

Heterogeneous execution (CPU vs GPU experts, cache state, PCIe promotion, adaptive swaps) can change rounding, and
therefore greedy output. Validation therefore requires, in order of strictness:

1. Exact routing: identical expert IDs and identical routing weights for the same inputs.
2. Bounded per-layer numerical error against the reference path, with the floor stated per change.
3. Logit KL and perplexity deltas on a fixed corpus for end-to-end changes.
4. Identical greedy tokens only for configurations that are deterministic by construction (fixed placement, fixed
   split, frozen cache state).

## Reference path

`Core.Moe.SparseFfnReference` executes a routed layer on the host in F32 from its descriptor. It composes the backend
op contracts (`MoeReference.Route`, `BuildDispatch`, `Combine`) and runs each expert through `ExpertProgramReference`,
a scalar implementation of the program. Backend expert kernels are to be checked against it; no backend kernel check
exists yet. Parity with the production `MoeFeedForward` is covered for softmax routing and for grouped sigmoid routing
with a selection bias. Token-kind routing is supported through the alternate bias. Shared experts, flat logit-space
selection (production `SigmoidLogitAdd`), and routers wider than `MoeRouteArgs.MaxExperts` are rejected explicitly
until covered.

## Open items at this layer

- Draft (MTP/DSpark) layers: gate-bias and shared-expert layout is not loaded yet, so their topology carries no selection
  bias and no shared expert. They do carry the backbone's clamped SwiGLU, which assumes the model-wide `swiglu_limit` applies
  to drafts. Confirm both against the checkpoint before running draft experts.
- Engram hash tables (DeepSeek-V4.1) are not MoE experts and are not in the topology yet; they belong to the auxiliary
  storage tier (see the placement milestone).

## Expert identity and sources

An expert is identified by `ExpertKey(Layer, Expert, Bank)`. The bank is cache-scoped: a model that shares a cache with
others gets its own bank, so the same layer number in two models is two layers. `Bank` defaults to 0, so single-model code
is unchanged. The cache counts, protects and evicts by `ExpertLayerKey` (bank and layer). A model describes where its experts
come from through `IExpertSource`, whose `ExpertBacking` (resident host, memory-mapped, pack, device) lets the runtime choose
policy. The bank resolves each expert once, and the cache copies from the bank; the model never sees device slots.

## Residency queries

`IResidencyAwareExpertCache.LookupResident` answers which experts are resident or already uploading, without changing any state.
`IResidencyAwareExpertCache.AcquireResident` pins only those experts and returns the rest as misses, and it never uploads or resolves.
A runtime that splits a routed batch uses these to run resident experts on the device and misses elsewhere, without
forcing every miss into the cache. The plain `Acquire` keeps its meaning: pin everything, uploading what is missing.
