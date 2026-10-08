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

`IExpertCache.LookupResident` answers which experts are resident or already uploading, without changing any state.
`IExpertCache.AcquireResident` pins only those experts and returns the rest as misses, and it never uploads or resolves.
A runtime that splits a routed batch uses these to run resident experts on the device and misses elsewhere, without
forcing every miss into the cache. The plain `Acquire` keeps its meaning: pin everything, uploading what is missing.
