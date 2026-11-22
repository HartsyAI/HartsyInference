# Attention support and admission contract

[The mathematical contract](ATTENTION_CONTRACT.md) defines the result of an admitted call. This document
defines which ordinary calls every backend must serve and which capabilities may be optional. It is a proposed
PR 03 review contract, not a claim that current implementations conform. It adds no production admission code.
The [evidence program](../benchmarks/attention/README.md) owns hardware and numerical verification.

## Required common subset

These cases must execute on CPU, CUDA and Vulkan. A backend may choose a native implementation or an explicit
device-side lowering. On a GPU backend, host reference execution does not satisfy a required case.
No generic GPU kernel is required to support every optional combination.

The initial common domain is F32 Q/K/V/O, contiguous head-major storage, equal Q/K and V/O head dimensions
64 or 128, batch 1 or 2, four query heads, positive finite scale, and disjoint output. The small conformance
corpus uses Sq in {1,3,7}, Skv=11 and scale=0.125, so unequal lengths and a scale independent of dimension are
tested. These corpus sizes are minimum verification cases, not new production limits.

| Case | Required semantics | Minimum evidence |
|---|---|---|
| M01 | Noncausal MHA; all eleven keys participate, even when Sq is smaller. | Nonzero Q/K and V; a changed late key changes output; double oracle. |
| M02 | F32 additive bias with broadcast over batch, head and query axes, stored key axis of eleven elements. Rank 2/3/4 forms compose with scores. | Shared, per-head, per-batch and key-only bias; mixed positive/negative values and a nonzero dot product. |
| M03 | Negative-infinite bias excludes a key; all excluded rows return exact zero, mixed rows still normalize over their visible keys. Finite sentinels remain finite biases. | Excluded and partially visible rows; finite `-1e30` case reported separately; NaN is not accepted as zero. |
| M04 | MHA, GQA and MQA with Hkv=4,2,1; query head h reads KV head floor(h/(4/Hkv)). | Distinct values per KV head; native versus repeated-head lowering recorded. |
| M05 | Causal attention uses logical bottom-right alignment: qOrigin=11-Sq, kOrigin=0. | Prefill and single-token decode; exact valid lengths and absolute positions. |
| M06 | Causal left window of two positions, inclusive; future keys remain excluded. | A boundary key at qpos-2 participates; qpos-3 and future keys do not. |
| M07 | Dense cache capacity sixteen with uniform valid length eleven; unused capacity changes no result. | Change tail values, including non-finite tails, without changing live output; record any gather/copy separately. |
| M08 | Contiguous token-major, batch one, with the same MHA/GQA/MQA noncausal semantics. | Logical output equals head-major oracle; permutations/repetition are explicit and measured. |
| M09 | A malformed call is rejected before output writes; a valid optional unsupported call is distinguished from an invalid argument. | Sentinel output, invalid head ratio/extents/view/bias/alias and no execution observation. Shared enforcement remains PR 04 work. |

M01-M03 use the dense call; M04-M07 may use the existing Flash entry to carry grouped heads, lengths and
positions. M08 uses the token-major entries. The semantic requirement does not mandate a legacy entry forever.
The current `Auto` path is not a calibrated all-backend planner; unifying execution and adapters is separate work.

The required F32 numerical policy is `AttentionNumerics.Exact`: no implicit TF32, narrowing, quantization,
saturation or low-precision accumulation. A strict benchmark profile may isolate arithmetic while existing
admission is repaired, but that profile does not prove that a default/forced approximate route honors Exact.
The default-knob, forced-knob, refused-format and numerical-grant matrix must also pass before claiming admission
conformance. A numerically plausible output from an unpermitted route fails.

## Optional capabilities

For a valid combination outside the common subset, a backend may execute it correctly or raise
`AttentionUnsupportedException` (a `NotSupportedException`) before writing output. Unsupported must identify
the unmet feature, layout, numerical permission or budget. Optional status is not permission to omit a term,
change its meaning, or silently copy GPU attention to the host.

| Capability | Contract when admitted | Current representation or limit to investigate |
|---|---|---|
| F16/BF16 storage or compute, TF32, INT8, FP8 | Exact stored input values and independent numerical grants; record actual QK/PV/accumulation formats and saturation. | `AttentionNumerics`, capability declarations and backend predicates; declarations alone are not proof. |
| Other head widths, batches, sequence sizes | Same mathematical result; preserve caller scale when padding. | Row shape/address/workspace limits apply; not every backend admits every dimension. |
| Offset/strided/packed views, token-major batches | Respect all element strides, offsets, bounds and output ownership. Copies must be declared. | CUDA has aligned cuDNN view rows; Vulkan token-major rows are batch-one; CPU reference reads views. |
| Dqk different from Dv | V/O width changes only value/output columns. | Host reference supports it; current F32 Auto flash lowering handles narrower V through padding/copy, not every mask/dtype combination. |
| Noncausal windows, right windows, block-causal masks | Apply the specified logical exclusions, independently of causality. | `AttentionMask` carries these; legacy entries cannot carry every composition. Existing noncausal-window differences remain gaps. |
| Softcap, ALiBi and sinks | Score order and non-finite behavior follow the mathematical contract; negative-infinite sink is absent. | CPU/CUDA Flash implement terms. Vulkan has a restricted unmasked F32 wide-sink route with optional softcap; other unsupported terms can still select host reference, a GPU-contract gap. |
| Zero uniform valid length | Admitted empty rows are zero, with LSE negative infinity if requested. | Host reference/CPU can represent zero; current CUDA Flash validation requires at least one key. Do not label this universally admitted. |
| Device positions/graph replay | Backend-owned valid addresses, capture-safe workspaces and replay lifetime; explicit error status for invalid positions. | CPU rejects device positions; CUDA/Vulkan have backend-specific rows/handle ownership. Address safety is not proven by a capability flag. |
| LSE, per-sequence Q/K lengths, paged views | Exact live lengths and denominators; no capacity-tail contribution. | A call can give a key length per sequence and name an LSE output (`AttentionCall.SequenceLengths`, `.Lse`): the host reference and the CPU backend run both; CUDA and Vulkan run a decode with sequence lengths, with or without the LSE (`flash.decode.lengths`: F32 queries and output; CUDA an F32, F16 or BF16 cache, head dim 64, 128 or 256 and group at most 16; Vulkan an F32, F16 or BF16 cache, a head dim of one to eight subgroup widths and at most 256, and group at most 8); every GPU refuses an LSE on a call with one shared length. No model or scheduler issues such a call yet. Paged keys and values: CUDA reads them through the page table for one causal decode row (`flash.decode.paged`: F32 pages of a power-of-two size, head dim 64 or 128, at least 64 valid keys, no sinks or ALiBi); every other paged call, and every paged call on CPU and Vulkan, is gathered by the caller into a contiguous buffer. |
| Boolean exclusions, segment IDs, arbitrary key-axis broadcast | Preserve exclusions and broadcast semantics; no implicit finite-sentinel substitution. | Several mathematical-contract components have no corresponding current `AttentionMask` field or legacy carrier. Record a contract/adapter gap, not imaginary support. |
| Rotated/prequantized attention, speculative merge, general Sol-Attn | Validate the specified algorithm and dense accuracy independently; retain complete sparse/merge metadata. | Specialized VSA is not general Sol. The pinned Kitchen feature program owns these implementation gates. The speculative merge exists as a backend primitive (`IBackend.DecodeTreeMerge`, defined by `DecodeMergeReference`): the CPU runs the reference, CUDA and Vulkan each have a kernel for F32 operands and at most 32 step rows, and nothing calls it yet. Rotated INT8 has a CPU reference, Vulkan quantize and attend entries that nothing calls, and on CUDA the INT8 v2 rows behind `numerics.int8AttnV2Rotate` (off by default, nothing sets it; H4 and signed H128 at head dim 128, see [contract §9](ATTENTION_CONTRACT.md#9-where-kernels-differ-today)); general Sol-Attn has a CPU reference only. |

The pinned Kitchen matrix independently makes required features mandatory for CUDA and Vulkan at final
completion. A capability being optional in the CPU/CUDA/Vulkan common subset does not make a required Kitchen
feature out of scope, waive its performance gate, or permit unsupported results at program completion.

## Precision-specific acceptance

Use the unchanged [conformance tolerances](ATTENTION_CONTRACT.md#8-conformance). Round inputs to their exact
storage bytes before the double oracle. CPU execution may use higher precision than a permitted GPU route;
identical semantics do not require identical rounding. A permission is not an obligation to approximate.

| Admitted precision | Existing relL2 bound | Additional condition |
|---|---:|---|
| F32 only | 2e-5 | No undeclared TF32/narrowing/saturation. |
| TF32 multiply | 2e-3 | Granted for each multiply actually using it. |
| F16 storage, F32 accumulation/softmax | 3e-3 | F16 operand/computation permission and range policy are explicit. |
| BF16 storage | 8e-3 | No implicit substitution of F16, whose exponent range differs. |
| BF16 compute | 1e-2 | Actual compute precision is recorded separately from storage. |
| INT8 Q/K | 1e-2 | Also mean absolute error <=2e-3 and the declared quantized-algorithm checks. |
| A route without an existing precision/algorithm tolerance | Unreviewed | Define and review its gate before examining candidate errors. |

Apply the contract's absolute floor and exact non-finite classifications; an all-zero reference must not hide a
nonzero/NaN result behind a relative-error denominator. NaN/Inf cases need classification checks outside an
ordinary finite relL2 calculation. Model-generation quality and permitted precision are independent gates.
Matched-quality timing requires equivalent policies and declared preparation/reuse; faster unpermitted math
cannot pass by staying inside a broad numerical tolerance.

## Failure and ownership obligations

- Validate invalid inputs and optional unsupported combinations before output writes. A failed preflight cannot
  be disguised as a successfully executed zero-case test.
- A recoverable decline means no externally visible state/output changed; then another admitted row may run.
  A launch failure or exception after writes began propagates, with no retry into a different implementation.
- GPU attention never silently falls back to host execution. A call no device row serves is refused unless the
  backend declares a host-reference row for it, named in the plan and the evidence and warned once; Vulkan Flash
  declares one (a migration gap, not a conformant GPU implementation) and CUDA declares none.
  `AttentionHostReference.Run` is the explicit host-execution path, and a backend without rows refuses
  `IBackend.Attention`. Exhausted recoverable declines raise `AttentionRecoverableException`, not an unsupported error.
- Knob compatibility, explicit legacy numerical-permission adapters, workspace/capture refusal and device/page
  ownership remain separately reviewable work. This document does not implement those mechanisms.

## Current evidence and gaps

Source admission is not runtime conformance. The following are review pointers, not a support guarantee:

| Area | Source/evidence owner | Current limitation |
|---|---|---|
| CPU selection | `CpuBackend.AttentionRows.cs`, `AttentionKernels.cs`, `FlashAttentionKernels.cs` | Dense SIMD handles plain MHA; grouped/view calls can use the reference. A read-only PR 03 diagnostic reproduced NaN for an all-negative-infinite dense row, violating M03. Fix requires semantic follow-up, not a mechanical move. |
| CUDA admission | `CudaBackend.AttentionRows.cs`, `CudnnSdpaPolicy.cs` | Native rows, explicit lowerings, free-memory routing and approximate-format knobs still need the PR 04 matrix. Flash beyond head dim 1024 is explicitly unsupported. |
| Vulkan admission | `VulkanBackend.AttentionRows.cs`, `VulkanBackend.cs`, `VulkanBackend.FlashWide.cs` | Scalar/tiled/CM2 support differs by dtype, dimension, view and mask. A dense call whose operands are all F32 takes the query-tiled F32 kernel by default up to head dim 128 (`numerics.vkTiledAttention`; the rows and the rule are in [contract §9](ATTENTION_CONTRACT.md#9-where-kernels-differ-today)), measured on one RTX 3060 only. Flash calls outside native scalar/wide admission run the declared host-reference row (C.1): softcap, causal sinks, ALiBi and wide head dims have no device kernel yet. Noncausal-window differences remain open. |
| Excluded rows, ordering and non-finite normalization | PR 02 [semantic progress](../benchmarks/attention/reports/pr02-semantic-progress.md) | 3060 route-forced cases are scoped evidence; do not generalize cuDNN 9.21 results to every version or to all common cases. |
| Kitchen and real generation | PR 01 [local status](../benchmarks/attention/reports/README.md), [GPU assignment](../benchmarks/attention/gpu-agent-handoff.md) | Hardware report and baseline/workload freeze remain pending; no Bfix, final parity or primary speed claim follows from this matrix. |

Before marking a row verified, retain case ID, exact call/fixture hashes, backend/device/cohort, numerical policy,
executed implementation and nested fallback, preparation/workspace facts, oracle/quality result and artifact
identity. Refused cases retain the reason and unchanged-output evidence. Missing assets, unexecuted routes,
resource-guarded passes and observation overflow are unvalidated.

PR 03 preserves existing implementations while the proposed common/optional matrix is reviewed. Current
nonconformance is not an allowed alternative result. Subsequent validation, planning/failure, GQA, workspace,
paged and package-wiring PRs must close their assigned gaps and rerun affected correctness and speed campaigns.
