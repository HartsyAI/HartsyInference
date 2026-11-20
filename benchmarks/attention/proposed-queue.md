# Proposed work after the PR 01 exit review

The user subsequently authorized PR 02 alongside PR 01 review fixes, then PR 03 while the GPU agent tests the
pinned PR 01/02 revisions. These remain separate stacked drafts; parallel work does not waive their exit gates
or change the GPU agent's immutable pins. PR 01 still needs review of its
executed routes, frozen workloads, numerical definitions, per-cohort Kitchen cells, timing, baselines and exceptions.
Full Kitchen features and per-cell operation non-inferiority remain final CUDA **and** Vulkan gates. Infeasibility
requires reassessment; do not weaken a gate after inspecting results.

| PR | Responsibility and exit gate |
|---|---|
| 02 | Confirmed arithmetic defects: ALiBi/softcap order and fully excluded rows; reject an unsuitable cuDNN row if correct lowering is unavailable. Freeze Bfix. |
| 03 | Minimal Core planner/contract ownership and a reviewed common/optional cross-backend support matrix. |
| 04 | Shared validation, truthful numerical admission and public behavioral/configuration compatibility ledger, including every legacy numerical-permission adapter. |
| 05A | Common planning/execution and thin legacy adapters. |
| 05B | Explicit unsupported/recoverable/sticky failures and declared fallbacks; no silent GPU-to-host execution. No universal GPU kernel requirement. |
| 06 | Native GQA with explicit compatibility lowering; zero repetition in every call admitted to a native GQA route. |
| 07 | High-water scratch, immutable policy snapshots and workspace/graph/page leases, including eviction, replacement and concurrent replay. |
| 08A | Native causal/window prefill without eligible quadratic mask preparation. |
| 08B | Complete preparation/execution cost accounting and calibrated Auto. |
| 09 | Dense-or-paged views, page-table ownership, update synchronization and safe page reuse; replay never retains released pointers. |
| 10 | Native CUDA paged decode; zero prefix gather bytes for admitted native calls, with explicit gather lowering elsewhere. |
| 11 | Complete 3060/4090 backend milestone. Prepare Blackwell bundle; unavailable rentals do not block package migration. |
| 12 | Qwen vertical integration: repeatable complete primary request-latency improvement against Bfix, or stop before further migrations. Preserve all four secondary lanes. |
| 13 | Krea vertical integration: repeatable complete primary generation improvement against Bfix, or reassess. |
| K01 | Independent pinned rotation/representative-key quantization reference and compatible contracts. |
| K02, K03 | Separately reviewed CUDA and Vulkan generalized rotated INT8/prequantization projects. |
| K04 | BF16 decode, lengths, capacity tails and LSE; split substantial backend implementations into individual PRs. |
| K05 | Speculative tree and causal-step merge; separate backend review. |
| K06 | General Sol-Attn contract and independent block-tail reference. Preserve specialized VSA semantics. |
| K07, K08 | Separate CUDA and Vulkan general Sol-Attn implementations. |
| K09 queue | Individual optimization PRs closing measured Kitchen gaps without averaging away failing cells. |
| 14 | Video, World and viewers: source audit plus executed-path evidence and demonstrated ordinary-attention bypass migrations. |
| 15 | Audio/Voice audit, reusing existing test infrastructure. |
| 16 | Vision/ThreeD audit. |
| 17 | Every package/consumer audited; renew LLM/Diffusion checks after the K-series. Specialized algorithms are not rewritten to satisfy a quota. |
| 18 | Retire internal duplicates only; preserve public adapters and document behavioral compatibility changes. Complete required hardware and final parity/speed evidence. |

Move managed backend families only with their functional owner. Keep pure `git mv` commits separate from edits.
Preserve namespaces, assemblies, public adapters and kernel/PTX/SPIR-V locations. Use one public type per file
where it improves ownership rather than forcing a preparatory rename project.

## Evidence cadence and hardware

| Change | Required evidence |
|---|---|
| Correctness | Full semantics, affected real-generation correctness and performance smoke. |
| Mechanical | API/artifact/route/output identity and timing smoke. |
| Validation | Affected correctness and timing smoke; escalate if candidate selection changes. |
| Planner/admission | Affected route matrix and every workload whose candidates can change. |
| Optimization/kernel port | Full affected paired latency/memory campaign using the frozen statistical protocol. |
| Backend milestone | Complete frozen corpus on 3060 and 4090. |
| Final | Complete required corpus and Kitchen comparison on every required hardware cohort. |

Affected real-generation correctness is never reduced. Missing resources, mandatory skips, wrong paths and zero
executed cases mean unvalidated. B0 describes the whole series with semantic differences disclosed; equivalent
semantics/permissions against Bfix govern unchanged-quality optimization claims. Freeze new tolerances before
examining candidate errors.

The RTX 3060 covers limited VRAM and CUDA/Vulkan common paths; the 4090 is the main performance lane. Consumer
Blackwell is a blocking **final** CUDA/Vulkan gate. Prepare an exact rental campaign and spending quote before
requesting provisioning. SM75 and other unexercised architecture branches stay explicitly unverified; AMD/Intel
Vulkan are outside this program's hardware claims. No hardware lane may be pooled with another cohort.

Completion requires all admitted semantics and explicit unsupported behavior, required Kitchen features and
matched-quality operation gates, both primary improvements against Bfix, safe lifetimes, package wiring evidence,
public compatibility and complete hardware validation. PR 01's descriptive baselines cannot establish those claims.
