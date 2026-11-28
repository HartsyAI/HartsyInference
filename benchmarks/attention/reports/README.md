# PR 01 local evidence status

Subsequent semantic work is tracked separately in the [PR 02 progress report](pr02-semantic-progress.md).

The [RTX 4090 evidence review](pr01-4090-review.md) records the subsequently received remote report, unresolved
correctness/path/measurement gates and the harness contract correction. It does not clear either PR for merge.

PR 01 remains **incomplete and draft**. The [3060 report](pr01-local-3060.json) contains exact raw-artifact hashes,
test outcomes, known baseline failures, telemetry identity results, descriptive operation pilots and outstanding
exit work. The [Kitchen dependency lock](k0-windows-3060.lock.json) describes a Python-only K0 cohort; it does not
establish native Sage, native decode or native Sol support. The [runtime index](k0-runtime-index.json) records the
individual probes; unexecuted source cells remain unexecuted.

No paired performance claim, primary workload speed claim or complete parity claim is made. The initial false
Sol D64 invalid classification is preserved with its correction to negative admission and a versioned matrix
amendment requiring review. It is not an upstream-correctness exception.

The CPU run has two failures reproduced on pristine B0 and one skip. GPU resource skips and reported passes
whose own output says they lacked multiple GPUs are explicitly unvalidated. A successful test-process exit does
not turn those cases into regression evidence.

## Fixture API review correction

The C# reader rejects specialized API markers, unknown call options, auxiliary operands and unimplemented prepared
reuse before loading tensors. It cannot certify tree/step merge, Sol-Attn, Sage or decode API parity with its dense
adapter/oracle. Keep these cells unvalidated until both operation semantics and an independent oracle exist; any
earlier C# dense result for such a cell is invalid evidence, not an upstream K0 exception. This is a tool-only change;
production GPU test pins remain immutable. Record the corrected harness commit and binary hashes separately.

The complete Core CPU suite passes 187 cases on each of net8.0 and net10.0, including 27 fixture cases. Generated tree-merge,
step-merge and Sol fixtures are rejected by both command-line consumers before result files are written. A dense
F32 OpParity pilot passes its unreviewed quality threshold (relative L2 `8.06e-8`), which is not mandatory acceptance.
Both consumer builds pass on net8.0 and net10.0 without warnings. Raw CLI logs are under external
`artifacts/attention-pr01/api-rejection-review/`.

An initial broad test filter accidentally included `GpuBenchmark`: its memory-pressure preload case failed the
latency limit (77.9 seconds for matmul, 10,262 MB initially free). It was not an isolated GPU campaign and establishes
no speed result. Preserve that failure; the subsequent CPU run explicitly excludes GPU integration and benchmark
categories and does not replace it with a passing GPU claim.

Continue with the [4090 handoff](../vps-handoff.md). The user's subsequent authorization to start PR 02 is recorded
in the [implementation queue](../proposed-queue.md); it does not certify the incomplete PR 01 evidence or create Bfix.
Raw binaries, models, tensors, full locks and captures remain in the external artifact directory
identified by the report; retain it through the handoff and exit review.

The [PR 12 Qwen campaign](pr12-qwen-primary.md) on the RTX 4090 is recorded in this folder as its own report.

Other reports in this folder: the [consumer audit](consumer-audit.md) of attention computed outside the backend
entry points (plan items 14 to 17), and the [Vulkan tiled F32 A/B](vulkan-tiled-f32-ab.md), the descriptive RTX 3060
comparison behind the default of `numerics.vkTiledAttention`, with the [workloads that default moves](vulkan-tiled-default-workloads.md)
(a source audit). The
[Blackwell rental campaign](blackwell-rental-campaign.md) is the stage plan and spending ceiling for the final
hardware gate, written before any pod existed. The [RTX 5090 gate report](blackwell-5090-gate.md) records what ran
afterwards: the gate is partly met, Vulkan did not run on Blackwell hardware and stages 3 and 4 were not run.

[Block-sparse attention on an RTX 3060](block-sparse-3060.md) is the descriptive timing of `IBackend.BlockSparseAttention`
beside dense attention; it is outside the PR 01 evidence program.

The [internal duplicates audit](internal-duplicates.md) lists the internal attention code that plan item 18 could retire, with
the callers and the evidence for each.
