# Internal attention duplicates: audit for plan item 18

Plan item 18 ([proposed queue](../proposed-queue.md)) says: retire internal duplicates only, preserve public adapters, and
document behavioral compatibility changes. This report is the evidence for what could be retired. It is a source audit at
`main` `ab67d61f6` (2026-11-21). **No source was changed**; the only edits in the pull request are this file and one link.

The [compatibility ledger](../../../docs/ATTENTION_COMPATIBILITY.md) names the adapters that stay; the
[contract](../../../docs/ATTENTION_CONTRACT.md) and [support matrix](../../../docs/ATTENTION_SUPPORT.md) define the shared
path. Host-loop attention inside models is the [consumer audit](consumer-audit.md)'s subject (plan items 14 to 17) and is not
repeated here.

## Result

Almost nothing in the attention backends is a duplicate any more. Every private attention method in the CPU, CUDA and Vulkan
backends is reached from a row or a public entry, and every `.ptx` entry point and every `.spv` is loaded by some host code.
What is left falls into four groups:

| Verdict | Count | Items |
|---|---|---|
| remove: no caller | 4 | V1, D1, D2, D4 |
| remove after migrating N callers | 11 | V2, V3, V4, M1 to M8 |
| keep: public adapter | 5 | K5, C2, C5, V9, M9 |
| keep: not a duplicate | 20 | K1 to K4, C1, C3, C4, V5 to V8, V10, M10 to M12, D3, D5 to D8 |

The counts are of table rows (each row may name several symbols; the rows say how many). Two groups of "remove: no caller"
are not duplicates of the shared path at all: they are dead accessors and diagnostics (section D), listed because they are
internal, unreferenced, and the same deletion build proves them removable.

## How this was done, and what is and is not verified

**Ran (the part to trust most):**

- Every symbol marked "remove: no caller" was deleted in a scratch edit of this worktree, and every `*.csproj` in the tree
  (`git ls-files '*.csproj'`, 42 projects, including the ones the solution file omits and the tools and benchmarks) was built
  for `net10.0` with the repository's warnings-as-errors settings. Result: see [Deletion build](#deletion-build). The edit was
  reverted with `git checkout` and `git status` was clean before this file was added. No deletion is committed.
- A one-off analyzer build of the solution (`dotnet build HartsyInference.sln -f net10.0`) with `IDE0051`, `IDE0052`, `IDE0060`
  and `CA1823` raised to warnings in a temporary `.editorconfig` edit (reverted). It reports unused private members and unused
  parameters; for attention code it found no unused private method, only the unused parameters in section C.
- Three throwaway scripts over `git ls-files` (deleted afterwards): a comment-stripped identifier count over all 5,124 tracked
  `.cs` files (a member whose only occurrence is its declaration has no caller); a check that every `.spv`, `.ptx` and `.cu`
  is named by some `.cs` file; and a check that every `.entry` in each committed PTX is requested by `GetFunction`, literally
  or through a name built in a loop (`lm_fa_decode_lengths_{type}_d{dim}` and `int8v2_attn_br{n}_{variant}`, both read).
- A `git grep` of the named symbols in the 28 local `origin/*` refs updated since 2026-11-18, to see whether an open branch
  starts using a symbol that `main` does not use. The refs are the local copies of remote branches; I did not fetch.

**Read, not run:** every claim about numerics, about which row takes a call, and about what a helper does. I ran no test
(CPU or GPU) and no kernel: the GPU is in use by other sessions.

**Read by me versus by a sub-agent.** I read the CPU, Host and CUDA sites and the shared layer myself. Two read-only
sub-agents surveyed Vulkan (the row table, dispatchers, every shader and `.spv`) and the per-model wrappers. I re-read and
re-grepped the findings behind every row that is not "keep: not a duplicate" in sections V and M; the rows I took from them
without re-reading are marked "(sub-agent)".

**Rules applied.** Public members, members a model outside `src/HartsyInference.Core` calls, and anything whose numerics
differ from the shared path (compute type, summation order, mask rule) are not duplicates until shown otherwise; each row says
which of these applies. "Callers" were found with ripgrep over the whole repository (`src/`, `tests/`, `tools/`,
`benchmarks/`); comments and `cref`s are not callers. `internal` members are visible to the test assemblies named by
`InternalsVisibleTo` in the `.csproj` files (Cuda: Core.Tests, Cuda.Tests, Engine.Tests, LLM.Tests, GpuBenchmarks; Vulkan:
Vulkan.Tests; Diffusion: Diffusion.Tests, Cuda.Tests, Engine, Video), and the greps cover those.

## A. CPU, Core and Host

| # | File and symbol | Access | Duplicates / shared equivalent | Callers | Ledger | Change to remove | Verdict |
|---|---|---|---|---|---|---|---|
| K1 | `Core/Backends/Cpu/Kernels/AttentionKernels.cs` `ScaledDotProductAttention`, private `VectorizedDot`, `SoftmaxInPlace` | public static class | Looks like `AttentionHostReference.Run` (`Core/Backends/Attention/AttentionHostReference.cs`). | `CpuBackend.cs:141` only (`cpu.simd` row). No test, tool or benchmark names `AttentionKernels`. | not named | n/a | **keep: not a duplicate.** F32 only, parallel over heads and query chunks, AVX2/FMA lane-wise dot products in `float`; the host reference accumulates in `double` over strided operands of any dtype. Different summation order and accumulation type, and it is the CPU's fast route. Public. |
| K2 | `Core/Backends/Cpu/Kernels/FlashAttentionKernels.cs` `Run` | public static class | Same reference. | `CpuBackend.AttentionRows.cs:24` only (`cpu.flash`). | not named | n/a | **keep: not a duplicate.** Online softmax in `float`, F32 only; the reference is a two-pass `double` computation. Public. |
| K3 | `Core/Backends/Cpu/CpuBackend.cs:122` `CheckDenseKernelOperands` | private | Operand checks in `AttentionCallValidation`. | `CpuBackend.cs:116` (before the call is built) and `:140`. | The ledger says a legacy entry's own check runs first and keeps its historical exception classes. | Drop the check. | **keep: not a duplicate.** It throws `NotSupportedException` for a non-F32 operand and runs before `AttentionCall` is built (`HeadMajor` indexes rank 4). Removing it changes the exception class and order a caller sees, which needs a ledger entry. |
| K4 | `Core/Backends/Host/SoftmaxReference.cs` `Rows`, behind the `IBackend.SoftmaxRows` default (`IBackend.Softmax.cs:11`) | public static class | Softmax inside attention is `AttentionKernels.SoftmaxInPlace` / the reference's own loop. | `IBackend.Softmax.cs:11`; `LLM/Transformer/AttnRes/AttentionResidual.cs:74`; `SoftmaxRowsTests.cs`. | not named | n/a | **keep: not a duplicate.** A general row softmax, not an attention op. Its mask rule differs: a row of all `-inf` gives NaN here, zeros in `SoftmaxInPlace` and the reference (contract section 3). Public. |
| K5 | `Core/Backends/IBackend.Attention.cs:154,164,175,190` and `IBackend.Decode.cs:81`: the interface defaults `ScaledDotProductAttention`, both `ScaledDotProductAttentionTokenMajor`, `FlashAttention`, `FlashAttentionDev` | public interface | They already build an `AttentionCall` and enter `Attention` (or `FlashAttention`). | Dozens; see the consumer audit. | The first three are the ledger's first rows; `FlashAttention` and `FlashAttentionDev` are public entries the ledger discusses only as defaults. | n/a | **keep: public adapter.** |

## B. CUDA

| # | File and symbol | Access | Duplicates / shared equivalent | Callers | Ledger | Change to remove | Verdict |
|---|---|---|---|---|---|---|---|
| C1 | `Cuda/CudaBackend.cs` `ValidateScaledDotProductAttentionContract` (`:5829`), `ValidateTokenMajorAttentionContract` (`:5790`), `ValidateFlashAttentionContract` (`:8422`); `CudaBackend.AttentionRows.cs:638` `ValidateTokenMajorGqaAttentionContract` | internal static | Geometry, dtype and scale checks that `AttentionCallValidation` now makes. | One call each, from `ExecuteDenseAttention` (`:5421`), `ExecuteTokenMajorAttention` (`:5701`), `ExecuteTokenMajorGqaAttention` (`:5721`), `ExecuteFlashAttention` (`:7787`) and `ExecuteFlashDeviceAttention` (`:8063`). None from tests or tools. | The ledger: "a legacy entry's own contract check runs before the shared boundary and keeps its historical classes". | Delete them and let the shared boundary run first. | **keep: not a duplicate.** They run first and throw `ArgumentOutOfRangeException` (scale, softcap, `kvLen`) and `NotSupportedException` (dtype) where the shared boundary throws `ArgumentException`; they also hold the 32-bit launch bounds. Removing them is a behavior change that needs a ledger entry. The three SDPA ones repeat one another's text (same checks, different messages); merging them would change messages. |
| C2 | `Cuda/CudaKernels.cs:3277` `LaunchSoftmax`, `:3281` `LaunchSoftmaxF16`, private `LaunchSoftmaxImpl` (`:1519`), the `_softmaxF32`/`_softmaxF16` fields and `softmax_f32.ptx` / `softmax_f16.ptx` (loaded at construction, `:655`, `:801`) | public, in public class `CudaKernels` | `LaunchAttentionSoftmax` (`:3285`) and `attention_softmax.ptx`, which every materialized and tiled row uses. | `tests/HartsyInference.LLM.Tests/Parity/AttentionSemanticParityTests.cs:290-291`, which uses them as the legacy oracle for "finite rows stay bit-identical". Nothing in `src/`. | not named | Migrate that test to stored bits or a CPU oracle; delete the two launchers, the fields, the two modules and both `.ptx` files. | **keep: public adapter.** Public API of a public class, not named in the ledger, and still the oracle of a parity test; removal is an API break that needs a ledger entry first. The numerics also differ from the shared kernel: no bias, no zeroing of fully excluded rows, no NaN propagation. `softmax_f32/f16.ptx` are hand-written PTX with no `.cu` (see `Kernels/README.md`), so they cannot be rebuilt either. |
| C3 | `Cuda/CudaBackend.cs:7795` `FlashAttentionDecode` and `:8070` `FlashAttentionDeviceDecode` | private | The F.4 to F.7 and V.x branch ladders inside rows `flash.decode` and `flash.decode.device`. | Row bodies in `CudaBackend.AttentionRows.cs:587` and `:628`. | not named | n/a | **keep: not a duplicate.** Kernel selection the row table does not express (grouped decode, split-K, F16 K/V, graph-capturable splits); each branch is a distinct kernel with its own numerics. |
| C4 | `Cuda/CudaBackend.cs:5415,5698,5717,7781,8044` `ExecuteDenseAttention`, `ExecuteTokenMajorAttention`, `ExecuteTokenMajorGqaAttention`, `ExecuteFlashAttention`, `ExecuteFlashDeviceAttention` | private | The adapter plumbing between the public overloads and `RunAttentionRows`. | The public overloads and `Attention` (`CudaBackend.Attention` partial, `AttentionRows.cs:65`). | The ledger's "CUDA/Vulkan Boolean overloads". | n/a | **keep: not a duplicate.** They add the legacy contract check (C1), NVTX ranges and the op scope; they are the thin legacy adapters, not copies of the planner. |
| C5 | `CudaBackend.cs:5412` `ScaledDotProductAttention`, `:5694` and `:5712` `ScaledDotProductAttentionTokenMajor`, `:7777` `FlashAttention`, `:8039` `FlashAttentionDev` | public | Forward to the executors above. | Many. | **Named** (CUDA/Vulkan Boolean overloads: "keep until the public compatibility review explicitly deprecates an overload"). | n/a | **keep: public adapter.** |

PTX and kernels: every `.cu` under `Kernels/` has a `.ptx`; every committed `.ptx` is loaded by name (`Ptx("...")`); every `.entry` in the attention PTX files is requested by `GetFunction`. Entry names built in loops were read and match
all 6 `lm_fa_decode_lengths_*` and all 16 `int8v2_attn_br*` entries. The 26 `.ptx` files with no `.cu` are the hand-written ones the
README describes, and only `softmax_f32/f16` (C2) are attention-related. The only other `Launch*` methods with no caller in `src/`
are `LaunchAddScalarF16` and `LaunchMulMatVecQ8_0Q8_1PerRow`, which are not attention.

## C. Vulkan (sub-agent survey, re-read where marked)

| # | File and symbol | Access | Duplicates / shared equivalent | Callers | Ledger | Change to remove | Verdict |
|---|---|---|---|---|---|---|---|
| V1 | `Vulkan/VulkanBackend.FlashTiled.cs:30` `TokenMajorStrides` | internal static | The token-major strides row B.1 builds inline (`VulkanBackend.AttentionRows.cs:234-235`). | None. `rg TokenMajorStrides` over `*.cs` in the whole repository: one hit, the declaration. | not named | Delete the method. | **remove: no caller.** Deletion build: clean. |
| V2 | `VulkanBackend.FlashCm2.cs:277` `AttnStrides` and `:269` `WriteStrides`, against `VulkanBackend.FlashTiled.cs:169` `TiledStrides` and `:161` `WriteTiledStrides` | private vs internal | Two identical `(uint Row, uint Head, uint Batch)` record structs and two identical three-line writers. Re-read. | `AttnStrides`: `FlashCm2.cs:92,93,181,193,202,214`; `AttentionRows.cs:170,171,234,235`. `WriteStrides`: `FlashCm2.cs:193,221-224`. | not named | Replace `AttnStrides` and `WriteStrides` by the tiled pair: 2 files, about a dozen lines. Layout only, no numerics. | **remove after migrating 2 callers** (`VulkanBackend.FlashCm2.cs`, `VulkanBackend.AttentionRows.cs`). |
| V3 | `VulkanBackend.AttentionRows.cs:116` `TokenMajorCm2Admits` against the inline lambda of row A.1 (`:159-164`) | private | Same predicate text: F16 inputs or the legacy F16 assertion, not BF16, `CanUseFlashCm2(...)`. Re-read. | `TokenMajorCm2Admits`: `:228`. The lambda is its twin. | not named | Make row A.1 call the named predicate (rename it). | **remove after migrating 1 caller** (row A.1). |
| V4 | `VulkanBackend.cs:2199` parameter `allowF16` of `ScaledDotProductAttentionNaive` | private parameter | The body never reads it: the compute type follows the operands. | Passed by `AttentionRows.cs:208-209` (`c.Numerics.LegacyAssertedBounded`). Nothing else. | The Boolean is the ledger's legacy permission; this row does not act on it. | Drop the parameter and the argument. | **remove after migrating 1 caller** (the row at `AttentionRows.cs:208`). No behavior change: the permission still reaches the planner through `AttentionCall.Numerics`. Deletion build: clean. |
| V5 | `VulkanBackend.cs:2199` `ScaledDotProductAttentionNaive`, `:2343` `DispatchMaskAdd`, `:2370` `DispatchSoftmaxRows`, `softmax_{f32,f16}.spv`, `mask_add_{f32,f16}.spv` | private | A second dense SDPA beside the flash rows. | Row A.3 only (admits every call; the last row). `DispatchMaskAdd` and `DispatchSoftmaxRows` are reached only through it. (sub-agent) | not named | n/a | **keep: not a duplicate.** Materialized scores in the compute dtype with GEMM summation order; the only row for head dim above 256, F16/BF16 above 128, a grid past device limits, or `Heads % KvHeads != 0` (sub-agent). If row A.3 ever goes, the two dispatchers and four `.spv` go with it. |
| V6 | `VulkanBackend.cs:1978` `DispatchFlashAttention` against `VulkanBackend.FlashTiled.cs` `DispatchFlashTiled` | private / internal | Two F32 flash kernels. | `DispatchFlashAttention`: `AttentionRows.cs:182,348`. Tiled: row A.2t, off by default (`numerics.vkTiledAttention`). | not named | n/a | **keep: not a duplicate.** The tiled kernel has a different tile and summation order; whether it replaces the scalar one is what [the tiled A/B](vulkan-tiled-f32-ab.md) is for. |
| V7 | `DispatchFlashAttentionDev` (`VulkanBackend.cs:3931`), `DispatchFlashWide` (`FlashWide.cs:32`), `DispatchFlashDecode` / `DispatchFlashDecodeLengths`, `TryFlashCm2CausalPrefill` (`FlashCm2.cs:75`), `FlashCm2PrefillPolicy.cs` | private / internal | Each resembles another flash dispatcher. | One row each. (sub-agent) | not named | n/a | **keep: not a duplicate.** Each covers what its neighbour cannot: device positions, sinks and head dim up to 1024, a key length per sequence with an LSE, F16 cooperative-matrix prefill with its runtime declines. Shader numerics were not read. |
| V8 | `VulkanBackend.Int8AttentionU8.cs` `Int8AttentionU8`, `Int8AttentionU8Tiles`, `Int8AttentionU8Drain`; `Int8AttentionCentering.cs`; `Int8AttentionQueries.cs`; the seven `int8_attn_*.spv` | public, not on `IBackend` | Nothing: no row takes them. | Tests only (`Int8AttentionU8VulkanTests` and four siblings). No `src/` caller. | not named | n/a | **keep: not a duplicate.** Building blocks of the Vulkan INT8 path (plan item K03) with no shared equivalent yet; unwired, not duplicated. (sub-agent) |
| V9 | `VulkanBackend.cs:1962` `ScaledDotProductAttention`, `VulkanBackend.FlashCm2.cs:33,39` token-major overloads, `VulkanBackend.cs:2122` `FlashAttention`, `:3913` `FlashAttentionDev` | public | Forward to the rows. | Many. | **Named** (CUDA/Vulkan Boolean overloads). | n/a | **keep: public adapter.** |
| V10 | `VulkanBackend.cs:2185` `ExpandKeyOnlyMask`, with the same "`RowStride == 0`, then expand" test at `FlashCm2.cs:169` (`WithRowsForCm2`) and `VulkanBackend.cs:2234` | private static | The key-only mask rule, repeated at two call sites around one shared helper. | Both. | not named | Fold the test into `WithRowsForCm2`. | **keep: not a duplicate.** The expansion is already one helper; only a three-line condition is repeated, which is a tidy-up, not a retirement. |

Shaders: 129 `*.comp.glsl` sources and 204 committed `.spv`. Every source is built by `Shaders/build.sh`; every `.spv` has a
source; every `.spv` is loaded by a `GetKernel` name, a `_f32`/`_f16` suffix (`VulkanBackend.cs:300`) or the tiled name table.
No orphan in either direction. Not checked: that each committed `.spv` matches its source (I did not run `build.sh`), and the
contents of any shader. The `<summary>` of `FlashMaxHeadDim` (`VulkanBackend.cs:1959`) is stale: it says a head dim above 128 falls
back to the materialized path, but dense F32 also has the tiled row up to head dim 256 when `numerics.vkTiledAttention` is `auto` (default `off` at `ab67d61f6`).

## D. Dead internal members that are not duplicates

These do not duplicate the shared path. They have no caller and are listed because they are attention-adjacent and internal;
removing them is a cleanup, not a retirement, and is outside item 18's rule unless the owner wants it done at the same time.

| # | File and symbol | Access | What it is | Callers (search) | Ledger | Verdict |
|---|---|---|---|---|---|---|
| D1 | `Diffusion/Models/Denoisers/DiTBlocks/WanVideoBlock.cs:33-37` `SelfAttentionStage`, `CrossAttentionStage`, `ImageAttentionStage`, `SelfAttentionF16Stage`, `CrossAttentionF16Stage` (5 properties) | internal | Accessors returning the private `_selfAttention` etc. Test seams: the tests that used them were deleted in `e9474bb06`. | `rg '\b(SelfAttentionStage\|CrossAttentionStage\|ImageAttentionStage\|SelfAttentionF16Stage\|CrossAttentionF16Stage)\b'` over every file type in the tree: 5 hits, the declarations. No `GetProperty` reflection names them. Not used on the 28 recent local remote refs. | not named | **remove: no caller.** Deletion build: clean. |
| D2 | `AttentionStage` in `Diffusion/Models/Denoisers/SeedVr2DitBlock.cs:60`, `MiniMaxH3Transformer.cs:51`, `DiTBlocks/MatrixGame3ActionModule.cs:36`, `DiTBlocks/OasisSpatioTemporalBlock.cs:53` (4 properties) | internal | Same shape. | `rg '\bAttentionStage\b'`: the four declarations, the fifth below, and nothing else. | not named | **remove: no caller.** Deletion build: clean. |
| D3 | `DiTBlocks/WanS2VAudioInjector.cs:26` `AttentionStage` | internal | Same shape. | None on `main`, but test code on `origin/codex/attention-integration-4090` and `origin/codex/attention-wan-s2v-injector-batch` calls `injector.AttentionStage` (the former forward is its oracle). | not named | **keep: not a duplicate.** Removing it would break an open draft; revisit when that merges. |
| D4 | `Cuda/CudaBackend.cs:67` `CudnnSdpaDiagnostics`, `:108` `CudnnSdpaSessionGeneration`, `:111` `CudnnSdpaDisposedSessionCount` (3 properties) | internal | cuDNN SDPA session health and "test diagnostic" counters. | `rg` for each name over the whole tree, all file types: only the declaration. Not used on the 28 recent local remote refs. The counters they read (`_cudnnSdpaSessionGeneration`, `_cudnnSdpaDisposedSessionCount`) are still incremented, so removing the properties leaves two write-only fields; that is a follow-on cleanup. | not named | **remove: no caller.** Deletion build: clean. |
| D5 | `CudaBackend.cs:48,63,89,116` `CudnnSdpaRefusedShapes`, `CudnnSdpaDimDiagnostics`, `CudnnSdpaEngaged`, `SageAttentionEngaged` | public | cuDNN and Sage health and engagement reports, which `AttentionEvidence` (a per-call route record) does not give. | None anywhere in the tree. | not named | **keep: not a duplicate.** Public and unread, so removal is an API break; an owner decision. |
| D6 | `Vulkan/VulkanBackend.FlashCm2.cs` `FlashCm2PrefillDispatches` | public | A dispatch counter whose comment says "tests and profiling". | None. (sub-agent) | not named | **keep: not a duplicate.** Public and unread. |
| D7 | `Cuda/CudaBackend.Int8AttentionV2.TestHooks.cs` `RunInt8AttentionV2ForTest`, `Int8V2*ForTest` | internal | Test hooks. | None on `main`, but `origin/hartsy/attention-k02-cuda-int8-rotation-b5a1bb` (an active session) calls `RunInt8AttentionV2ForTest` in new tests. | not named | **keep: not a duplicate.** In use by in-flight work. |
| D8 | `Core/Backends/Attention/Int8AttentionExp2.cs` `Int8AttentionExp2` and `DecodeLengthsReference.cs` `DecodeLengthsReference` | public | References, not row bodies. | Tests only (`DecodeLengthsReferenceTests`, `AttentionSequenceLengthsCallTests` and others in `tests/HartsyInference.Core.Tests/Backends/`); none in `src/`. | not named | **keep: not a duplicate.** Oracles with no shared equivalent. |

## E. Per-model attention helpers (sub-agent survey; re-read where marked)

The shared layer is `Core/Layers/Attention/AttentionLayer.cs`. `AttendHeadMajorOperands` (`:164`) builds
`AttentionCall.FromScaledDotProductAttention(..., allowF16: false) with { Numerics = _config.Numerics }` and calls
`backend.Attention`; a layer with `AllowF16 = false` has `Numerics == FromLegacy(false)`, so it builds the same call the legacy
SDPA entry builds, and `MergeHeads` (`:362`) is one `Permute0213` that disposes its input. Rows M6 to M8 rest on that.

| # | File and symbol | Access | Duplicates / shared equivalent | Callers | Numerics | Change to remove | Verdict |
|---|---|---|---|---|---|---|---|
| M1 | `Diffusion/Models/TextEncoders/MiniMaxH3TextEncoder.cs:403` `CausalGqaAttention` | internal static | A null check and a divisibility check, then `backend.FlashAttention(..., causal: true, ...)`; shared form `AttentionCall.FromFlashAttention`. Re-read. | `MiniMaxH3TextEncoder.cs:635`. | Identical call. | Inline the `FlashAttention` call; the helper's own error message goes. | **remove after migrating 1 caller** (`MiniMaxH3TextEncoder.cs:635`). |
| M2 | `Diffusion/Models/Denoisers/DiTBlocks/BooguImageDoubleBlock.cs:377` `RepeatKv` | private | Allocates and calls `StridedCopies.RepeatInterleave` (`Core/Backends/Layout/StridedCopies.cs:31`). Re-read. | `BooguImageDoubleBlock.cs:357,358`. | Same copy. | Inline at the two sites. | **remove after migrating 2 callers** (same file). |
| M3 | `DiTBlocks/Lumina2Block.cs:461,479`, `DiTBlocks/Lumina2ContextRefinerBlock.cs:227,245`, `TextEncoders/T5Block.cs:268,290`: private `ReshapeToMultiHead` / `ReshapeFromMultiHead` | private static | Copies of `DiTUtils.ReshapeToMultiHead` / `ReshapeFromMultiHead` (`DiTBlocks/DiTUtils.cs:198,218`), a Diffusion helper, **not** the shared attention path. Re-read: the two Lumina copies are byte-identical to each other and index exactly as `DiTUtils` does; `T5Block` copies element by element. | Lumina2Block `:259-261,302`; Lumina2ContextRefinerBlock `:128-130,167`; T5Block `:124-126,145`. | Pure F32 copies: bit-identical output. | Call `DiTUtils.` instead; delete six methods. | **remove after migrating 3 callers** (Lumina2Block, Lumina2ContextRefinerBlock, T5Block). Note that `DiTUtils` is itself a host loop that `Permute0213` could replace; that is a separate matter. |
| M4 | `Audio/Models/PocketTts/PocketTtsMimiDecoder.cs:202` `MergeHeads` | private static | `AttentionLayer.MergeHeads` (`AttentionLayer.cs:362`); the same file already calls `_attention.AttendHeadMajorOperands` (`:156`). Re-read. | `PocketTtsMimiDecoder.cs:159`. | Pure copy. The layer's version permutes on the device and disposes its input, which the caller disposes itself at `:160`. | Call the layer's `MergeHeads`; drop the dispose. | **remove after migrating 1 caller** (`:159`). |
| M5 | `Audio/Models/ControlFoley/ControlFoleyVitOps.cs:70` `MergeHeads` | internal static | Same equivalent; `SelfAttention` (`:91`) already uses `AttentionLayer.Shared(...).AttendHeadMajorOperands(..., OperandOwnership.None)`. Re-read. | `ControlFoleyVitOps.cs:98`, `ControlFoleySynchformer.cs:345`. | Pure copy, F32. The caller keeps `ctx` alive and disposes it with `using`, so ownership changes. | Use the layer's `MergeHeads`; fix the two `using`s. | **remove after migrating 2 callers** (the two above). |
| M6 | `ThreeD/Models/Idol/IdolAttention.cs:21` `Attend` | internal static | `AttentionLayer.AttendHeadMajorOperands` plus `MergeHeads`, after the helper's own permutes. Re-read. | `IdolBlock.cs:153`, `IdolDecoderAttention.cs:70`. | The helper passes `allowF16: true`. Equivalent only through `new AttentionLayer(new AttentionLayerConfig { ..., AllowF16 = true })`, which builds `FromLegacy(true)`; `AttentionLayer.Shared` cannot express it. | Migrate both callers to a configured layer. | **remove after migrating 2 callers** (`IdolBlock.cs:153`, `IdolDecoderAttention.cs:70`). Needs the route evidence the ledger's "Removal evidence" section requires, and the plan item 16 audit owns the move. |
| M7 | `ThreeD/Models/TrellisV2/Trellis2FlowBlock.cs:265` `Attend` | private | SDPA with a null mask, then `Permute0213` back: `AttentionLayer.Shared(heads, headDim, scale, DType.F32).AttendHeadMajorOperands` plus `MergeHeads`. Re-read. | `Trellis2FlowBlock.cs:177,203`. | `allowF16` is false on both sides, so the calls are equal by reading. The layer also checks that K and V shapes match, and malformed input reaches the planner by a different path, so exception classes may differ (not tested). | Replace the body with the layer calls. | **remove after migrating 2 callers** (`:177`, `:203`). Needs route evidence. |
| M8 | `ThreeD/Models/CoLa3D/CoLa3DBlock.cs:169` `Attend` | private | Same pipeline. Re-read. | `CoLa3DBlock.cs:84,112,146`. | As M7. | As M7. | **remove after migrating 3 callers** (`:84`, `:112`, `:146`). Needs route evidence. |
| M9 | `Vision/Detection/RtDetrAttention.cs:21` `MultiHead` | public static class | A thin wrapper over `AttentionLayer.Shared(...).AttendHeadMajorOperands`. (sub-agent) | Five: `GroundingDinoEncoder.cs:220,221`, `GroundingDinoMultiheadAttention.cs:34`, `RtDetrDecoder.cs:230`, `RtDetrHybridEncoder.cs:288`, `Sam3MultiheadAttention.cs:66`. | Same call. | n/a | **keep: public adapter.** Public in a package; not named in the ledger. |
| M10 | `Audio/Models/Dia/DiaHeads.cs:53` `RepeatKv`, `FlatToHeads`, `HeadsToFlat`, `SliceLeading` | public static class | Host copies of `StridedCopies.RepeatInterleave` and `Permute0213`. (sub-agent) | `DiaAttention.cs:193,194` for `RepeatKv`; the others also in `NeuCodecDecoder.cs`, `DemucsCrossTransformer.cs`, `Text2Semantic`. | Pure copies. | n/a | **keep: not a duplicate.** Public, with callers in other models; consolidating them is model work. |
| M11 | Causal-mask tensor builders: `ClipTextEncoder.cs:417`, `Sam3TextEncoder.cs:136`, `WhisperOps.cs:90`, `GptBackbone.cs:157`, `OasisDit.cs:310`, `MiniMaxH3AudioVaeEncoder.cs:421`, `TextEncoderTensorHelpers.cs:140`, `ZonosBackbone.cs:114` | private or internal | No shared tensor builder exists: `AttentionMask` is a descriptor (`Causal`, `WindowLeft`), not a bias. (sub-agent) | Local to each model. | The sentinels differ (`-inf`, `-1e9`, `-1e30`) and so do the shapes; moving to the structural `Causal` mask selects different kernels. | n/a | **keep: not a duplicate** of any shared helper. A new shared helper would be needed first. |
| M12 | The 45 direct `ScaledDotProductAttention(` / `FlashAttention(` / `FlashAttentionDev(` calls under `src/` outside `Backends/` that add RoPE, QK norm, masks, ALiBi, GQA repeat or windowing, and small wrapper classes such as `HeadGatedAttention` and `HunyuanPaintAttention`; `MusicGenBlock.cs:115` calls `FlashAttentionDev` on purpose to keep its own rows on Vulkan | various | Inline copies of what `AttentionLayer` does around the call. (sub-agent) | n/a | Moving them onto the layer's token-major or GQA routes could change summation order; not tested. | n/a | **keep: not a duplicate.** These are model migrations (plan items 14 to 17), not item 18. |

## Closing

### The "remove: no caller" items, which could be done first

| Item | Symbol | Why it is safe |
|---|---|---|
| V1 | `VulkanBackend.TokenMajorStrides` (`VulkanBackend.FlashTiled.cs:30`) | One hit for the name in the whole repository: its declaration. A duplicate of strides the row builds inline. |
| D1 | `WanVideoBlock` `SelfAttentionStage`, `CrossAttentionStage`, `ImageAttentionStage`, `SelfAttentionF16Stage`, `CrossAttentionF16Stage` | Five declarations and no other hit in any file type. Dead test seams. |
| D2 | `AttentionStage` on `SeedVr2DitBlock`, `MiniMaxH3Transformer`, `MatrixGame3ActionModule`, `OasisSpatioTemporalBlock` | Same. (`WanS2VAudioInjector.AttentionStage` is **not** in this list: open drafts use it.) |
| D4 | `CudaBackend` `CudnnSdpaDiagnostics`, `CudnnSdpaSessionGeneration`, `CudnnSdpaDisposedSessionCount` | One hit each, the declaration. Leaves two write-only counter fields to delete after. |

V1 is the only one that is a duplicate. D1, D2 and D4 are dead members; none of the four changes behavior a user can observe,
so none needs a ledger entry. Everything in this table compiled away cleanly in the deletion build below.

### Deletion build

The scratch edit deleted V1, D1, D2 and D4 and also dropped the `allowF16` parameter of V4 (and its argument at
`AttentionRows.cs:208`). `dotnet build <project> -f net10.0` then ran for each of the 42 projects from
`git ls-files '*.csproj'`: **41 built with 0 errors.** The one failure,
`tests/python-reference/trellis2_stage_parity/csharp/Capture/Capture.csproj`, is `CS1529` at `Program.cs(29,1)`, a `using`
placed after statements; no edited symbol is involved, but I did not build that project without the edit to prove it was
already failing. The projects without a test assembly of their own (`HartsyInference.Cuda.Tests`, `HartsyInference.Vulkan.Tests`)
are not built by this, which is why the greps, not the build, cover them. The edit was reverted with `git checkout -- src/`.

### Public adapters confirmed kept

Read in the source, each forwards through the shared adapter and is named in the ledger or is a public entry the ledger
covers: the `IBackend` defaults (K5); the CUDA and Vulkan Boolean overloads (C5, V9); `AttentionCall.FromScaledDotProductAttention`
and `FromTokenMajor` (`AttentionCall.cs:155,166,177`, the implementation of the overloads); `AttentionLayerConfig.AllowF16`
(`AttentionLayerConfig.cs:110`) and `JointAttentionConfig.AllowF16` / `AllowF16Batched` (`JointAttention.cs:32,35`). Also kept for
being public although the ledger does not name them: `CudaKernels.LaunchSoftmax` and `LaunchSoftmaxF16` (C2),
`RtDetrAttention.MultiHead` (M9), `DiaHeads` (M10) and the public diagnostics in D5 and D6.

### Removals that would change behavior a user can observe (each needs an entry in `docs/ATTENTION_COMPATIBILITY.md`)

- **C1** (the four CUDA legacy contract checks): the exception class and message a malformed legacy call raises.
- **K3** (the CPU operand check): the same, on the CPU.
- **C2** (`LaunchSoftmax*` and the two `.ptx` files): removes public API, and the oracle of a parity test.
- **D5, D6** (public diagnostics): removes public API.
- **M6 to M8** (Idol, Trellis, CoLa3D): route selection and exception classes for malformed input may differ on CUDA and
  Vulkan; the ledger's "Removal evidence" section asks for the caller, configuration, explicit policy, route evidence and a
  real-generation result first.
- **M4 and M5**: the output is a copy and does not change; on a GPU backend the host loop's device-to-host sync goes away, which
  is a timing change only.

V1, V2, V3, V4, D1, D2, D4 and M1 to M3 do not change observable behavior.

### Leads, not findings

- `RunInt8AttentionV2(in c, bool f16Pv)` (`CudaBackend.Int8AttentionV2.cs:103`) ignores `f16Pv`; the row pair `int8.v2` and
  `int8.v2.f16pv` (`CudaBackend.AttentionRows.cs:145-148`) both reach it, and the PV format comes from `CurrentInt8V2Pv()`. I did
  not trace why. It is the K02 session's area, so I left it alone.
- The analyzer also reports unused parameters: `AttentionHostReference.cs:191` `BiasView.Of(..., call)`, and
  `VulkanBackend.FlashCm2.cs:164` `CanUseFlashCm2(..., sq, skv, ...)` (`sq` and `skv`; the Vulkan agent noted `TryFlashCm2CausalPrefill`
  repeats the same minimum-rows rule elsewhere). Not duplicates; trivial.
- `Krea2Attention.Forward` takes an unused `seqLen` (`Diffusion/Models/Denoisers/DiTBlocks/Krea2Attention.cs:75`).

### What I did not examine

- Any shader's contents; whether each committed `.spv` or `.ptx` matches its source (I did not run `build.sh` or an NVRTC
  build).
- Numerical equality of any "remove after migrating" pair beyond reading the code: no test was run, CPU or GPU, and no
  real generation.
- `Core` internals beyond `AttentionLayer.cs`, `IBackend.Attention.cs`, `AttentionMask.cs`, `AttentionCall.cs` (the
  `From*` builders), the CPU files and `AttentionHostReference.cs`: the planner, `AttentionCallValidation`,
  `AttentionFlashLowering`, `JointAttention` and `AttentionLayer.Kv.cs` were not read for duplication.
- Specialized backend entries (`SolAttention`, `DecodeTreeMerge`, `NeighborhoodAttention`, `SparseLatentAttention`, the video
  sparse session) beyond confirming they are separate operations and not rows.
- Consumers of `JointAttention` (Flux, SD3 and the like) and `AttentionLayer` wrappers such as `Kandinsky5BlockOps.Attention`,
  for pure forwarders.
- Open pull requests and other worktrees beyond the symbol check on 28 local remote refs; a branch older than 2026-11-18 may
  use a symbol I call dead. The local remote refs were not refreshed.
- The scripts' limits: the identifier count does not look inside `tools/` as a caller class of its own (it is a repository-wide
  count), and a common name such as `Attend` or `MergeHeads` cannot be judged by count alone, so those rows were read by hand;
  the artifact check counts any quoted name in a `.cs` file as a load, so a name quoted only in a test or a comment would pass
  it (the entry check does not have this limit for the entries it names).
- Assemblies named in `InternalsVisibleTo` that have no project in this repository (`HartsyInference.Cuda.Tests`,
  `HartsyInference.Vulkan.Tests`, the Audio phonemizer and ModelAssets tokenizer test assemblies): any caller in them is
  invisible to the deletion build; the greps cover every `.cs` file in the tree, including `tests/HartsyInference.Cuda.Tests/`.

## TL;DR

The attention backends have been cleaned up enough that there is almost nothing internal left to retire: one stride helper
(V1), plus three groups of dead accessors and diagnostics (D1, D2, D4) that compile away cleanly. Eleven more items are real
duplicates but have callers to migrate first; three of them (Idol, Trellis, CoLa3D) need route evidence. The two obvious
"old path" candidates, the CUDA legacy validators and the CUDA legacy softmax launchers, are deliberately kept: they carry
historical exception classes and a public API.
