# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N` (the scheme moved
up from `1.0.0-alpha.N`; entries below that pre-date the change and keep their original numbers). The single
source of truth is `<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props` — see
[`docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md`](docs/Checklists/ROADMAP.md) for what a
stable release will require. Dates are UTC.

## alpha.120

- **`QkvSplitNorm` runs on Vulkan**, the one op on the Flux/DiT path whose host default was a TRUE fallback rather
  than composition. It reads `DataPointer` on six tensors, so every call meant a device-to-host sync, a scalar loop
  over every token and head, and an upload of the results. Flux runs 19 double plus 38 single blocks per forward,
  each calling it once per step, so the round-trip was paid 57 times a step.
- Chosen by measurement rather than by working down the list: of the eight Phase-3 ops the Flux path calls, seven
  have defaults that compose other GPU ops — correct, just extra dispatches — and this was the only one that left
  the device.
- **The cross-subgroup fold is not optional here even though the reduction is only headDim wide.** `subgroupAdd`
  reduces within a subgroup, and subgroup width is hardware: 32 on NVIDIA, 64 on AMD, as low as 8 on Intel. A
  workgroup spanning more than one would have normalized by the wrong denominator and still produced plausible
  output — the failure worth spending shared memory to avoid, and the one that would have shown up first on the
  AMD card this work is for.
- Verified against the CPU reference at four head geometries chosen to straddle every plausible subgroup width,
  including a 256-wide head that spans several. Max absolute error 4.8e-7 on q and k; `v` is a straight copy and is
  exact. And on a real generation — Flux.1-dev fp8, 512x512, 4 steps, seed 42 — the decoded image is byte-identical
  to the same run through the host fallback.

## alpha.119

- **`LayerNormNoAffine` runs on Vulkan.** It had no override there, so every call fell through to `IBackend`'s host
  default — which refuses anything but F32 outright. That is what a Flux generation on an AMD card hit: the DiT
  runs its block activations in F16 by default (`numerics.ditF16`), so the pre-modulation norm handed the default an
  F16 tensor and the generation failed with "LayerNormNoAffine default fallback only supports F32". Every DiT
  normalizes before modulating, so the op is on the hot path of the whole family, not a corner of it.
- The shader is deliberately its own kernel rather than `layernorm` with an identity weight. The identity would
  cost a per-element multiply-add and, more to the point, two device buffers that do not exist at the call site —
  every caller would have to allocate and fill them per call.
- `IBackend.LayerNormNoAffineReference` joins the other reference statics, so the host default and any backend
  falling back share one implementation. An override cannot reach the default through `((IBackend)this)`: that
  re-enters the override and recurses until the stack ends.
- Verified against the CPU reference at four shapes, including a row count that is not a multiple of the workgroup
  and a dim that is not a multiple of the subgroup, since the cross-subgroup fold is where a norm like this goes
  wrong on small-subgroup hardware. Max absolute error 1.4e-6 in F32, 4.9e-4 in F16 — the latter being F16's own
  precision rather than a disagreement.

## alpha.118

- **A quantized video build keeps the semantics of the build it came from.** Video planning resolves by exact
  file hash, so a checkpoint quantized here is a stranger to it — an H3-class output planned as an unknown base
  and the task, acceleration and step count its source declared were simply gone. `QuantizationService` writes
  `<output>.hartsy-video-profile.json` when the SOURCE's hash resolves to a known artifact, and writes nothing
  when it does not: inventing provenance for a file nobody has verified is worse than leaving it unknown.
- **Quantizing a large block-quantized source now refuses instead of being OOM-killed.** Every tensor is widened
  to F32 before the writer sees it, so requantizing a 13 GB Q4_K checkpoint wants about 74 GiB — and the process
  died with no message, no partial file and nothing to read. It measures that from the real element counts and
  says so up front, with the numbers.

## alpha.117

- **Twenty-seven Vulkan ops were dispatching outside an op scope.** The scope is what suppresses the batched
  auto-flush, so without one a flush could land between two dispatches of the same op and free the transients the
  later dispatches still read — the per-slice loops are exactly that shape. `Silu`, `Add`, `RmsNorm`, `LayerNorm`,
  `Concat`, `Split`, `BatchedMatMul` and twenty more were missing it, including the internal entry points the
  coopmat benchmarks call directly. They were not failing yet because the flush threshold is eight dispatches.
- **The dispatch path checks three correspondences that were maintained by hand and verified by nothing.** The
  storage-buffer count against the count the kernel was built for, the push-constant size against the range every
  layout reserves, and that an op scope is open. Each failed silently when wrong: a short buffer list leaves the
  remaining bindings pointing at whatever bound them last, an over-long push block is truncated at the layout, and
  the flush hazard above. Turning the check on is what found the twenty-seven.
- `PushConstants` builds a push block by appending, so byte offsets are not written by hand. An op used to
  `stackalloc byte[9 * 4]` and write nine offsets that had to agree with a GLSL struct in another file and with
  each other; inserting a field meant renumbering every line below it, and an error does not fail — the shader
  reads a plausible number from the wrong place. It is a MUTATING ref struct deliberately: chaining by value would
  copy it, and every field would land at offset zero.
- `VulkanDescriptorManager.PushConstantRangeBytes` names the 128-byte floor that was a bare literal in the layout
  and an implicit assumption in every op's `stackalloc`, with nothing connecting the two.
- A source lint enforces both rules without a GPU: no override reaching its own interface default (which recurses
  to a stack overflow), and every dispatching op opening a scope. The runtime guard only fires on a path some test
  runs, and Vulkan's least-covered ops are the ones most likely to be written next.

## alpha.116

- **`(word:1.5)` works on Wan.** umT5 keeps token weights, so Wan is a ComfyBlend family: the prompt is encoded at
  face value and the output blended toward the empty-prompt encode, `z = (z − z_empty)·w + z_empty`. The baseline
  is encoded in the same batch as the prompt and the negative, because it has to share their padding and layer
  selection exactly. The negative is weighted too, which is what ComfyUI does.
- The emphasis grammar comes off the text whether or not it does anything. Declaring a weighting mode is what
  stops the service stripping `(word:N)` upstream, so a weight of exactly 1.0 would otherwise have reached umT5
  as literal parens and digits — and differed from the same prompt written plainly.

## alpha.115

- **The engine asks a backend what it can do instead of what class it is.** Every `is CudaBackend` outside the CUDA
  package is gone. They were not stylistic: each one silently denied a capability to any backend that was not CUDA,
  so a second backend inherited the restriction whether or not it applied.
- `dequantizeToF32` is `!Capabilities.SupportsQuantized` in both `TextService` and `ModelManager`. Equivalent today
  — CUDA publishes true, Vulkan and CPU false — but it means Vulkan stops paying an F32 expansion the moment it
  publishes the capability, rather than forever because it is not CUDA.
- `PreloadWeights` is called unconditionally. It is a no-op on a backend with no device memory, and a backend that
  HAS device memory wants its weights resident; gating on the class meant Vulkan re-uploaded every weight over PCIe
  on every op. Same for `FreeAllDeviceMemory` on unload, which simply never ran on Vulkan and left VRAM held.
- The Ideogram 4 and Boogu VRAM preflights use `GetVramInfo()` and `StreamingCache`, so they now apply on any
  backend that reports memory rather than being skipped entirely off CUDA — which is what made them silent there.
- **Two of these change behaviour rather than just spelling, and deliberately.** SeedVR2's BF16 VAE activations
  were keyed on "is CUDA" and are now keyed on `SupportsBF16`, which on CUDA is compute capability 8.0 or newer —
  so a pre-Ampere card gets F32 activations where it previously got BF16 it has no hardware for. MiniMax-Music3's
  half-precision KV moves from "is CUDA" to `SupportsF16` on the same reasoning.
- Left alone on purpose: `ValidateShardDevices` still requires a CUDA device for LLM layer-split. The real
  requirement is a backend that computes on quantized tensors, and Vulkan does not yet — relaxing the check to any
  GPU kind would admit it to a path that would fail further in. It changes when Vulkan publishes the capability.

## alpha.114

- **`POST /admin/models/quantize`** exposes offline quantization over the API, with the same formats and presets
  the CLI takes so a caller does not have to learn two vocabularies for one operation.
- It is synchronous on purpose. A caller that gets a 200 has a finished file on disk; a multi-GB write that
  reported success before it was durable is the sort of thing nobody notices until the file is loaded.
- Validation happens before the filesystem is touched, so a bad format or preset is a 400 rather than a
  partially-written file.

## alpha.113

- **The op scope every GPU backend needs is written once.** `GpuBackendBase` owns it: the point at which a backend
  runs the device cleanup a finalizer could not, and reclaims what the previous op displaced. Both backends had
  written it separately, and it is easy to get subtly wrong because it is defined by what has ALREADY happened
  rather than by what the op is about to do. Nesting is handled there too — an op built out of other ops must not
  sweep in the middle of itself, since the buffers it would reclaim are the ones its own later dispatches read.
- The `IBackend` members that are simply the residency cache live there too: the D2H counters, preload, free,
  pin/unpin and bulk offload. Expanding low-rank adjuncts before a preload or free is part of that — both backends
  did it, because a weight carrying an adjunct is read as its factors, so preloading the weight alone leaves the
  factors to miss one at a time on the hot path.
- **Vulkan reports VRAM.** Every memory decision on that backend logged "no VRAM report" and fell back to fixed
  budgets, because `GetVramInfo` did not exist there. Total now comes from the device's DEVICE_LOCAL heaps, which
  is exact; free is total minus what this backend's allocator holds, which underestimates what the card has left
  because another process is invisible to it. Reported anyway: the alternative was no number at all. A live figure
  from `VK_EXT_memory_budget` is the replacement, and the capability is already detected with nothing querying it.
- Profiling stays a hook rather than a shared object. CUDA's is NVTX ranges and Vulkan's is `VulkanProfiler`;
  unifying those is its own change, and all the scope needs is whether to time an op and somewhere to hand the
  answer.
- **CUDA does not derive from the base yet, and that is deliberate rather than unfinished-by-accident.** Converting
  its 181 op entries to the shared scope makes three `CudaGraphTests` fail at teardown with
  `CUDA_ERROR_INVALID_VALUE`, freeing an activation whose address the driver already reclaimed with a directly
  constructed `CudaGraph`. Bisected: the base plus the Vulkan adoption is clean, disposal routing is not the cause,
  and forcing the old always-enter behaviour on nested scopes does not fix it. Landing the half that is proven
  beats landing a regression next to it.

- **A LoRA can be merged into a convolution weight.** SD1.5 and SDXL UNets are mostly convolution, so a LoCon or
  LyCORIS adapter targeting conv layers had its deltas silently skipped — counted as unmatched and stepped over.
- The delta arrives flattened to `[out, in*kh*kw]` while the weight is `[out, in, kh, kw]`. Row-major those are
  the same bytes in the same order, so the previous rank-2-only gate was conservative rather than necessary; the
  add only needed the two shapes to agree.
- **DoRA on a convolution is refused by name.** Its magnitude vector normalizes by a row norm of the weight as a
  matrix, and a convolution has no such matrix until it is flattened — which axis the vector then describes is the
  file's choice, not ours.

## alpha.112

- **`hartsy quantize --format` writes ComfyUI's two safetensors quant shapes as well as GGUF.** `fp8-scaled`
  stores each eligible weight as F8E4M3 beside the scalar it was divided by; `int8-convrot` stores it as I8 beside
  a `[rows,1]` row scale and a `.comfy_quant` descriptor.
- Two details that exist only because interoperating with published files is the point. The row scale is written
  `[rows,1]` rather than the codec's flat `[rows]`, which is the shape real repacks carry — our own reader goes by
  element count either way. And the descriptor uses ComfyUI's key names, where `convrot` gates `convrot_groupsize`:
  a group size written without that flag reads back as zero.
- Verified by loading what we wrote back through the container and checking the `QuantInfo` it attaches, rather
  than only that the bytes parse.

- **CUDA's residency cache is the shared one now.** `GpuTransferHelper.State` derives from
  `GpuResidencyCache<ulong>`, so the weight/activation/cast collections, the four-step rebind, weight demotion,
  the keyed bindings, orphan parking, bulk offload and teardown exist once instead of twice. Every static signature
  is unchanged — the 1399 call sites in `CudaBackend` did not move — and what is genuinely CUDA's rides the
  overrides: the graph-capture arena in `AllocateDevice`, the persistent weight allocator in `AllocateWeight`, the
  stream-ordered copy and H2D profiling in `Upload`, Q8_1 sidecars in `OnActivationEvicted`, auto-promotion in
  `TryMakeResidentOnMiss`, and re-promotion blocking in `OnWeightDemoted`/`OnActivationOffloaded`.
- **Five behaviours the shared algorithm did not have, each found by a failing test rather than by reading.** The
  CUDA suite went 71 failures to 0 as they were fixed, and they are what "one implementation" actually costs:
  - *Arena-backed buffers must be recorded at allocation time.* Asking "is this address in a live arena?" at free
    time is a different question: a captured graph's arena leaves the live list when the graph is disposed, and
    after that every buffer it handed out looks ordinary, so teardown frees memory the driver already reclaimed.
  - *The orphan sweep may not run during a stream capture.* `cuMemFreeAsync` on a buffer allocated before the
    capture began is rejected outright — the guard existed, naming the three tests it fixed, and the port lost it.
  - *Teardown frees synchronously on a drained stream.* Mid-op a transient goes back to the pool so the free is
    ordered after the work that used it; at teardown that ordering is meaningless and the stream is about to go.
  - *A demoted weight is parked as an orphan OR queued for the persistent free, never both.* Both freed one
    pointer twice.
  - *`CachedBytes` is computed, not accumulated.* It was maintained by hand at half a dozen sites, and delegating
    those silently stopped updating it. A counter that drifts to zero while the memory is still resident is worse
    than no counter.
- **Bulk offload's pinned policy is stated rather than assumed, and the two backends assumed opposite things.** The
  shared cache skipped pinned activations; CUDA paged them out first. CUDA is right and its reason is in the type
  now: a pin means "survive `FreeActivations`", which DESTROYS the device copy, whereas offloading is
  non-destructive — contents go to host and come back on the next read. The low-VRAM lever's whole target is that
  cross-step state, so `MayOffload` is a hook with the conservative default and CUDA overrides it.
- `TryGetWeightCast`/`CacheWeightCast` take the target dtype. CUDA kept one cast per weight while the shared cache
  keys per (weight, dtype) — a superset — and both call sites already knew the GEMM dtype.

## alpha.111

- **CUDA's residency cache is the shared one now.** `GpuTransferHelper.State` derives from
  `GpuResidencyCache<ulong>`, so the weight/activation/cast collections, the four-step rebind, weight demotion,
  the keyed bindings, orphan parking, bulk offload and teardown exist once instead of twice. Every static signature
  is unchanged — the 1399 call sites in `CudaBackend` did not move — and what is genuinely CUDA's rides the
  overrides: the graph-capture arena in `AllocateDevice`, the persistent weight allocator in `AllocateWeight`, the
  stream-ordered copy and H2D profiling in `Upload`, Q8_1 sidecars in `OnActivationEvicted`, auto-promotion in
  `TryMakeResidentOnMiss`, and re-promotion blocking in `OnWeightDemoted`/`OnActivationOffloaded`.
- **Five behaviours the shared algorithm did not have, each found by a failing test rather than by reading.** The
  CUDA suite went 71 failures to 0 as they were fixed, and they are what "one implementation" actually costs:
  - *Arena-backed buffers must be recorded at allocation time.* Asking "is this address in a live arena?" at free
    time is a different question: a captured graph's arena leaves the live list when the graph is disposed, and
    after that every buffer it handed out looks ordinary, so teardown frees memory the driver already reclaimed.
  - *The orphan sweep may not run during a stream capture.* `cuMemFreeAsync` on a buffer allocated before the
    capture began is rejected outright — the guard existed, naming the three tests it fixed, and the port lost it.
  - *Teardown frees synchronously on a drained stream.* Mid-op a transient goes back to the pool so the free is
    ordered after the work that used it; at teardown that ordering is meaningless and the stream is about to go.
  - *A demoted weight is parked as an orphan OR queued for the persistent free, never both.* Both freed one
    pointer twice.
  - *`CachedBytes` is computed, not accumulated.* It was maintained by hand at half a dozen sites, and delegating
    those silently stopped updating it. A counter that drifts to zero while the memory is still resident is worse
    than no counter.
- **Bulk offload's pinned policy is stated rather than assumed, and the two backends assumed opposite things.** The
  shared cache skipped pinned activations; CUDA paged them out first. CUDA is right and its reason is in the type
  now: a pin means "survive `FreeActivations`", which DESTROYS the device copy, whereas offloading is
  non-destructive — contents go to host and come back on the next read. The low-VRAM lever's whole target is that
  cross-step state, so `MayOffload` is a hook with the conservative default and CUDA overrides it.
- `TryGetWeightCast`/`CacheWeightCast` take the target dtype. CUDA kept one cast per weight while the shared cache
  keys per (weight, dtype) — a superset — and both call sites already knew the GEMM dtype.

## alpha.110

- **SD1.5 and SDXL open through the container**, the last two recipes still calling `SafeTensorsLoader` directly.
  A `fp8_scaled` or GGUF UNet of either was invisible to them, and companion scales went unfolded.
- SDXL's checkpoint mapping is now `using`-scoped. It was disposed only on the failure branch, so a successful
  construction left it open for the life of the pipeline.

## alpha.109

- **LTX (0.9.x and 2.x) and HunyuanVideo open through the container.** Wan was flipped in Phase A; these three
  still opened their checkpoints with `SafeTensorsLoader` directly, so a GGUF was invisible to them and a
  `fp8_scaled` or `int8` build had its companion scales left unfolded unless a converter happened to fold them
  itself. Their loader lists hold `IDisposable` now, because what owns the mapping depends on the format.
- HunyuanVideo's side-component loader drops its explicit `ApplyFp8ScaledDequant`. The container folds companions
  on open, and leaving the call in would have implied it does not.

## alpha.108

- **`hartsy quantize` writes a quantized copy of a checkpoint offline.** The engine still never quantizes at load,
  so what a generation runs is the file on disk rather than a runtime decision about it; this is how the smaller
  file comes to exist.
- **It reads through the container, which is the part that matters.** `GgufQuantizer` could already write a GGUF,
  but only from a raw `SafeTensorsLoader` dictionary — and a `fp8_scaled` or `int8` checkpoint keeps its scales in
  companion tensors, so quantizing those bytes without folding them first writes a file wrong by exactly those
  scales. A plausible file, not an error. Reading through `CheckpointSource` folds them, and also means a GGUF can
  be re-quantized into a smaller one.
- Each tensor is materialized to F32 by the route its own form needs, because picking the wrong one is silent:
  `CastTo` folds an fp8 scale into the values but refuses a block-quantized source by design, and an int8 weight
  carrying no descriptor has an unknowable scale and is refused by name rather than guessed at.

## alpha.107

- **Qwen-Image and Mage-Flow schedule their conditioning per step**, finishing C.2 for every family that applies
  prompt weighting. Each distinct step-text is tokenized through the recipe's own template, so a branch carrying
  its own `(word:1.5)` emphasis is weighted per branch, encoded once inside the single text-encoder-resident
  window, and selected by step.
- Both recipes flatten for their base ids. Declaring `PromptScheduling` is what stops the tag being collapsed
  upstream, so the raw tag text reaches the recipe and would otherwise be tokenized as prose — the bug the Flux.2
  gate caught, avoided here by construction.
- Qwen-Image narrows two things that assume fixed conditioning: its cross-generation prompt cache is keyed on a
  single token array and is bypassed, and its first-block step cache is calibrated on drift between consecutive
  steps under fixed conditioning, so a reused feature would carry one branch's text into another's step. The
  activation reserve is sized from the longest variant rather than whichever is current.
- Which families declare the scheduling bit is now ledgered. Whether a pipeline really consumes a schedule cannot
  be checked by reflection, and declaring it without one is silently wrong rather than an error.

## alpha.106

- **The shared residency cache learns that weights and transients can need different allocators.** A new
  `AllocateWeight` seam, defaulting to the transient allocator so a backend with one allocator ignores it. CUDA is
  the case that forces it: a preloaded weight comes from `cuMemAlloc` and is released with `cuMemFree`,
  deliberately outside the stream-ordered pool every transient uses, and routing weights through the transient
  allocator would hand a pool block to a synchronous free — an error on that API, not a style question.
- **And that a cache miss can be satisfied by the backend itself.** `TryMakeResidentOnMiss` fires after the miss is
  counted and BEFORE a transient is allocated. The original design mapped CUDA's auto-promotion onto the
  post-upload `OnTransientUploaded` hook; measured, those are different moments and different buffers — promotion
  happens before any transient exists and allocates a fresh persistent buffer — so a post-upload hook could only
  have promoted the pool block it was handed, or uploaded the same bytes twice. Both hooks now exist, each with one
  caller and one clear moment.
- `PromoteToWeight` drops any activation for the same tensor rather than assuming there is none. A lookup checks
  weights first, so an activation left behind is shadowed by the weight on every later read — the device write
  silently discarded. Today's only caller fires on a miss, where the tensor is in neither tier, but it is a general
  seam and the invariant is cheaper to keep than to assume.
- The weight-cast cache keys conversions by `DType` instead of by `DType.Name`. `DType` is a `readonly record
  struct` with structural equality; projecting it to a string to use as a key was stringly-typed for no gain.

## alpha.105

- **Swapping one model for another in a single process is now tested, and the test can fail.** Load, generate,
  tear down, load something else — the ordinary shape of a long-running server, and the scenario the
  `State.Unregistered` guard was written for after a GGUF model switch crashed. Nothing exercised it: the
  byte-identity harness runs one model per CLI invocation, and the CUDA suite's backends come and go without a
  second model taking their place. It alternates an orderly dispose with abandoning both model and backend, since
  only the latter leaves a queued cleanup pointing at a state the GC may already have collected.
- **It asserts device memory comes back, which is the assertion with teeth.** All three bugs the Vulkan half of
  this work produced were teardown leaks — invisible to a single generation, and to any test that builds one
  backend and stops. Disabling `cuMemFree` makes this test fail with 16 GB unreturned across four swaps; that is
  what it is for.
- **The measurement needs a pool trim, and finding that out was the interesting part.** Before the trim it
  reported 6.9 GB "unreturned" on a perfectly healthy run: every activation free goes through `cuMemFreeAsync`,
  which hands the block back to the stream-ordered mempool, and a pooled block still counts as USED in
  `cuMemGetInfo` until trimmed. Measuring without the trim would have reported a leak every time.
- Recorded honestly: the soak does **not** reproduce the original crash. That needs a `ConditionalWeakTable`
  finalized and then resurrected, a GC-timing window no test can force — verified by removing the retirement guard
  entirely and watching the soak still pass. What it does cover is the swap working end to end, every state
  retiring, and the memory returning.
- Missing models make it skip, as the suite's convention is — except under `HARTSY_REQUIRE_MODEL_SWAP_SOAK=1`,
  which turns absence into a failure. These tests print `SKIPPED` and return, which xunit records as a **pass**,
  so a gate that silently skips is a gate that silently passes.
- `TestPaths.Llm` gains Qwen3-4B Q4_K_M: a different family and a different quantization from the Llama beside it,
  so a swap between them crosses the dequantize path as well as the residency cache.

## alpha.104

- **Flux.2 schedules its conditioning per step.** `<alternate:a, b>` and `<fromto[N]:a, b>` reached it as prose or
  not at all; now each distinct step-text is tokenized through the recipe's own chat template — so a branch
  carrying its own `(word:1.5)` emphasis is weighted per branch — encoded once, and selected by step. `<alternate:>`
  over 30 steps costs two encodes, not thirty, because variants are deduped globally rather than per contiguous run.
- Two things a schedule cannot coexist with, both narrowed rather than ignored. **Step-graph capture** replays one
  op sequence with the conditioning pinned as step-invariant, so it is skipped for a scheduled prompt and logs why
  — the output is unaffected, the step is slower. **A regional prompt** builds its text stream and attention bias
  once from the base conditioning, and switching underneath it would leave both stale while the shapes still line
  up, so that pairing is refused by name.
- **`<fromto[0]:a, b>` was a silent no-op for every family that cannot schedule.** A fromto switches at
  `step < when`, so at `when = 0` it has already switched before step 0 runs and `b` is its step-0 text — but the
  flattener returned the first branch unconditionally, resolving the tag to exactly the phrase it was written to
  replace. Pre-existing; it took wiring a family far enough to compare against a plain prompt to see it.

## alpha.103

- **The residency cache answers questions about a tensor now, instead of handing out its dictionaries.**
  `TierOf(tensor)`, `IsPinnedActivation`, `OwnsBuffer` and the four counts land on `GpuTransferHelper.State`, and the
  ~35 test assertions that reached into `WeightCache`/`ActivationCache`/`WeightCastCache`/`PinnedActivations`/
  `CachedPointers` now ask those instead. The collections are an implementation of residency, not the definition of
  it, and every assertion written against them was coupled to that choice — which matters immediately, because the
  shared `GpuResidencyCache<TBuffer>` keeps them `protected` and those are the teardown and isolation tests that have
  to keep working across the migration.
- **`GpuResidencyTier` makes "resident as both" unrepresentable, and `TierOf` throws when it happens anyway.** A
  lookup checks weights first, so a tensor in both tiers has its activation — the bytes an op just wrote — shadowed
  by a stale weight on every later read. That is the auto-promote-discards-device-writes bug, and until now nothing
  asserted it could not occur: `PreloadWeight` tests only the weight cache before inserting, so a tensor computed as
  an activation and then preloaded was the plausible route in. The whole CUDA suite passes with the check live,
  including the RoPE-table lifecycle tests that take exactly that route, so the invariant is now a tested fact
  rather than an assumption the shared cache was about to be built on. The check stays on the query surface and off
  the production lookups deliberately: a guard at the weight-cast call site would fire after `CopyToDevice` had
  already served the stale bytes, crashing the generation somewhere unrelated to the cause. Guarding the read
  itself is a design call for the migration, not for this PR.
- **The write that could create that state is guarded, at the write.** `RegisterCachedWeight` refuses to make a
  tensor a weight while it is still a live activation. Nothing could reach that state today, but only by accident:
  all three callers read `DataPointer` to find the host bytes to upload, which fires the activation's sync callback
  and evicts the entry. An invariant held by a side effect of an unrelated read is one line from being lost — a
  caller uploading from a pinned or mapped buffer would never touch `DataPointer`, and `CudaStreamingWeightCache`
  is already most of the way there. The check sits where the state would be established rather than where it would
  be noticed, and weight registration is a load-time path, so it costs a dictionary probe per weight.
- `HartsyInference.Cuda` references `HartsyInference.Gpu` for the first time. Nothing depends on it yet beyond the
  enum — it is landed here, on its own, so the package-graph change is proven separately from the cache migration
  that needs it. Verified by packing rather than by building: `HartsyInference.Cuda.nupkg` declares
  `HartsyInference.Gpu 2.0.0-alpha.103`, which is the claim that matters at publish time.
- **The publish workflow's partial-release guard did not know `HartsyInference.Gpu` exists.** That list is there to
  refuse a release where one project failed to pack, precisely so consumers pinning exact versions do not find a
  dependency missing from the feed — and `Gpu` has been shipping unguarded since it was created, with `Vulkan`
  already depending on it. This PR adds a second dependent, so it adds the package to the list.

## alpha.102

- **Six more Runtime knobs were frozen, in mutable statics the previous check did not look at.** The scope lint
  required `readonly`, so `internal static bool FusedMmaGemm = EngineKnobs.Int8FusedMma.Value;` and five like it
  slipped through — bound at type-initialization exactly as a readonly field is, and worse rather than better, since
  a mutable process-wide static has no per-request isolation at all. The lint no longer asks for `readonly`: a
  static field initialized from a knob is frozen, full stop.

  The tell was who wrote to them. Every one of the writable ones was assigned only by tests, reaching past a knob
  that could not reach the code — `CudaBackend.FusedMmaGemm`, `FuseHeadGateIntoQuant`, `GroupedLinear` and
  `LtxVideo2Attention.TokenMajorAttention`. Those are live reads now and the tests set the knob instead, so what
  they exercise is the path a real request would take.

  `WanVideoDebugDump` kept its runtime setter, which is legitimate, but seeded it from the knob at
  type-initialization — so the first generation in a process named every later one's dumps. An explicit tag now wins
  and the setting decides when nobody set one.

- **The dump directory next to that tag was frozen too, one hop further out.** `DebugDumpSink` resolved its knob in
  its constructor, and all nineteen dump sinks are held in `static readonly` fields, so eighteen of them bound the
  directory at type-initialization. The lint cannot see this shape — the read is inside an instance constructor —
  but the tell was there again: the one sink that opted out of caching was the one whose parity tests needed the
  knob to reach the code. The opt-out is gone and every sink resolves per access, which also removes a
  created-once flag that would have sent later dumps to a directory it never made.

- **And a third shape, in a lazily-initialized static.** `Hunyuan3DDebugDump` resolved its dump directory behind a
  double-checked flag, so the first `Enabled` read in the process decided it for every later one. The lint could not
  see that either — the read sits in a property body, which is normally the safe place for one — so it now also
  flags a knob read whose result is assigned to a static field, wherever that assignment lives. It gained a second
  detection at the same time: a namespace-qualified `Configuration.EngineKnobs.X.Value` was invisible to it, and
  though no read in `src/` is written that way today, a check with a way around it is worth less than the diff that
  closes it. Both rules were mutation-tested — a planted violation of each fails the build, and a live read in a
  property still passes.
- `WanVideoDebugDump.SetTag(null)` pinned an empty prefix instead of handing the choice back to the setting, so the
  CFG pipeline's first clear masked the configured tag for the rest of the process. Three places said it should do
  the opposite, including the doc comment directly above it.

- **A fourth shape, and the one that reached real numerics: an instance field on a cached object.**
  `MiniMaxMusic3ArPipeline` bound `numerics.mm3CfgBatch` into a `readonly` field in its constructor, and that
  pipeline is built during model load and then cached and reused by `MusicService`, so every request after the
  first inherited whatever the loading request happened to see — its own `KnobProfileScope` override could not
  reach it. The value is now read once per generation rather than per use, deliberately: the two call sites must
  agree, since the feedback is built with two rows exactly when the batched step consumes two, so reading live at
  each would let them tear. The lint flags an instance field initializer now as well; a syntax tree cannot know
  which objects outlive a request, and presuming the freeze costs nothing when no legitimate instance exists.
- `WanVideoDebugDump`'s tag override is `AsyncLocal`, for the reason `KnobProfileScope` is. The CFG pipeline sets it
  around each branch forward, so on a plain static two generations on two devices would relabel each other's dumps.
  Only filenames are at stake, but a dump whose name lies about which branch produced it is worth nothing.

## alpha.101

- **MiniMax-H3 runs from a GGUF, verified by generation.** The `unsloth/MiniMax-H3-GGUF` Q4_K build renders the
  same scene as the fp8 checkpoint it is a requant of, at 141 frames 512x288 seed 1 and 30 steps, and its hash is
  now bound in `VideoProfileManifest` so it plans with H3's real task semantics instead of `UnknownBaseProfile`.
- **The GGUF is about twice as fast per step here** — 5.20 s/step against fp8's 10.39, measured as a two-point
  difference so load and decode cancel. That is the opposite of the expectation that a transiently-dequantized
  build must be slower. The likely reason is residency rather than arithmetic: 19.4 GB of fp8 weights in a 24 GB
  card leaves little room for anything else, where Q4_K needs roughly 3.7 GB. Inferred, not measured.
- **Plain Q2_K from that repack is published but unusable**, and the loader is not at fault. It renders a
  repeating lattice where Q4_K renders the scene at identical settings, though the two files differ only in the
  format of the same 208 weights; our Q2_K dequant agrees with our CPU codec and matches ggml's reference walk.
  Whether 2.6 bits is simply too coarse for an already-pruned DiT, or that repack's quantizer is at fault, is not
  established — so the hash is deliberately left out of the manifest rather than recorded as broken.
- `h3_bench.sh` stops claiming you need more steps when the truth is that it found no timings at all. The CLI
  prints `denoise [n/m]` with no per-step figure, so the harness cannot report s/step; it now says so and names
  the two-run method instead.

## alpha.100

- **A GGUF MiniMax-H3 text encoder loads.** Preflight refused the published repack for a shape error —
  `visual.patch_embed.proj.weight` "must be [1152,3,2,16,16], got [3456,2,16,16]" — which reads like a corrupt
  download but is a container limit: ggml caps a tensor at `GGML_MAX_DIMS = 4`, so no GGUF can hold a rank-5 weight
  and every repack folds the leading pair (1152 x 3 = 3456). Row-major the two layouts are the same bytes in the
  same order, so the fold is a relabeling and nothing needs converting.
- Behind that refusal was a silent one. The input channel count was read as `Shape[1]`, which is 3 on the rank-5
  weight and the temporal patch size — 2 — on the folded one, so the tower would have built a `[1152, 1024]`
  projection out of a `[1152, 1536]` weight and copied the front of it. The count is now divided out of the element
  count, which gets the same answer from either layout, and the copy checks its source length instead of trusting
  the caller's arithmetic.
- The fold is accepted as that exact shape, not as "rank 4 that multiplies out": `[1152,6,16,16]` is still refused,
  and the refusal names both forms it would have taken.
- A patch embedding that is itself block-quantized is decoded rather than cast. Our own GGUF policies quantize
  rank>1 non-norm weights, and this is one; `CastTo` refuses a quantized source by design, so letting the shape
  through at preflight without decoding here would only have moved the same failure into construction.

## alpha.99

- **`(word:1.5)` reaches five more families the way SwarmUI means it.** Prompt weighting worked only on SD1.5 and
  SDXL, because the code that applied it was written against `ClipTextEncoder` and nothing else could reach it.
  Qwen-Image, Flux.2 and Mage-Flow now apply the mechanism ComfyUI would apply to them, and SD1.5/SDXL keep theirs
  unchanged.
- The mechanism is not a per-family opinion. SwarmUI runs one probe —
  `use_attn_token_weights = not token_batches_have_weights(clip.tokenize("(x:2)"))` (`SwarmText.py:553`) — so a
  family blends on the encoder output only when EVERY tokenizer arm keeps weights, and one weight-keeping arm puts
  the whole family back on the blend. Reading that rule rather than grepping for `disable_weights` moved four
  families off the mode the plan had assumed for them, Lumina2 among them.
- **A recipe may not declare a mode its pipeline cannot act on**, and a test now enforces it. The declaration is
  what keeps the `(text:N)` parens in the prompt; a family that declared weighting without applying it would hand
  the parens and the digits to its encoder as prose — worse than not weighting at all. Thirty-three ledgered
  families therefore still declare nothing, each with its reason recorded.
- **Weighting on Kandinsky5 is a no-op, and that IS parity.** SwarmUI selects the blend for it, but
  `Kandinsky5TEModel.encode_token_weights` returns the Qwen conditioning plus CLIP-L's pooled vector and discards
  the blended hidden states, so the weights never reach the model. Implementing the blend there would have broken
  parity rather than achieved it.
- **The refiner prepares the caller's prompt for its own family**, rather than inheriting whatever the base was
  left with. Preparation is destructive in both directions: a weighting base's `(text:N)` grammar would reach a
  refiner whose encoder reads parens as prose, and — the direction easier to miss — an unweighted base collapses
  `<weight[1.5]:x>` to `x` before a refiner that *can* weight ever sees it, so the emphasis is silently gone.
  Neither is recoverable from a prompt already resolved for someone else.

## alpha.98

- **A per-request setting was silently ignored for 53 of the engine's knobs.** `KnobScope.Runtime` declares a knob
  "read each generation; safe to override per request", and `KnobProfileScope` exists to carry a request's settings
  into generation — it is pushed per request by the image and video services, and uses `AsyncLocal` specifically so
  two engines generating on two devices cannot decide each other's numerics. But 56 knobs were read into
  `static readonly` fields, which bind once at type-initialization and are never re-read. For all of those the
  request override did nothing, and worse than nothing: a process-wide field means whichever request touched the
  type first decided for every later request and both GPUs, which is the exact failure the scope's `AsyncLocal` was
  chosen to prevent.

  The scope was a declaration nothing enforced. It is now read at the point of use — activation dtype, step-graph
  capture, KV-cache precision, orphan sweeping, INT8 row budgets, im2col band caps, every probe and dump — and
  `KnobScopeIsEnforcedTests` fails the build if a `Runtime` knob is frozen again. There is deliberately no allowlist
  file: the knob's own declared scope is the allowlist, so marking one `Construction` is how you say a value really
  is baked in, and the three that still are (Vulkan coopmat, profiling and submit-per-op, all decided when the
  device and its pipelines are built) are named by a test.

  Measured first, because the fix depends on it. `KnobResolutionBenchmarks` is kept so the resolve cost stays
  reproducible — roughly 57 ns with a request profile pushed, which is the production path. The other half of the
  number was a one-off: counting the per-op orphan sweep, the hottest reader, gave 9,741 calls for a 1024x1024
  8-step image, so about 0.56 ms across an eleven-second generation. Live reads, rather than a snapshot cache built
  for 0.005%.

- **Tests configured the engine through environment variables nothing reads.** 95 call sites across 24 files set a
  variable that was retired when settings moved to knobs, so they were measuring defaults under a name claiming
  otherwise. One comparison ran both of its arms in the same configuration. `TestsDoNotSetKnobEnvVarsTests` now
  fails on any test that sets a knob's legacy name, taking the list from the registry rather than a hardcoded copy;
  harness gates like `HARTSY_REQUIRE_REAL_WEIGHTS` are untouched, being real environment variables read by the
  tests themselves.

- **Wan-Animate-2's driving-cache policy named a dead environment variable** in its logs, its unrecognized-value
  warning and the note on its out-of-VRAM message, telling the reader to export something inert. It now names the
  setting id, which `--set` and the settings file accept.

## alpha.97

- **MiniMax-H3 opens every component through the container**, so a GGUF build of the DiT, either VAE or the text
  encoder loads like any other checkpoint. The planner learned to read a GGUF header in alpha.90, which left the
  recipe as the only thing between an H3 GGUF and a generation — and meant asset resolution had to keep pretending
  `.gguf` files were not there, since offering a candidate the recipe could not open only moved the failure later.
- Norm promotion no longer goes through `CastTo`, which refuses a quantized source by design: decoding a block
  layout is the dequantizer's job, and a GGUF build carries quantized norms.
- **Q2_K and Q3_K dequantize on the GPU.** These were the two shipped H3 quants CUDA had no kernel for, and the gap
  mattered more than the others: at 2.6 bits per weight a 6.7 GB Q2_K build expands roughly sixfold when the loader
  has to widen it on the host, putting a model that would fit a 12 GB card out of reach of a 24 GB one.
- **Planning no longer refuses an audio VAE it can actually load.** It demanded the unfused PyTorch weight-norm
  parametrization for every convolution, but both forms are in circulation — the vendor release carries it, the
  ComfyUI repack ships it collapsed — and the decoder and encoder load either. Eighteen tensors present under their
  fused names were reported missing. That is the worst shape of planning error: the refusal is authoritative and the
  capability it denies exists.
- H3's text encoder reads its embedding row scale from wherever the fold left it. Reading only the companion key
  made a container-opened checkpoint look like one with a missing scale, and the refusal below rejected the
  published int8 build.
- MiniMax-H3 asks only the devices that will execute its blocks before deciding whether a quantized DiT can stay
  packed. It never runs context-parallel or CFG-parallel — the recipe warns the operator about exactly that — so a
  peer with no packed-weight kernel used to widen the whole checkpoint on the host for a device that reads none of
  it, which turns a 6.7 GB Q2_K build into roughly 40 GB.
- A quantized `.gguf` component no longer displaces a proven dense one. Component ranking read only the ComfyUI
  markers (`fp8`, `int8`, `nvfp4`), so `…video_vae-Q4_K.gguf` landed in the dense class beside the FP16 build and
  the size tie-break then chose it — the silent substitution the video VAE's explicit-selection gate exists to
  prevent.

## alpha.96

- **YuE2 loads its own published `int8_convrot` repack.** It was refused by name, on the grounds that the per-layer
  quant reader did not exist; it does now, and it lives at the container, so the refusal's reason was stale. The
  refusal itself was not: with it simply deleted the file does not load, because YuE2 quantizes more than the
  refusal's comment assumed. Reading the published header rather than the doc showed 229 `int8_tensorwise` weights
  at ConvRot group 256 — and among them the **embedding table**, the **output head**, `llm2vae` and the timestep
  MLP, three of which this model reads on the host.

  So four things had to change before the refusal could go. The fused QKV and gate/up splits now narrow each part's
  per-row dequant scale to the rows it takes and size the copy through the block layout rather than
  `DType.SizeInBytes`, which is 0 for every block quant and copies nothing. The host-read entries are decoded
  instead of cast — raw int8 bytes in a Hadamard-rotated basis are a different weight, not a wrong magnitude.
  `GenericTransformer` stops byte-concatenating Q/K/V and gate/up into one fused dispatch when a part carries a row
  scale the concatenation renumbers away, and its host-gathered embedding table decodes such a weight rather than
  casting it — both apply to every model that loads a ComfyUI int8 checkpoint, not just this one. The AR stack's
  32,769-row semantic head is decoded once at load rather than kept packed, because that row count is not a
  multiple of four and the packed GEMM would otherwise dequantize the whole window once per token.

  Not yet run against the real 7.8 GB file: the numbers here come from the published header and from synthetic
  cases, and the song itself still needs a listen against the BF16 build.
- **YuE2 opens through `CheckpointSource`**, so a GGUF build loads as readily as a safetensors one, and
  `AudioLmQuant` selects a placed `q4_k`/`q8_0` GGUF when one exists. Nothing is quantized at load and nothing new
  is downloaded — the hub ships no GGUF YuE2, and writing one is the offline tooling's job.
- A LoRA sent to YuE2 now says so in the log. It reached the runner cache key but nothing applied it, so the
  request got a fresh runner that generated exactly the base model's song.

## alpha.95

- **Seventeen more image families open their checkpoints through the one container**, so each accepts a GGUF or a
  quantized repack rather than safetensors alone: Chroma and its Radiance and Zeta variants, Z-Image, Lumina-2,
  AuraFlow, Anima, Ideogram 4, ERNIE-Image, Krea 2, HiDream, Boogu, Mage-Flow's side models, Kandinsky 5, Lance,
  OmniGen 2 and F-Lite. Every image recipe now loads this way except SD1.5/SDXL, which need rank-4 quantized
  convolution support first, and Lens.
- **Chroma's four split helpers sized their copies from `DType.SizeInBytes`, which is 0 for every block quant.** On
  a GGUF they produced correctly-shaped, entirely zero projections while the dense path stayed byte-perfect — the
  same arithmetic that was fixed in the QKV splits, in four more places. They go through a new quant-aware
  `SplitRows`, which narrows each piece's row scale before the first allocation so a refusal cannot strand one.
- **F-Lite never folded its quantization companions at all**, so an fp8_scaled build ran every weight at `1/scale`.
- Zeta-Chroma's attention fusion concatenated Q/K/V and dropped every companion; it refuses a per-row-quantized
  weight by name instead, and likewise a build that stores Q, K and V in three different dtypes — the fused tensor
  can declare only one, and a K or V wider than Q used to be copied past the end of it. Krea 2's pre-rename
  companion carry is deleted, dead now that folding precedes renaming.
- Lumina-2, HiDream and OmniGen 2 widen to the dtype their transformer actually runs, rather than the F16 default
  that would have left a GGUF mixing dense F32 with widened F16.
- **A component published as `.gguf` inside a diffusers folder was invisible.** F-Lite, Lance, Kandinsky 5, Krea 2
  and Boogu discovered their components with a `*.safetensors` glob and threw before the container could be sniffed,
  so the GGUF path they now advertise was unreachable without renaming every file to a lie. Folder discovery reads
  the leading bytes instead, the way single-file loading always has, and a set that mixes the two containers is
  refused rather than merged into one dictionary.
- **Selecting a Lumina-2 GGUF stored beside the original sharded release loaded the release instead.** Any sibling
  `*.safetensors.index.json` used to expand the selection into every safetensors in the folder; the index now has to
  list the selected file before it expands anything, so a repack — or any second checkpoint parked there — loads as
  itself.

## alpha.94

- **A LoRA now applies to a block-quantized base without requantizing it.** The classic codecs a GGUF uses have no
  quantizer at all, and requantizing a merged result degrades both the base and the LoRA — so the delta rides on the
  weight as a low-rank adjunct and is accumulated inside the GEMM, which is what ComfyUI-GGUF does per forward.
- The adjunct is never written to the shared weight. Cached converted dictionaries, resident models and the
  identity-keyed device cache all hold that object, so a patched base would have leaked one request's LoRA into the
  next with nothing in the cache key to show for it. It rides a borrowed view that the stack owns and disposes.
- **Every GEMM entry a patched weight can reach either applies the adjunct or refuses by name.** A quiet miss reads
  as "the LoRA looks weak", never as an error, so `LinearImpl` was split and the accumulate moved into a wrapper
  around the dozen fused paths that return early. The fused GELU and head-gate entries un-fuse instead of adding
  afterwards, because the delta has to land before the activation.
- **One call site for LoRA merging.** `LoraApplier` is gone and all 36 recipe sites go through `RecipeLoraMerge`,
  which owns the merge-before-load invariant.
- **A text-encoder LoRA strength now does something.** `TencStrength` reached the cache key but never the merge, so
  the stack applied one strength to everything; CLIP-L, CLIP-G and the new `TextEncoder2` target take it properly.
- **A DoRA adapter is now actually decomposed on a dense base.** The magnitude vector reached `LoraDelta.DoraScale`
  and the decomposition was written and pinned against ComfyUI's reference, but nothing on the merge path called it —
  so a DoRA file merged as a plain LoRA with its magnitudes dropped, no warning anywhere. The quantized path already
  refused by name; only the dense one was silent, which is the exact failure the rest of this work exists to remove.
  A DoRA aimed at one slice of a fused projection is refused instead: its normalizer is defined over a whole weight,
  and on the input axis that is a column norm across every row, which a third of them cannot supply.
- A stacked LoRA carrying layers for a component the caller passed no dictionary for now warns by target and count.
  Adding `TextEncoder2` would otherwise have introduced exactly the silent partial merge this work is about — those
  keys used to be skipped as unrecognized and would now parse cleanly into a target nothing consumes.

## alpha.93

- **The shared residency cache now reclaims buffers an op displaces, and stops serving stale weights.** Four things
  CUDA's own cache had already solved, adopted before CUDA moves onto the shared one — so that migration is a port
  onto familiar ground rather than a rediscovery of the same bugs in a new place.

  A buffer displaced by a rebind used to stay owned by nothing until teardown. Whether it is the op's own input —
  whose cleanup will free it — or nobody's is not knowable at the moment of displacement, so it now parks: the
  caller's release claims it, and whatever is left is freed when the NEXT op starts, by which point every previous
  op's `finally` has provably run. Freeing at teardown instead was not a fix but a deferral; measured on CUDA before
  it had this, twelve `Linear` calls at a 563 MB output stranded 5942 MB.

  More seriously, a resident weight whose tensor an op bound to a new buffer stayed a weight, and lookups check
  weights first — so every later read returned the pre-op bytes and the device write was silently discarded. That is
  the same shape as the auto-promotion bug fixed in CUDA in August, and it was waiting in the shared base for the
  first backend to write through a weight. A tensor bound as an op's output is no longer a weight, whatever route
  made it one, and its cached dtype conversions go with it.

  `PromoteToWeight` gives a backend the seam it needs to promote a tensor it has seen uploaded twice — the buffer
  has to enter the weight cache and the owned set together, or the caller's own cleanup frees what the cache now
  points at — and plants the demotion binding that makes promotion safe. Promotion happens behind the caller's back,
  so host data stays authoritative: a later host write drops the device copy rather than syncing it back, because
  without that a write would leave the stale device bytes cached and every later lookup would serve them. An
  explicit preload deliberately plants nothing, since there the caller asked for residency and owns the lifetime.

  The callbacks a tensor fires on read or dispose now check disposal before touching anything, behind a gate a
  backend can hold closed while it retires: a binding outlives the cache that planted it, and on CUDA that path
  threw during a model swap.

## alpha.92

- **Two backends on the shared residency cache would have handed out the same binding key.** The counter was a
  static field inside the generic class, and such a field exists once per CLOSED type — so a cache over one buffer
  type and a cache over another each started at 1. That is the collision the key exists to prevent, moved up a
  level: a host tensor resident on two devices, which split placement makes ordinary, would carry both bindings
  under one key and their finalizer-cleanup buckets would collide, so one backend's drain runs the other's device
  cleanup. Latent until a second backend joined the base, which is the next step.

  Moving the counter off the generic class was only half of it: CUDA's state registry allocates binding keys from a
  private counter of its own, also starting at 1, so a CUDA cache and a Vulkan cache collide today — not after some
  future step — the first time one tensor is resident on both. The sequence now lives in Core beside the bindings it
  names, where every backend already reaches, and CUDA draws from it. A source test keeps it the only one: the bug
  has now been written twice, each copy correct alone, and neither was visible until a second backend existed.

## alpha.91

- **Vulkan's residency cache is now the shared one.** Its three caches, lookup order, tensor-binding lifecycle,
  weight-cast cache and offload policy come from `GpuResidencyCache<TBuffer>`; what stays behind is what is
  genuinely Vulkan — the ReBAR-or-staging upload, the non-coherent flush, deferred frees against the command
  stream's timeline, and the step-graph retain list. Output is byte-identical: SD1.5 and Krea2 both hash-match
  their pre-migration images exactly.
- **Vulkan drains its finalizer cleanup queue.** A tensor finalized rather than disposed cannot free its device
  buffer from the finalizer thread, so the work is queued for a safe point — and nothing on this backend ever ran
  it, so those buffers stayed allocated until the backend itself was torn down. The outermost op scope runs it now.
- Each cache instance takes its own binding key instead of every Vulkan device sharing `0`, so one device's
  teardown can no longer drop another's binding on a tensor resident on both.
- Three teardown bugs found and fixed while migrating, all recorded in the troubleshooting notes: a deferred free
  at teardown is never serviced, `public new` silently keeps the base implementation where `override` was meant,
  and a buffer orphaned by an in-place re-cache is reachable from the owned-buffer set but from no cache.

## alpha.90

- **Every checkpoint now opens through one container, whatever format it is in.** Quantized-checkpoint support
  had been wired architecture by architecture: a recipe that wanted GGUF grew a second constructor parameter and
  a second code path, so GGUF loaded for four image models and no video ones, and the ComfyUI fp8/int8 companion
  fold was copy-pasted into 24 of 47 checkpoint converters and simply missing from the rest.
- Nothing about a container is architecture-specific, which is the whole point. A diffusion GGUF is a repack —
  the publisher quantizes the released safetensors file and keeps its tensor names — so once the keys are mapped,
  the shapes relabelled and the quantization companions folded, a converter cannot tell the two apart.
  `CheckpointSource` does those three things once, sniffing the container from its magic bytes rather than its
  extension, because repacks are routinely published under the wrong one.
- **The fold runs before the converter, and that ordering is the fix.** A converter renames `.weight` and has no
  rule for `.weight_scale`, so folding afterwards pairs nothing and drops the scale — and a weight without its
  scale is not an error, it is a weight hundreds of times too large that renders as noise at the end of a
  generation. Folding at the container makes that class of bug unreachable.
- The same failure existed one level down: a `Tensor` built over another's bytes — a reshape, a dtype relabel, a
  same-device copy, a fused-projection split — started life with no scales at all. Those now carry the per-tensor
  fp8 factors always and the per-row companions whenever the row numbering survives, and refuse rather than pair
  each row with another row's scale when it does not.
- **A quantized weight the backend has no kernel for is now caught at load.** It used to fail inside the first
  GEMM, minutes into a generation, with a stack trace naming a kernel rather than a file; the loader asks the
  backend what it can hold packed and widens the rest on the host, saying which dtypes and why. A Q2_K or IQ4_NL
  diffusion GGUF — both routinely published — loads slowly instead of crashing.
- **Flux.1, SD3/SD3.5 and the Wan video family load GGUF too.** They worked on safetensors and simply could not
  open a quantized build; routing them through the container is the whole change. Verified with real generations
  from real community files: Flux.1-dev Q4_K_S (city96), SD3.5-medium Q8_0 (city96) and Wan 2.1 T2V 1.3B Q8_0.
- Two things that had to be fixed for those, both invisible until a published file was actually read. The SD3
  converter only routed keys under a `model.diffusion_model.` prefix, and city96's SD3.5 GGUF ships bare LDM keys,
  so every transformer tensor was dropped and the DiT loaded empty. And Wan's checkpoint is the first to carry
  tensors that are not matrices — 31 rank-3 modulation tables and a rank-5 Conv3d patch embed — which only arrive
  with the right shape because the container now reverses every ggml axis rather than just the two of a matrix.
- **A shipped checkpoint that rendered black now works.** Black Forest Labs' own `FLUX.2-klein-4b-fp8` carries 80
  fp8 weights and 160 companion scales, and the Flux.2 converter was one of the 23 that never folded them — so every
  fp8 weight ran at scale 1.0 instead of `stored x scale`, and the generation saturated to a fully black image. This
  is what the fold gap looks like when it lands on an official release, and why the fold belongs to the container
  rather than to a 24th converter.
- **A GGUF video checkpoint reaches planning.** The planner opened every checkpoint as safetensors, so a GGUF
  build died before any recipe was reached, and MiniMax-H3's component resolution could not even see a `.gguf`
  file. Both read the shared header now, and a component's format reports as `gguf-q4_k` and the like.
- **A fused projection can be read in windows when it is block-quantized.** This is what put a GGUF MiniMax-H3
  out of reach rather than merely making it slower: H3 reads its packed `qkv_proj` in two windows, that chunking
  is how the model runs at all, and the weight it chunks is the one the quantization applies to.
- bitsandbytes NF4 checkpoints load, through a decoder that had been written and never wired. Every quantity the
  file declares is reconciled against its actual byte counts first, so a layout misread refuses by name instead
  of decoding to plausible noise.
- Fourteen GGUF key mappers were the same class fourteen times — a family name, a recognition rule, and a method
  returning its argument — and are now one table of recognition rules. Writing them side by side surfaced that
  Flux.2 was asked after Flux although it keeps Flux's block naming, so a Flux.2 GGUF with no declared
  architecture detected as Flux.1.
- The GGUF writer emitted dimensions in the engine's order while every other tool reads ggml's, so a file the
  quantizer produced came back with every matrix transposed. Nothing consumed those files yet; they are now
  readable by other tools and by our own recipes.

## alpha.89

- **A shared GPU layer, `HartsyInference.Gpu`.** Nothing references it yet: the package is built and tested first so
  each backend can be moved onto it one at a time, with its own suite green at every step.
- `GpuResidencyCache<TBuffer>` is the device-residency cache both GPU backends had written separately — the same
  three caches, the same weight → activation → fresh-upload order, the same four-step activation bind. A backend
  supplies five operations that genuinely need an API (allocate, free, upload, download, make-current) and inherits
  the rest. Two drifts between the old copies are settled by having one: only one of them drained the finalizer
  cleanup queue, so tensors finalized rather than disposed leaked their device memory on the other; and one keyed
  every device's tensor binding as 0, which holds only until two of its devices are used at once.
- Because the cache decides *which* tensor is resident rather than doing any transfer itself, it is testable against
  a fake buffer with no GPU at all. Nine tests cover the cases that previously needed hardware to reach: re-caching a
  tensor in place, a weight surviving an activation sweep, pinning, per-device binding independence, arena-owned
  buffers, and a host read during capture.
- `OpProfile` gives both backends the per-op timing only one had, so the pipelines that already call
  `ResetOpProfile`/`DumpOpProfile` stop silently producing nothing on the other.

## alpha.88

- **The backend contract now describes a device instead of listing the two backends that exist.** `IsGpu` tested for
  CUDA-or-Vulkan by name, so a device added later would have read as a CPU to every caller that gates on it — same-
  device serialization, weight preloading, VRAM reclamation would each have skipped it silently. It asks whether the
  device is not a CPU, and `DeviceType` names ROCm and Metal so that question has real answers to give.
- `CacheWeightCasts` moved onto `IBackend`. Both GPU backends already had the property with the same name and the
  same meaning, and the one caller reached them through a type test naming each — so a third backend would have been
  skipped while the log line still claimed the flag had been applied. The recipe helper no longer references the CUDA
  or Vulkan packages at all.
- Backends report their vendor, device name and total VRAM. A VRAM tier can only be resolved from a number somebody
  publishes, and nothing published one. Vendor matters because several decisions are per-vendor rather than per-API —
  cooperative-matrix reliability above all — and a software rasterizer is its own vendor, since llvmpipe otherwise
  reports the silicon vendor of the host and would walk into a hardware comparison.
- Vulkan finally declares that its convolution bands its im2col workspace. It has done so since the Krea2 VAE-decode
  fix, but never said so, leaving the VAE planner to assume the naive blow-up on that backend.

## alpha.87

- **A mistyped command-line option now fails instead of being ignored.** Spectre collects an option no command
  declares as a "remaining" argument, and nothing reads those — so a typo did not fail, it ran the generation with
  a different setting than the one asked for and reported success. `--cfgscale 2` (the option is `--cfg`) quietly
  generated at the model's default guidance, and `--detect` (it is `--mode detect`) quietly fell through to CLIP
  and then died several layers down in a model loader complaining about a missing text-encoder weight. Both were
  found the hard way, drawing a wrong conclusion from a run that had not used the settings it was given.

## alpha.86

- **Vulkan convolves a batch.** `Conv2D` refused `batch > 1` outright, which is what made SDXL unusable on the
  Vulkan backend: its fused denoise loop runs one batch=2 UNet forward per step, with the positive and negative
  prompt concatenated for classifier-free guidance, so the first convolution of the first step threw. SD1.5 never
  hit it only because it runs CFG as two separate batch=1 passes. No kernel changed: the im2col shader already
  wrote each image's columns as its own block, and `matmul_tiled` has carried the `aOffset`/`bOffset`/`cOffset`
  push constants "for batched dispatch" all along — the convolution now walks them per image the way
  `BatchedMatMul` already did. The column-tile budget is divided across the batch, so the cap that exists to bound
  peak im2col memory keeps meaning what it says instead of being exceeded by a factor of the batch.
- **Three dtype fallbacks stopped calling themselves.** `((IBackend)this).X(...)` reads as "run the managed
  default", but the class method implicitly implements the interface member, so interface dispatch re-enters the
  override: `WanRmsNormChannel`, `GatedResidualLastDim` and `RopeApplyDecodeStep` each recursed until the stack
  overflowed, taking the process with them, for any input that took the fallback branch. This is a bug class the
  troubleshooting notes already describe and had already cost two earlier instances; these were three more. The
  first two now call a static reference, as that note prescribes, and the third throws — it is a device-position
  op whose interface default is an empty body, so "falling back" would have silently skipped the rotary embedding
  and returned a plausible wrong token rather than failing.
- `WanRmsNormChannel`'s reference states its F32-only contract instead of assuming it. It reads every operand as
  `float*`, so an F16 tensor reaching it would have been reinterpreted bit-for-bit into plausible garbage.

## alpha.85

- **`--backend vulkan` now reaches Vulkan for text generation.** `TextService` derived its device key as "not CPU,
  therefore CUDA", so every non-CPU selector became `cuda:{ordinal}`: a Vulkan engine silently ran its LLM on CUDA,
  and on a machine with no CUDA device it failed with a driver error instead of generating. The key now carries the
  resolved kind as well as the ordinal, and the slot builds its backend through `BackendFactory.Create` like every
  other modality — so an unknown device key reports the selectors that exist rather than naming CUDA and CPU as the
  only choices. Every Vulkan LLM measurement taken before this was a CUDA measurement.
- Same-device generation gating covers every device backend rather than CUDA alone. Two Vulkan generations on one
  card contend for its VRAM exactly as two CUDA ones do, and were running concurrently by default.
- **A registered recipe name is accepted as a family id.** The video planner's own error text advertises the
  drivable families, but several of them — the Wan compat classes — exist only as recipe names, with no catalog
  entry behind them. `-m wan-22-5b` was therefore listed as drivable, accepted, and then refused as family
  'unknown'. A name either registry knows now resolves as itself; a name neither knows still falls through to
  header detection, so an unregistered model keeps reporting what IS drivable.
- Video sparse attention is gated on what the backend can execute rather than on the spelling of the selector. The
  check compared the resolved selector against the string "cuda" before asking the capability, so a backend that
  implements the profile could never be reached through it. A backend without the native kernel is still refused,
  by name, with no dense fallback.
- `hartsy video` and `hartsy world` default to `--backend auto` and no longer describe themselves as CUDA-only.
  Neither ever enforced it: the restriction is per-profile (H3's sparse attention), not per-command.

## alpha.84

- Driving audio refuses the output timing edits that would slide the picture against it: a start trim, a boomerang,
  or an fps other than the native 24. Frame edits reach the frames only, and the soundtrack is trimmed at its end
  alone, so each of those quietly broke the lip sync the feature exists to provide.
- Video: **MiniMax-H3 can be driven by a soundtrack you supply** (`--driving-audio`, `VideoRequest.VideoAudioReference`).
  H3 has no audio-driven mode of its own and its reference audio is a soft exhibit the generated soundtrack can
  drift away from, which is no use when the words have to match. The mechanism that does drive video is the one
  long-form chaining already uses: hold every audio row fixed at the supplied track and let video denoise against
  it, so the model composes a picture around audio it cannot change. The mask is built inside the pipeline, so the
  request-level mask surface keeps its own release gate. A track longer than the clip is trimmed and a shorter one
  zero-padded, which is what the audio VAE's fixed-length encode already did. It is a planned feature
  (`VideoFeatures.DrivingAudio`), so a family that cannot consume it rejects the request during planning instead of
  accepting the option and ignoring it, and a sparse VSA profile — which is T2VA-only — is refused there rather
  than at the execution boundary.
- Video: **Wan-S2V declares `VideoFeatures.DrivingAudio`.** Classifying `VideoRequest.VideoAudioReference` as a
  planned feature reached every family that reads that field, not only H3 — and S2V reads it as the driving speech
  it cannot run without. Undeclared, the generic planner answered `video.feature.unsupported` before construction,
  so the documented speech-to-video path would have failed on its own mandatory input. The declaration is the whole
  fix; the gate itself is unchanged, and a family that does not consume a supplied track still refuses one.
- Measured on a 90-frame 512x288 pair, same seed and prompt, differing only in the locked track: the output audio
  correlates 0.9771 with the driving track through the VAE round-trip; a silence-driven run emits rms 0.00002
  rather than inventing a soundtrack; the two clips diverge at SSIM 0.556; and motion runs 2.36x higher while
  speech plays than after it stops, against 0.80x for the silence control.

## alpha.83

- Video: **an unchunked MiniMax-H3 geometry is no longer charged for its own buffers twice.** Below
  `MinChunkableRows` the forward runs whole, and then `Attention`'s qkv and head-major q/k/v ARE the full-sequence
  buffers the pass-1 term models — the floor added both, counting one allocation twice (about 257 MB at seq 4700).
  Chunked they are genuinely distinct, since kFull/vFull outlive each chunk in flight, so the correction applies
  only when the scratch spans the whole sequence.
- Sizing that scratch by the real sequence rather than a fixed 4,096 rows (alpha.80) made the over-count reachable:
  it added 99 MB at seq 4700, turning a 51 MB margin into a 48 MB deficit and refusing a 90-frame 512x288 clip on
  a 24 GB card that had generated the same geometry minutes earlier. Every calibrated estimate sits above the
  chunking threshold, so none of them covered this branch; the new test pins an unchunked floor against what the
  unchunked forward actually allocates.
- The floor now models attention per implementation rather than assuming one shape. `AttentionSparse` keeps a
  full-sequence gate and the token-major buffer it permutes from alongside qkv and head-major q/k/v, an 8x
  projection peak against the dense path's 6x — and it holds that at ANY length, because `ForwardNamedBlock`
  selects it before testing `seq > chunkRows` and there is no chunked sparse path. Charging the released VSA
  profile a chunk's worth would have approved a near-limit generation that then ran out of VRAM, which is the
  dangerous direction for a pre-flight. The MLP chunks in both modes and is sized separately. A caller that does
  not name the mode gets the larger sparse reservation, so an omitted argument cannot under-estimate; the
  calibrated boundaries name themselves dense, since they measured the fp8 FL2VA path.
- The unchunked peak also reserves the modulated attention/MLP input. `ForwardNamedBlock` disposes it only after
  the call returns, so it is a second `[seq, hidden]` buffer live beside the residual — about 168 MB just below the
  chunking threshold. `Modulate` can emit fp8, but not on a bf16 checkpoint or with `numerics.modulateEmitFp8` off,
  so the reservation is F32. Chunked, the kFull/vFull term covered it incidentally; unchunked nothing did.
- The activation-accounting tests move out of `SyntheticSmoke` into the unit lane. They are arithmetic only — no
  model, GPU, checkpoint or network — but the class trait meant the documented CPU command skipped every one of
  them. This accounting has regressed twice now; quarantining its guards is what let the first one through.

## alpha.82

- Audio: **SheetSage2 transcribes a recording into a score.** With the symbolic half from alpha.81, the model is
  now complete: the MERT2 Conformer encoder, the score decoder, the greedy decode under its grammar, and the
  pass that reads a song longer than the encoder's window in overlapping passes and stitches them onto one
  timeline. `hartsy transcribe -m sheetsage2` returns ABC rather than words.
- This is what makes covering an existing recording possible. Until now YuE2 could only edit scores it had
  written itself, because it has no audio input at all; a transcription gives it a melody it did not compose.
- **Checked end to end, not only stage by stage.** Each stage is pinned against a dump of that stage — the mel
  frontend, the encoder and its learned layer mix, the decode, the grammar mask, the window plan, the stitch and
  the serializer. On top of those, a real recording is run through the whole chain and the resulting score is
  compared as text against what the released implementation produces from the same file. A single wrong note,
  bar line or chord symbol fails it, and it is the only check that can catch an error living in a handover
  rather than in a part.
- The decode is **token-exact** against the reference on both the host and CUDA: reference and port both run
  float32 over the same weights, so exactness is a real criterion rather than a coincidence of dtype. The
  largest logit drift observed consumed an eighth of the margin between the top two candidates.
- One thing the gate had to be built carefully to prove: the checkpoint has largely internalised its own
  grammar, and on a real clip the unmasked argmax is already legal at every step — so a port that dropped the
  mask entirely would still have passed. The mask is therefore pinned on a case constructed to need it, where
  the model would otherwise write a fifth consecutive subbeat shift and the grammar forces a pitch instead.
- Both renderings of the score, with chord symbols and without, come from one decode. The decode is the entire
  cost — the encoder attends over a fixed five-minute window and the token loop is autoregressive — while
  serializing events already in hand is free, and the two renderings are not a substitution apart.
- The weights are cached under the YuE2 repo they ship in, so a machine that already generates with YuE2 does
  not fetch a second copy. They are CC BY-NC 4.0, unlike the engine.

## alpha.81

- Audio: **the SheetSage2 port's symbolic half is complete** — the events-to-ABC serializer and the sliding-window
  stitcher. Between them they turn a decoded token stream into a finished two-voice lead sheet, which is the
  artifact YuE2 edits and re-renders, so this is the half that makes covering an existing recording possible
  rather than only editing scores YuE2 wrote itself.
- The serializer infers the beat grid and meter from the decoded timestamps, spells accidentals against the key
  and the running bar, splits durations that no single ABC token can express, and pads the idle voice so both
  voices carry the same bar count in every parallel chunk. It is checked string-exact against the released
  implementation on ten cases and, separately, against a transcription the model itself produced: the port
  regenerates that score byte for byte from the model's own events.
- **Writing the score without chords is not the same as deleting the chords from one that has them.** A bar
  carrying a chord symbol cannot fold into a multi-bar rest, a chord change inside a held note splits it into
  tied parts, and the two spell rests within a bar differently — four of the ten gated cases differ, six happen
  to coincide. So the mode is a parameter on the serializer and both renderings are produced from one decode,
  rather than one being derived from the other after the fact.
- The stitcher resolves each window's own subbeat and second counts onto the whole clip's timeline and decides
  which window is trusted for each passage. Its seam tolerance is deliberately asymmetric: symmetrising it
  duplicates or drops a bar at every seam. The gate covers fourteen seams across seven clip lengths, including
  events landing exactly on an accept boundary, and was mutation-tested — twenty deliberate breakages, with
  every surviving mutant either turned into a new case or shown to be mathematically equivalent.
- Two reference behaviours are reproduced rather than tidied, each with a note saying so: an unclipped interior
  interpolation in the time map (unreachable in the shipped pipeline, since transcription always asks for the
  full window), and several guards that the surrounding code makes dead. Two are deliberate divergences that
  throw instead: a melody track index outside the two voices, which the reference silently routes to the wrong
  staff, and a beat period below the timestamp resolution, which the reference would expand into millions of
  synthesized beats.

## alpha.80

- Video: **a VRAM posture now reaches video at all.** `--vram-mode` existed only on the image command, so passing
  it to `video` was accepted and ignored rather than rejected; `Vram` was put on the request only in the image
  dispatch branch, so video, music, world, restore, mesh and speech dropped it even when supplied; and `Program`
  read `diagnostics.logLevel` before pushing the `--set` profile, so raising the log level from the command line
  could not work — which is what made the first two hard to see. Every tier request for video was silently
  becoming `Auto`.
- Video: with the tier arriving, `MiniMaxH3ChunkPolicy` honours its `ChunkScale` lever, and a long-form chain
  hands device memory back between segments. The conditioning encoders load ahead of the DiT *within* a segment,
  but across a chain the previous segment leaves weights cached and the next segment's mask-source encode competes
  with them. Gated on `PhaseUnload`, so `VramTier.Auto` stays byte-identical to an unchained run.
  `MiniMaxH3Recipe` declares `PhaseUnload | Chunking` only because both are now wired. The unload frees only the
  pipeline's own DiT tensors: video takes no device gate, so evicting the backend's shared caches could strand a
  concurrent generation on the same device.
- Verified on a 12 GB RTX 3060, which previously ran out of VRAM at the mask-source encode: a 3-segment chain at
  512x288 producing 192 frames and 8.00 s of audio, the unload firing twice, and colour drift of 0.045 per frame.

## alpha.79

- Audio: **the SheetSage2 port continues** — its event codec, which reads a decoded token stream as musical
  events and writes events back as tokens. SheetSage2 transcribes audio into a symbolic score, which is what
  gives YuE2 a melody to cover, so it is the piece that makes editing an existing recording possible rather than
  only editing scores YuE2 wrote itself. Each event keeps both its raw tokens and their read values: the values
  are what a serializer wants, the tokens are what a later window replays verbatim as context, and re-encoding
  from values would not reproduce them.
- The vocabulary, decode grammar and sliding-window plan landed in alpha.78 without a changelog note, so for the
  record: the vocabulary is 31,678 tokens whose ranges are laid out in one pass from offset 260, an id's meaning
  is decided by nothing but which range it falls in, and decoding is greedy argmax over logits masked by the
  grammar — so the grammar does not guard the result, it decides it. All of it is checked against the released
  implementation's own output, dumped by `tests/python-reference/dump_sheetsage2_reference.py`, which needs no
  weights.
- Nothing in the port is reachable yet: the encoder, decoder and the events-to-ABC serializer are still to come,
  along with their parity gates, which do need the checkpoint.

## alpha.78

- Audio: **YuE2 can write its score without rendering it.** The model composes in two passes — an autoregressive
  model writes an ABC score, then a second pass turns that score into sound — and the score is the only editable
  artifact it exposes. Planning it alone takes seconds where the full render takes minutes, so it is now its own
  operation: `IMusicService.PlanScoreAsync` returns the score plus what the context left for audio, and the score
  goes back in through `MusicRequest.Yue2Abc` once edited. `BudgetAsync` answers the budget question on its own,
  for a score a caller already has. Measured 2026-09-16 on a 4090: a 25 s request planned in 14 s and reported a
  25.0 s audio budget; rendering that same score back took 13 s and produced 23.9 s of audio, inside its budget.
- Neither rides `/v1/native/music`, which rejects a result with no audio — they are `POST /v1/native/music/score`
  and `POST /v1/native/music/budget`, and `hartsy music --score-only` writes the score out as a `.abc` file.
  `IMusicRunner` grew optional `PlanScore`/`Budget` members, in the shape `SttRunner.Timed` already uses, so a
  model with nothing symbolic to report simply leaves them null and the service reports it as unsupported.
- `Yue2Pipeline.Generate` now plans through `PlanScore` instead of a second copy of the same sampling call, so a
  score asked for on its own and a score planned on the way to audio cannot drift apart.
- CLI: fixed `hartsy music --help`, which rendered nothing but an error. The `--genre` help text names the
  `[verse]`/`[chorus]` lyric markers, and Spectre.Console read those as markup tags.

## alpha.77

- Video: **MiniMax-H3 long-form chaining is released.** Its real-generation gate passed on 2026-09-16 against the
  fp8 FL2VA base: 3 chained segments at 512x288x141f produced one 345-frame clip whose video and audio both ran
  exactly 14.375 s, with seams at the 2.0th and 14.5th percentile of the clip's own adjacent-frame SSIM
  distribution (0.8856 and 0.9202 against a 0.8627 minimum and 0.9492 median) — a busier-than-average step rather
  than a cut. Verified the same way through the CLI and the native HTTP API. Guides and AV denoise masks stay
  release-blocked: chaining builds its masks inside the pipeline and never sets those request objects, so it
  carries its own gate rather than theirs.
- Each segment VAE-encodes a full-length source clip on top of its own generation, so a chain needs more headroom
  than a single generation of the same segment length — 23.4 GB peak at 512x288x141f.

## alpha.76

- Video: **MiniMax-H3 can generate past one denoise, as a chain of segments.** `VideoRequest.ChainTotalFrames`
  splits a longer target across successive generations; each one after the first copies the previous segment's tail
  into its leading rows and masks those rows out of denoising, so they carry that segment's motion and soundtrack
  phase into the new frames' attention context. The protected head is context rather than output — it is dropped at
  assembly, which leaves one continuous sequence with no duplicated frames and no seam to blend. Lengths stay on
  H3's `17k+5` grid (`MiniMaxH3ChainPlanner`) so a protected head always covers whole latent tokens; a head ending
  inside a token would hold part of it fixed while denoising the rest, which the sampler's row masks cannot express.
  Trim and boomerang apply once, to the whole video. Exposed as `--chain-frames` / `--chain-seconds` /
  `--chain-context-frames` on the CLI and as `chainTotalFrames` / `chainContextFrames` over the native video API.
  **Release-gated**: `video.h3_expansion.release_blocked` refuses it in published builds until its
  operator-provided real-generation and output-inspection gate passes.
- Video: `VideoDenoiseMask` gained `MaskFrameValues` (one spatially-uniform value per latent frame, the video mirror
  of `AudioDenoiseMask.Values`) and `SourceFrames` (raw frames instead of an encoded clip). A binary temporal
  boundary cannot survive `MaskVideo`'s lossy codec — mask values quantize upward, so a smeared black partly
  denoises rows meant to be preserved — and a caller holding decoded frames no longer pays an encode/decode round
  trip to hand them back.

## alpha.75

- Image: **Kohya SD1.5/SDXL LoRAs that use LDM block names now load.** `LoraFormatDetector` recognized only the
  diffusers spellings (`lora_unet_down_blocks_` / `up_blocks_` / `mid_block_`), so a file whose UNet keys are
  `lora_unet_input_blocks_` / `output_blocks_` / `middle_block_` — what sd-scripts emits, and what the large
  majority of community SDXL LoRAs on CivitAI ship — fell through every arm and was refused at load as
  "Could not detect LoRA format". Detection alone was not enough: the loaded UNet dict is diffusers-named, so
  those keys are mapped through `SdxlCheckpointConverter`/`Sd15CheckpointConverter.ConvertUNetKey`, the same
  LDM→diffusers map the checkpoint itself goes through. The text-encoder halves (`lora_te1_`/`lora_te2_`) were
  already correct and are untouched. Verified against a stock CivitAI SDXL LoRA: all 986 modules (722 UNet,
  72 CLIP-L, 192 CLIP-G) resolve to keys the converted checkpoint holds, and the merged image is a large
  visible change from the base (SSIM 0.34) while strength 0 is inert to the pixel (SSIM 1.0).
- Image: LoCon/conv LoRAs for those families reach the resnets, whose LDM sub-keys (`in_layers`, `out_layers`,
  `emb_layers`, `skip_connection`) are compound names the underscore→dot pass used to split into keys that match
  nothing — a partial merge that no error reports, since the attention layers still merge.

## alpha.74

- Audio: YuE2 songs can run to **900 seconds**, up from a hard 360. Nothing in the checkpoint required the old
  cap — 25 tokens a second against a 24,576-token context holds about 980 s minus the prefix — but three
  separate constants enforced it, and the duration knob was inert above 360 s regardless because the release's
  9,000-token sampler preset won every `Math.Min` against it. `Yue2Protocol` now separates the release's
  default from the ceiling it will accept. Measured at 479.9 s of real music from extended lyrics and 843.1 s
  against a forced budget. Raising the ceiling does not make the model write longer songs — length is decided
  by the lyrics and the score — it stops one that wants to be longer from being cut off.
- Audio: a prompt that leaves less context than the requested duration now **shortens the song and says so**
  instead of throwing. The old check raised after the score had already been planned, so an over-long lyric
  cost ~20 s of planning and then failed; the budget is now fitted to whatever the prefix leaves (the longer of
  the two branches under guidance) and travels back as `Yue2Result.BudgetSeconds` and `meta.budgetSeconds`.
  This is a deliberate divergence from the release, whose sampler refuses the case outright with "no implicit
  truncation" — our caller-facing knob is a duration ceiling rather than a token count. A request that already
  fits is unaffected token for token, which the identical 360 s output confirms.
- Audio: the Oobleck VAE decodes in **bounded tiles**, as the release does by default and we did not — 1,024
  frames of core with a 16-frame halo. Every core sample carries its whole input support inside its own tile,
  so there is no crossfade and the samples are identical to a whole-song decode; what changes is that peak
  activation memory is set by the tile rather than by the song. Peak VRAM is 18.4 GB at 360 s and flat at
  ~22.0 GB past 480 s.
- Audio: **`OobleckConfig.DecodedLength`** — a decode is `frames × HopLength` only when every stride is even.
  A transpose conv here runs `k = 2s, padding = ceil(s/2)`, which emits `sL` for an even stride but `sL − 1`
  for an odd one, and YuE2's stride of 5 sits under a further 64× of upsampling: every YuE2 decode is exactly
  64 samples short of the round multiple. Tiling that assumed the multiple overran its last tile. The older
  all-even presets (Stable Audio Open, ACE-Step 1.5) are unaffected, which is why an all-even test config could
  not have caught this — the tiling tests now carry an odd-stride config for that reason.
- Audio: the score planner's repetition-penalty window is reachable from a request
  (`MusicRequest.Yue2AbcPenaltyWindow`), matching ComfyUI's PR 16293. Its 1-100 validation is unchanged: that
  mirrors the release's own bound, which their node does not re-check.
- Not ported from ComfyUI PR 16293: the `decode_buffer`/`rotary_buffers` work is CUDA-graph capture address
  stability, and graph decode is a measured 23% regression on YuE2.
- Perf note: alpha.74 measures 98.7 / 98.8 / 98.9 s on the standard 360 s request against 98.6 / 99.0 s for an
  unmodified alpha.73 tree in the same session — tiling costs nothing. The 97.1 s recorded for alpha.73 was
  taken in an earlier session; between-session spread is ~2% and a 1-2 s difference cannot be attributed to
  code across one.
- Tests: `Yue2BudgetTests` covers the duration/context arithmetic and the multi-chunk split with no checkpoint
  or GPU; `OobleckVaeTests` gains tiled-vs-whole-song equality (which is what validates the halo) across even
  and odd strides, and the exact-length rule.

## alpha.73

- CUDA: `ApplyRopeSingle` accepts an F16 activation against an F32 cos/sin table, matching the asymmetric
  contract `ApplyRope` already honours. `dit_rope_f16` is indexing-identical to its F32 twin and declares the
  table as `const float*`, so the earlier crash chasing this was an F16 TABLE being over-read by exactly 2x,
  not a layout mismatch and not a missing kernel. It also now rejects a non-rank-4 tensor: head count and head
  dim are read from `Shape[2]`/`Shape[3]`, so a head-major input silently rotated the wrong rows.
- Audio: YuE2's acoustic attention block runs at the key/value buffers' dtype end to end — the input norm, the
  q/k/v projections, the per-head qk-norm, RoPE and the permutes. Its weights are BF16, so an F32 activation
  was cast down on every projection, the same inefficiency alpha.69 fixed in the feed-forward; RoPE was the
  one stage blocking it. Q now reaches the fused attention entry already at its dtype, removing the per-layer
  cast alpha.71 added. The grouped K/V step back to F32 only for `KvCacheAppend`, which narrows from F32 and
  has no F16-source form. The acoustic pass goes 23.5s to 22.4s and a 236-second song 98.2s to 97.1s.
- Tests: `RopeSingleF16Tests` covers F16-vs-F32 rope at full and partial rotary, and pins both halves of the
  contract — an F16 table and a non-rank-4 input must both be rejected.

## alpha.72

- CUDA: causal PREFILL no longer runs on the decode-tuned flash kernel. `GenericTransformer` issues both
  shapes through `IBackend.FlashAttention`, but prefill is thousands of query rows rather than one against a
  long cache, and that kernel is tuned for the latter: measured at 4,096 tokens (D=128, 16q/8kv, RTX 4090)
  64.8 ms a layer against 9.5 ms through cuDNN's fused engine, and 6.74 s across 28 layers on YuE2's
  8,664-token prefill. The plain causal prefill now takes the fused engine, carrying the causal rule as an
  additive bias built on the device by a new `lm_causal_bias_mask_f32` kernel — a host fill is 75M floats at
  that length (~0.2-0.3 s), more than the attention it accelerates. Sliding windows fold into the mask; a
  soft-cap, attention sink or ALiBi has no bias-shaped equivalent and keeps the general kernel, as does a
  non-tight K/V buffer or a head dim cuDNN cannot take, and any cuDNN failure falls through.
- Audio: a 236-second YuE2 song generates in 98.2s, from 107.2s — 0.416 s per second of audio against the
  reference implementation's 0.473 on the same lyrics. The acoustic stage drops 29.8s to 23.5s, but that is
  the per-chunk AR prefill which the progress buckets count there: **the acoustic transformer itself is
  unchanged**. The semantic pass goes 105.0 to 108.6 tok/s and the ABC planner 20.0s to 19.3s, both from
  their own prefixes' prefill.
- Tests: `CudaFlashAttentionTests.CausalPrefill_MatchesCpuReference` checks the fused path against
  `AttentionReference` at 512 tokens with and without a query offset, GQA and MHA, and with a sliding window
  — the pre-existing prefill case runs at 7 keys with no offset, which cannot reach the path and whose oracle
  (SDPA plus an explicit mask) is what the path itself uses. It also asserts cuDNN actually engaged, since a
  gate that silently fell back would satisfy every numeric assertion without running the new code.
- Kernels: `Kernels/lm/build.sh` compiles the two hand-tuned kernels but no longer INSTALLS them unless
  `--install-tuned` is passed. `lm_f32` and `mul_mat_vec_q6k_q8_1` are LLM-decode hot paths held at a tuned
  register allocation, and a routine rebuild to add an unrelated kernel to that domain silently overwrote
  both with ~1,550 lines of different codegen — caught here only because the artifacts showed up in
  `git status`. They still compile, so the drift check still covers them.
- Tests: `CudaKernelDriftTests` rebuilds all 9 kernel domains from source and compares against the committed
  PTX. `dotnet build` never compiles a `.cu` — MSBuild only copies the artifacts and there is no nvrtc
  fallback in the runtime — so an edited kernel whose PTX was not regenerated keeps running the old code
  silently, which shipped 8 stale kernels once already. Skips rather than fails when the local toolchain
  cannot reproduce the artifacts.

## alpha.71

- Audio: YuE2's acoustic pass holds one chunk's attention keys and values for the whole ODE solve instead of
  rebuilding them per velocity evaluation. The AR prefix's K/V are the same on all 64 evaluations of a 32-step
  midpoint solve, but were being re-concatenated and re-widened to full heads every time — ~8,700 identical rows
  a layer. They are now written once per chunk, each evaluation appends only its own rows at the tail, and the
  buffers are stored at F16 where the backend has F16 kernels, which lets the attention take cuDNN's native-F16
  entry and cast nothing (the F32 route re-narrowed the whole key and value on every call).
- Audio: YuE2's acoustic RoPE tables are built once per chunk rather than per velocity evaluation. They depend
  only on the chunk's token span, and building them is a host loop over `tokens * head_dim/2` Pow/Cos/Sin triples
  — 378,000 of them per evaluation, 64 times a chunk, for identical values.
- Together the acoustic pass goes 32.0s to 29.8s on a 236-second song and the whole generate 108.9s to 107.2s.
  The six parity gates still pass, including the acoustic velocity and the full 32-step solve against the
  reference, so the F16 key/value storage is within the gates' bar.

## alpha.70

- Audio: YuE2's autoregressive passes project and sample only the contiguous id window their phase can draw from.
  Every id outside a phase's span is masked to -inf before sampling, so the semantic pass was spending a
  184,704-wide head GEMM, a 739 KB device-to-host copy and ~8 full-vocabulary host scans per token to choose among
  32,769 candidates. The semantic head is now an owned BF16 row-slice (134 MB) covering `[MusicEnd, CodecOffset +
  CodecSize)` and the sampler works in window coordinates, so all three costs shrink 5.6x. Measured on a
  236-second song: the semantic pass goes 90.1 to 105.3 tok/s (65.5s to 56.1s) and the whole generate 119.0s to
  108.9s. The ABC phase keeps the full projection — its span is not contiguous — but shortens its read-back.
- Audio: the six YuE2 parity gates now check the semantic phase over its head window rather than the full
  vocabulary, which is every logit that phase's sampler can reach; A3's greedy tokens still match the reference.

### Measured and rejected

- CUDA-graph decode for YuE2's AR is a **23% regression** (105.3 to 81.1 tok/s) and was not kept. Capture succeeds,
  but the win it targets is not there: eager decode's ~500 per-token kernel launches already overlap with GPU
  execution, so collapsing them buys nothing, while the device-position graph ops are slower than the eager
  kernels for this geometry. The 6.5 us/launch cost measured from a tiny-kernel loop only bites when the GPU is
  idle — it is not a per-token tax on a decode step that keeps the device busy.
- An F16 KV cache for the AR changes the per-token body cost by 0.8% (9.076 to 9.001 ms), so YuE2's attention is
  not KV-bandwidth-bound at song-length contexts and `KvCaches.F16Enabled` is not a lever here.

## alpha.69

- Audio: YuE2's acoustic feed-forward runs its activations at F16 where the backend has F16 kernels. Its weights
  were already BF16, so an F32 activation was being cast down on every one of the three projections; at F16 the
  GEMM stays 16-bit end to end. Worth ~1.2s on a full-length song's acoustic pass (33.0s to 31.8s) — less than an
  isolated op benchmark predicted, which is worth recording: op timings that reuse warm tensors overstate what the
  same change does in situ. Parity improved rather than degraded (velocity corr 0.999809 to 0.999825, nrms 1.96%
  to 1.88%), as expected — the release runs the whole transformer in bfloat16, so this moves toward its numerics.
  The residual stays F32; it accumulates over 28 layers and rejoining it costs one 0.08 ms cast.

## alpha.68

- Audio: YuE2's acoustic stack ran its attention through `IBackend.FlashAttention`, whose kernel is tuned for LLM
  decode — one query row against a long cache. The acoustic pass is the opposite shape: every frame of the song
  is a query row, attending bidirectionally over the whole AR prefix. Routing it through
  `ScaledDotProductAttention`, which reaches cuDNN's fused engine, is **26x** on that op (74.6 ms a layer to 2.8 ms
  at 2308 frames on a 4090), and attention was ~89% of the stack. A full-length song's acoustic pass drops from
  ~816s to 33s and a 3.9-minute song end to end from **17.8 minutes to 119 seconds**. The fused engine is MHA-only,
  so the grouped KV is widened to full heads first; that copy costs ~0.2% of what it buys. All six parity gates
  hold (velocity corr 0.999809, 32-step solve 0.999939 — unchanged to five decimal places), which is expected:
  F16 attention ingest matches the release, which runs the whole transformer in bfloat16.

## alpha.67

- Audio: YuE2 — lyrics- and style-conditioned song generation at 48 kHz stereo, up to six minutes. Despite the
  name it shares no architecture with YuE v1: a Qwen3-geometry autoregressive LM plans an editable ABC score and
  emits one semantic codec token per 25 Hz frame, then a second stack of identical geometry but its own weights
  flow-matches 64-channel acoustic latents while attending to the first model's per-layer KV cache as a
  bidirectional prefix, and an Oobleck VAE decodes those to audio. Loads the Comfy-Org single-file repack, which
  carries both stacks, the VAE and the tokenizer in one checkpoint, so there is no subfolder fetching and no side
  assets. Weights are CC BY-NC 4.0. The `int8_convrot` repack is refused by name rather than silently falling back
  to a different precision than the caller asked for.
- Engine: `MusicRequest` gains YuE2's own knobs — `Yue2Cot` (planning mode), `Yue2Abc` (render a score verbatim,
  skipping the planning pass), a separate sampler block for the score planner, which samples far cooler than the
  semantic pass (`Yue2AbcTemperature`/`TopP`/`TopK`/`RepetitionPenalty`/`MaxTokens`), plus `Yue2PenaltyWindow` and
  `Yue2MinTokens`.
- CLI: `hartsy music` was building its request from four fields, so `--steps`, `--cfg-scale`, `--temperature`,
  `--top-k`, `--top-p` and `--repetition-penalty` were unreachable from the command line for *every* music model.
  All six now reach the engine, alongside the new `--cot`, `--abc` (a file path or inline notation),
  `--abc-temperature`, `--abc-top-p`, `--abc-top-k`, `--abc-repetition-penalty`, `--abc-max-tokens`,
  `--penalty-window` and `--min-tokens`.
- API: the audio endpoints serialise `AudioResult.Meta`, which the response shape already promised to mirror but
  silently dropped. Music generations carry the model, seed and channel count, and YuE2 adds its planned score and
  two truncation flags — a request for a duration the token budget cannot cover previously returned a song that
  stopped mid-phrase with the only warning in a server-side log.

## alpha.66

- Images: Qwen-Image-Edit accepts more than one reference image. `ImageRequest.ReferenceImages` carries the extra
  references, presented after `Img2Img.InitImage`, so a prompt that says "the first image" / "the second image" is
  addressing that order; up to three are consumed, matching the slots the edit-plus template was trained with. The
  Qwen-Image recipe now also builds the text encoder's Qwen2.5-VL vision tower and conditions on the full
  `Picture N: <|vision_start|>…<|vision_end|>` edit template, and a 2511 checkpoint's `index_timestep_zero`
  reference method is honoured instead of being left off. References keep their own aspect-preserving rescales
  (~1 MP to the VAE, ~384² to the vision tower) rather than being squashed to the output size, which changes the
  single-reference path too. Reference images with an explicit denoise mode are refused by name instead of being
  silently dropped.
- CLI: `hartsy image --init-image-mode denoise|reference|auto` selects how an init image is consumed, and a
  repeatable `--reference-image` adds the extra edit references.

## alpha.65

- Engine: `ImageRequest.RemoveBackground` cuts the subject out of a finished generation. RMBG-1.4's matte lands
  in the new `ImageResult.Alpha` plane and the generated RGB is left byte-for-byte untouched, so a consumer
  composites the partial-coverage edge exactly once instead of receiving pixels already blended toward a
  background. The stage runs last — after the refiner and segment-refinement passes, over the pixels the caller
  actually receives — and is family-independent, so no recipe has to declare it. It shares `VisionService`'s
  weight cache rather than loading a second copy of the net, and it is not gated on denoise strength: a
  strength-0 img2img request still gets its cutout.
- CLI/API: `hartsy image --remove-background`, and `removeBackground` on `/v1/native/images`. Both encode
  color-type-6 (RGBA) PNGs when a matte is present and color-type-2 otherwise.

## alpha.64

- Audio: MiniMax Music 3's one-time GGUF quant caches (language model and depth decoder) are written beside the
  weights they derive from, under `AudioModelCache.CacheRoot`, instead of a hardcoded `~/.cache/hartsyinference/`.
  They are multi-gigabyte files and were landing on the boot drive regardless of how the cache was relocated;
  they now follow the same root resolution the weights themselves use (`paths.modelCacheRoot`, else
  `paths.modelsRoot/audio`, else the user cache directory). An existing cache re-quantizes once on the next
  `:q8`/`:q4` run, after which the old directory can be deleted.

## alpha.63

- Krea 2: the recipe honors the `ImageRequest.Components` Qwen text-encoder and VAE picks instead of always
  loading its own pinned side models, resolving them through `ModelFileLocator.Require` so an unresolvable pick
  is refused by name. The pinned fallbacks auto-download when absent, the contract `LtxVideo2Recipe` already
  uses and what SwarmUI's own `RequireClipModel`/`DoVaeLoader` do. First of the `TODO(E-IMG-4)` recipes closed.
- Models: `SideModels.Qwen3VL_4B` moves to the flat `text_encoders/qwen3vl_4b.safetensors` that SwarmUI's Comfy
  backend downloads under the identical SHA-256, so the two share one file instead of fetching 5 GB twice. The
  `Krea2/` subdir it previously used matched nothing on either side.
- Engine: `RecipeContext.Cancel` carries the originating request's cancellation token into recipe construction, and
  Krea 2 honors it when acquiring a side model. The fallback download it newly enables can transfer several
  gigabytes, so an HTTP client disconnect would otherwise have left it running with the request already gone.
- Models: `ModelAsset.LegacyTargetNames` records the names an asset was saved under before its canonical name
  changed, and `ModelDownloader.TargetPath` resolves to an existing legacy file when the canonical one is absent.
  Without it, renaming a shared asset costs every upgrading install a multi-gigabyte re-download and makes the
  recipes that resolve it through the strict non-downloading overload (Mage-Flow shares Krea 2's encoder) fail as
  though the file were missing. A fresh install still downloads to the canonical name.

## alpha.62

- Diffusion: Z-Image generates again. Since alpha.42 every generation threw a `NullReferenceException` on its
  first denoise step: the new per-step preview snapshotted the transformer's fixed CUDA-graph latent, which
  only exists on the step-graph route, and that route is default-off for Z-Image. The preview now unpatchifies
  the loop's own packed tokens through the backend op, so it works on both routes and keeps the tokens
  device-resident instead of draining them to the host every step. `SnapshotGraphLatent` now names the unmet
  precondition instead of dereferencing null.

## alpha.61

- Prompting: SwarmUI's 2026-09-01 parser update hands every backend Swarm tags in place of Comfy-native prompt
  syntax — `(word:1.5)` arrives as `<weight[1.5]:word>`, `[a|b]` as `<alternate:a,b>`, `[a:b:N]` as
  `<fromto[N]:a,b>`. The engine read those as prose, so SD1.5/SDXL prompt weighting was silently inert and the
  tag text was tokenized into the conditioning. `PromptTagFlattening` converts the weight tag back to the
  `(text:N)` grammar `PromptWeighting` implements and collapses `alternate`/`fromto` to their step-0 value,
  running in `ImagesService`/`VideoService`/`MusicService` ahead of every pipeline and of region/segment
  parsing; every other tag passes through byte-for-byte.
- Diffusion: real per-step prompt scheduling. SD1.5 and SDXL declare `ImageFeatures.PromptScheduling` and keep
  the scheduling tags raw, so `PromptTagScheduling` and `WeightedConditioning.Build{Single,Dual}ClipScheduled`
  build a multi-variant `ConditioningSchedule` — one encode per distinct (positive, negative) variant pair that
  occurs, not the full cross product. `fromto` thresholds are a 1:1 port of the reference `SwarmText.py` float
  comparison, so a fraction lands on the same step and a `when` above 1 stays an absolute step index. The old
  `PromptScheduling` (Comfy bracket grammar, never wired into a pipeline) is removed.
- Diffusion: SDXL's pooled/ADM conditioning follows the prompt schedule. The denoise loop switched hidden states
  per step but kept passing the single pooled encode to every UNet and ControlNet call, pairing a later variant's
  hidden states with variant 0's ADM vector; `ConditioningSchedule.PooledVariants` now carries one pooled tensor
  per variant. Null keeps the single encode, which is the unscheduled path and the only option for SD1.5.
- Prompting: `<weight[N]:text>` collapses to its inner text for architectures without per-token weighting rather
  than becoming `(text:N)`. The parens form is only meaningful where a tokenizer applies the weight; emitting it
  to an LLM-conditioned DiT handed the literal digits to Qwen/T5/Gemma as prose. `ImageFeatures.PromptWeighting`
  gates it and only SDXL/SD1.5 declare it; video and music strip unconditionally. The weight itself is still
  unimplemented for LLM encoders, so it is dropped rather than applied for them.
- Diffusion: a scheduled conditioning build that fails partway — an OOM on the third variant, say — disposes the
  tensors it already encoded instead of leaking them, since the schedule that would own them is never returned.
- Diffusion: `WeightedConditioning.HasWeightingSyntax` no longer treats a bare `[` as weighting syntax.
  Brackets carry no grammar now, and counting them put bracket-bearing prose on the schedule path — which
  forfeits SD1.5's fused Euler loop and made any non-default sampler selection fail outright.

## alpha.59

- Benchmarks: a standalone `hartsy-bench` runner produces reproducible community evidence from frozen,
  hash-pinned text and image workloads. Each session runs in its own process with immutable attempts and a
  resumable journal, and every trial retains its raw timings, native token trace and saved output. `validate`
  checks protocol and evidence rather than trusting submitted numbers, `export`/`extract` move data-only
  bundles carrying a full hash inventory, and `publish` builds the static explorer from reviewed evidence
  only. Trusted workflows validate results PRs without executing contributor code. `--cache` resolves as
  `--cache`, then `$HARTSY_BENCH_CACHE`, then `~/.cache/hartsy-bench`; the content-addressed
  `<cache>/<sha256>/<file>` layout lets a checkpoint already stored elsewhere be hard-linked in under its
  pinned hash instead of downloaded again.
- Engine: optional generation diagnostics. `EngineOptions.Diagnostics` is null by default, so
  `StartDiagnostics` returns 0 and `ReportDiagnostic` returns on its first comparison, and the per-token hook
  is only built when an observer is active. An observer that throws is disabled once rather than being allowed
  to alter inference.
- Core: a knob profile that pins a nullable knob to `null` now keeps that null instead of falling through to
  the machine's `HARTSY_*` override. This also corrects `--profile reference`, which pins
  `numerics.cfgInterval` and `numerics.sagePv` to null and until now silently inherited either variable from
  the environment, defeating the profile's stated purpose. Value-type knobs are unaffected.

## alpha.58

- Restore: SeedVR2 runs on the CPU backend. Its kernels are F32-only (`Linear` casts a half weight on the fly but
  not the bias, and the VAE's 3-D convs do not cast at all), so the fp16 checkpoints stopped at the first GEMM.
  `RestoreService` now casts the DiT and VAE tensors to F32 once at load when the backend is CPU (about 13.5 GB for
  the 3B model) and releases the copies with the pipeline; CUDA still consumes the half weights directly.

## alpha.57

- Vision: Real-ESRGAN x2plus produced 4× output. BasicSR's 2× RRDBNet is the same ×4 network fed a 2× pixel-unshuffled
  12-channel input, and `RealEsrganConverter.InferConfig` read the factor from the presence of `conv_up2`, which every
  checkpoint has. The factor now comes from `conv_first`'s input channels, `UpscalePipeline` unshuffles on the host
  before tiling (odd edges replicated and cropped back), and the network always runs both upsample stages.
- Restore: the SeedVR2 catalog's positive embedding is now the upstream `ByteDance-Seed/SeedVR2-3B/pos_emb.pt` (public,
  hash-pinned) instead of a private Hartsy-hosted safetensors that answered 401, so a fresh install's first-use download
  completes. `RestoreService` reads that bare-tensor pickle directly; a `*emb*.safetensors` sibling still works.
- Catalog: `briaai/RMBG-1.4` is no longer gated on HuggingFace; the entry's comment says so and names a byte-identical
  ungated mirror in case that changes.

## alpha.56

- Vision: `VisionMode.Upscale` runs the Real-ESRGAN generator that `HartsyInference.Vision/Upscale` has carried
  without a caller. Three catalog ids fetch the official BasicSR checkpoints on first use — `real-esrgan-x4plus`,
  `real-esrgan-x2plus`, `real-esrgan-anime6b` (`Models/Vision/Upscale/`) — and the request's new
  `TargetWidth`/`TargetHeight` fit the result: enough passes to cover the target (at most two), then a bicubic
  downsize, never a stretch past what the last pass produced (`UpscalePlan`). Tiled at 256 px input.
- Vision: `ImageData.Alpha`, an optional 8-bit straight coverage plane. `VisionMode.BackgroundRemoval` now fills
  it with the RMBG-1.4 matte beside the gray composite the image→3D preprocessors keep reading, so a caller can
  build a real cutout. `PngEncoder.Encode(ImageData)` writes colour type 6 when the plane is present; the
  `/v1/native/vision` route and `hartsy vision --mode removebg` both return RGBA for it.
- CLI: `hartsy vision --mode upscale --width/--height`; the mode is inferred from an `esrgan`/`upscale` model id.

## alpha.55

- Wake: a satellite may declare `"width": 1` in its `hello` and send G.711 µ-law, one byte a sample instead of
  two. Not about a link's average throughput — a device on a marginal link loses audio in stalls, where one
  dropped packet costs a retransmission timeout of about a second and everything queued behind it is dropped.
  A send buffer covers a fixed number of bytes, so halving the bytes doubles the seconds it covers. µ-law
  rather than 8-bit linear because a satellite microphone can sit at a few hundred counts out of 32768, which
  linear truncation would quantize to two or three levels. Width 2 remains the default and is unchanged.

## alpha.54

- Wake: `WakeServiceOptions.HostHandlesTurns`, and a settable `WakeService.HostHandlesTurns` to match, put
  `"handled":true` on every `transcript` frame. A satellite reads it as "do not answer this yourself". Without
  it a host that answers turns has no way to tell a device to stand down in time: `Detected` is raised after
  the transcript frame is written, so by the time a subscriber could send anything the device has already
  started its own assistant call, and two replies end up sharing one audio ring. Off by default; the frame is
  byte-identical to before when it is off.
- Wake: `WakeService.BeginAudio` returns a `WakeAudioStream` — one spoken reply, written in as many pieces as
  it arrives in. A reply synthesized sentence by sentence used to go out as one `SendAudioAsync` call per
  sentence, and each call numbered its frames from zero and marked its last one final, so the device read every
  sentence after the first as a new reply, reset its playback ring and cut off the one before it. The stream
  also holds back a piece's ragged tail rather than sending a frame with an odd number of bytes, and closes
  itself on the way out of an abandoned turn. `SendAudioAsync` still works, and is now one of these.

## [Unreleased]

### Added
- **Spoken audio can travel on the wake socket.** `WakeFrameCodec.WriteAsync` gained an overload that writes a
  header plus raw bytes — the same header-then-payload shape a satellite has always used to send audio, now in
  the other direction — and `WakeService.SendAudioAsync` pushes a reply through it in 40 ms frames.

  Speech reaches a device over HTTP today, which costs it a second connection and a second protocol per turn.
  The socket it is already holding open can carry it.

  The server does the pacing, which is the part that is not obvious. A device paces an HTTP body by withholding
  TCP acknowledgements, but doing that here would also stall the `ping` and `detection` frames queued behind
  the audio on the same connection, and a satellite that stops answering pings is dropped after twenty seconds.
  So the server writes a little ahead of real time — 400 ms — and never further.

  Header and payload go out under one lock and in one pair of writes. A header promising bytes that never
  arrive desynchronizes the stream permanently, because the reader then takes the next frame's header as
  payload; that exact failure has been seen on this protocol in the other direction and it cost a night to
  find. The round trip is tested, not just the bytes.
- **`BackendFactory.ProbeCuda`** runs a real matmul on the GPU and checks the answer, instead of asking whether
  a GPU exists. `CudaContext.IsAvailable` answers the second question, and everything between it and a working
  backend is untested by it: kernels built for another architecture, a PTX directory that did not ship, a card
  with no free memory, a driver and toolkit that disagree. Each of those says available and then throws on the
  first real operation, in the middle of somebody's request.

  Found on the first run, on the machine it was written on: `IsAvailable` returns true, `Resolve("auto")`
  returns `cuda`, and the shipped kernels target sm_80 against an sm_75 card, so the driver's JIT refuses every
  one of them. `ResolveProbed("auto")` returns `cpu` there, with the reason.

  A wrong answer counts as a failure too, not just a throw — a GPU that computes the wrong thing is worse than
  one that stops, because nothing downstream notices. Cached after the first call, and opt-in: `Resolve` is
  unchanged and stays cheap.
- **`status` frames on the wake socket.** A voice turn is several seconds spread across transcription,
  generation and synthesis, and from a satellite's side all of it looks the same: it sent audio and nothing has
  come back. So its light stayed on one colour for the whole wait, and a user with no screen could not tell
  "still working" from "did not hear you" — and would repeat themselves into the middle of a reply.

  `WakeService` now emits `captured` (the speaker can stop talking) and `transcribing` as it reaches them, plus
  `error` and `done`, and `SendStatusAsync` is public so whoever owns the turn after the transcript leaves can
  send `thinking` and `speaking` too. `WakeStatus` names the states and builds the frame's data object, so the
  wire contract a C++ satellite parses is one function with a test on its exact bytes rather than an
  interpolation at four call sites.

  Advisory throughout: a device that does not know a state ignores it, an unreachable device is not an error,
  and no turn fails because a light could not be updated.
- **Piper streams a sentence at a time.** Piper is a whole-utterance model — nothing comes out until every
  phoneme of the text has been through the decoder — so a spoken reply used to begin only once all of it had
  been synthesized. It now implements `IStreamingTtsRunner` by splitting on sentence boundaries and yielding
  one `AudioChunk` per sentence, so the first one can be played while the rest is still being made.

  Measured on the four-sentence passage from the latency audit, 8-core CPU: **8.315 s** to synthesize whole,
  **1.797 s** to the first sentence, 22% of the wait. Total synthesis also fell to 4.058 s, because VITS cost
  grows faster than linearly in sequence length — so this is a throughput win as well as a latency one, which
  was not the point but is not unwelcome.

  The audio is not sample-identical to the whole-text call: each sentence gets its own prosody contour. The
  non-streaming `Synthesize` path is unchanged and still passes the text through in one piece.
- **`SentenceSplitter`** (Audio, `Frontends`) cuts a passage into sentences for exactly that. Cutting where a
  sentence does not end is the expensive mistake — the halves are voiced separately and the seam is audible —
  so it declines to cut on abbreviations, initials, decimal points, and anything not followed by something that
  looks like a new sentence, and it merges a fragment shorter than 24 characters into what follows it.
- **End-of-speech detection on the wake path.** After a wake word fired, the service waited a fixed three
  seconds and then transcribed the preceding eight — so a question longer than three seconds was cut off
  mid-word, and a two-word command still cost the full three seconds. Measured on a real satellite before this
  change: "Hey Jarvis, can you tell me what the weather is going to be like this afternoon and whether I should
  bring an umbrella…" came back as "…like this after".

  `SileroVadStream` has been implemented and parity-tested since it was written but was wired to nothing. It
  now runs per session on the wake worker's own thread and backend, alongside the detection pipeline and the
  denoiser, and `WakeService` ends the utterance on `EndOfSpeechSilenceMs` (500 ms) of silence instead of a
  fixed wait — capped at `UtteranceSeconds`, so someone who never stops talking still gets an answer.
  `UseEndOfSpeech` turns it off; without VAD weights installed the old fixed wait is still the fallback.
- **`WakeEvent.Command`, and `command` on the device frame.** Transcription covers the wake word and the
  command together, so the transcript begins with the words that woke the device — "Hey Jarvis, what time is
  it?" — and a small model will answer the greeting instead of the question. The engine knows which head fired,
  so it now reports the command separately. The full transcript is still sent; nothing is silently edited.
- **`OnnxWeightLoader.SubgraphConstants`** reads the float `Constant` values of a nested graph, and the ONNX
  parser now walks node attributes for subgraphs. Silero VAD keeps its weights this way — inside the
  `then_branch` of an `If` on sample rate rather than as graph initializers — so an initializer-only reader
  finds nothing in the file at all. `WakeModelSet` can therefore load `vad/silero_vad.onnx` straight from its
  canonical MIT source with no conversion step and nobody hosting a repacked copy; `vad/silero_vad_16k.safetensors`
  is still read when present. Verified by scoring both formats through the service's own loader and requiring
  identical probabilities.
- **`tools/convert_silero_onnx.py`** emits the safetensors form and, with `--verify`, checks the export against
  onnxruntime end to end (1.25e-6 max abs over 343 chunks).


### Fixed
- **End-of-speech waited for the silence window twice.** The utterance clock was driven by
  `SileroVadStream.InSpeech`, which describes a segment rather than a moment: it stays true for the stream's
  own `minSilenceMs` after the speaker stops. `WakeService` then waited its own `EndOfSpeechSilenceMs` on top,
  so with both set to 500 ms every command paid a full second of silence before transcription even started.
  The stream now exposes `LastChunkWasSpeech`, the per-chunk verdict with hysteresis applied, and the session
  times silence from that. Measured on a real satellite before this change: detection to transcript 3.95 s on
  "what time is it?", of which a second was this double wait.
- **A question longer than 6.5 s was still truncated.** The end-of-speech wait was capped at
  `UtteranceSeconds - LeadInSeconds`, which quietly spent the lead-in allowance out of the speaker's time
  rather than out of the buffer it actually comes from. The cap is now the whole of `UtteranceSeconds`, and
  that default moves from 8 s to 12 s — a spoken question runs longer than it reads, and the capture buffer
  holds fifteen. Measured before: an 8.6 s clip came back as "…tomorrow afternoon and", losing its last four
  words. After: transcribed whole.

### Changed
- **The CPU kernels now use every core.** `HartsyInference.Cpu` contained no threading of any kind, so a
  synthesis or a transcription ran on one core of whatever machine it was given. Four kernels now fan out
  through a new `CpuParallel` helper in `HartsyInference.Core.Numerics`, and `Conv1d` gained a vectorized
  stride-1 path:
  - `Conv1dKernels.Conv1d` splits over (batch, output channel) and, when a layer has too few output channels
    to fill the machine, over the time axis as well. A vocoder's last convolution has a single output channel
    and tens of thousands of samples, so splitting on channels alone would have left exactly the widest layer
    serial. With unit stride — every 1x1 projection and every dilated k3 residual in a VITS graph — the loops
    are also restructured from a per-element gather into a contiguous vector accumulate.
  - `Conv1dKernels.ConvTranspose1d` is re-expressed output-channel-first. The arithmetic and the accumulation
    order are unchanged, but with the input channel outermost every channel in a group accumulated into shared
    output rows, which no work split could have been made safe.
  - `MatMulKernels.MatMul` and `LinearTransB` split over (row tile x column tile) rather than rows alone,
    because an LLM decoding one token at a time calls them with M = 1.
  - `AttentionKernels.ScaledDotProductAttention` splits over (batch, head, query slice); its one shared scratch
    row, the only thing preventing this, is now allocated per worker.

  Measured on an 8-core dev box, Release, same commands before and after:

  | Benchmark | Before | After |
  |---|---|---|
  | `Bench_Piper` (3.2 s utterance) | 9.872 s, RTF 3.115 | 0.658 s, RTF 0.208 |
  | `Bench_WhisperBase` (3 s clip) | 7.240 s, RTF 2.413 | 2.112 s, RTF 0.704 |

  Small calls still run serially: dispatch costs more than the work below `CpuParallel.MinWorkForParallel`.

### Added
- **`HARTSY_CPU_THREADS`** (`numerics.cpuThreads`) caps the workers those kernels use; 0, the default, means one
  per logical core. It exists because the CPU device gate is a no-op, so an LLM decode and a TTS synthesis
  genuinely overlap on the same box and would otherwise each claim every core. The cap is enforced through a
  shared task scheduler rather than `MaxDegreeOfParallelism` alone, which bounds only one call at a time and
  would let two concurrent generations oversubscribe between them.
- **`SttBenchTests.Bench_WhisperBase`**, the transcription counterpart to the existing TTS benchmarks. Speech
  recognition sits on the critical path of a voice turn and had no benchmark at all, so kernel work could speed
  up synthesis and leave it untouched unnoticed.
- Above-threshold correctness tests for all four kernels (`Conv1dParallelKernelTests`,
  `MatMulParallelKernelTests`). Every pre-existing kernel test runs at 32x32 or smaller, which is below the
  parallel threshold — the whole suite exercised only the serial branch and would have passed regardless of
  what the parallel one did.

### Fixed
- **A missing checkpoint is now one contract: HTTP 400, naming the model the caller asked for.** Selecting a
  model that is neither in the catalog nor on disk reported `Model path not found: ` — an empty path, and no
  mention of the selection that failed. `InferenceEngine.ResolveFamilyId` is reached through
  `SupportedFeatures`/`DefaultsFor` before either construction guard, and handed its null `LocalPath` straight
  to `ModelLayoutResolver`, which could only echo the empty path it was given. Three engine guards now share
  one helper, and every modality service was brought onto the same contract:
  - Text, embedding, mesh, world and restore name the requested model instead of a generic noun.
  - Vision's embed path and its per-annotator `RequirePath` raised `InvalidOperationException`, which no
    `GenerationErrors` arm maps — so an absent checkpoint surfaced as **HTTP 500 `server_error`**. They now
    raise `FileNotFoundException` (400), and `RequirePath` names both the caller's id and the family it
    resolved to rather than a hardcoded label the caller may never have typed.
  - The four vision annotator "not installed" errors (CLIPSeg, YOLO, Grounding DINO, RT-DETR) were 500s for
    the same reason and are now 400s.
  - An empty model id resolved `Path.Combine(root, "Vision", "")` to the modality *folder*, which exists — so
    it passed every null guard and failed deep inside a loader, naming a server path the caller never sent.
    `ModelResolver` now declines a blank id.
  - `WorldService` checked for a checkpoint before its "catalogued but not loadable" cases, so `matrix-game-2`
    and `matrix-game-3` reported a missing file instead of the explanation that no file would help.

### Changed
- **LTX-2.5 defaults are template-faithful.** Distilled checkpoints now run the shipped ComfyUI workflow by
  default: 8-step fixed-sigma base pass at the half grid, learned x2 latent upsample (auto-downloaded side
  model), 3-step refine (`HARTSY_LTX2_TWO_STAGE=0` restores single-pass). A `distilled`-named checkpoint
  selected under the plain `ltx-2.5`/`ltx-2` id auto-routes to that contract with a log line, which also
  makes it reachable from SwarmUI. Default geometry for both LTX-2.5 families is now 1280x736 (two-stage
  decodes 1280x704), 121 frames @ 24 fps — the template's 5-second clip — and the dev family's default
  sampling moved from the never-measured 50 steps / cfg 3.0 to the measured parity profile 20 steps /
  cfg 4.0. Default generations are minutes, not seconds; pass explicit `--width/--height/--frames/--steps`
  for quick turnarounds. The 2.3-lineage ids share the dev defaults.

### Removed
- **`VideoRequest.VideoExtendModel`** — never consumed by any recipe on either side of the contract since the
  DTO's introduction (video extension was explicitly out of scope in the extension's own plan). Breaking
  record-shape change for transports that set it; the SwarmUI extension's mapping was removed in the same pass.

### Added
- **ComfyUI `int8_tensorwise` quantization, resident (± the `convrot` Hadamard rotation).** This is the format
  the *official* Lightricks LTX 2.5 and Comfy-Org MiniMax-H3 quantized releases ship in, and it was previously
  rejected by name at load. Weights now stay int8 on the device at 1 byte/param instead of expanding to BF16,
  so a 21 GB DiT fits a 24 GB card. Format notes:
  [`docs/Research/QUANTIZATION_COMFY_FORMATS.md`](docs/Research/QUANTIZATION_COMFY_FORMATS.md).
  - **`Tensor.QuantInfo`** (`QuantWeightInfo`) carries a packed weight's companions — per-output-row scale,
    ConvRot group size, `full_precision_matrix_mult` — the way `Fp8ScaleFactor` already carried fp8's scalar.
    Attaching them to the weight rather than to a per-model linear wrapper is what lets one backend branch
    serve every model that loads such a checkpoint, with no change to any model's code.
  - **`CudaBackend.Linear`** gained a resident-int8 branch reusing the existing W8A8 chain: activation ConvRot
    (`convrot.ptx`, a radix-4 butterfly — `H` is `kron(h4, …)`, so no matrix is materialized), per-row dynamic
    int8 quant, cuBLASLt IMMA, then the `rowScale·wScale + bias` epilogue. Chunked over rows against live free
    VRAM, because the int32 accumulator is 4 bytes per output element and the weights have already filled the
    card. Short sequences pad to the 32-row IMMA granularity rather than falling back, matching comfy-kitchen.
  - **`Int8ConvRotCodec`** provides the un-rotating dequant for the CPU/Vulkan backends and for layers tagged
    `full_precision_matrix_mult`. Verified against comfy-kitchen's eager reference at **relL2 5.1e-8–2.7e-7**
    with F32 activations; `H` itself matches bit-exactly.
  - **`ComfyQuantDescriptor`** replaces three divergent private copies of the `.comfy_quant` blob parser. The
    per-layer blob is now authoritative over the file-level `_quantization_metadata` mirror, which re-quants of
    the same model disagree with.
  - MiniMax-H3's `int8_convrot` rejection is deleted; `MiniMaxH3Assets` no longer sinks `convrot` filenames.
- **ComfyUI `nvfp4` weights stay resident too.** They were unpacked to BF16 at load, which turned the official
  18.72 GB LTX-2.5 distilled nvfp4 DiT into 42 GB. `Nvfp4Codec.TryAttachResident` now relabels the packed
  weight to `DType.F4E2M1 [N, K]` — the dtype already existed for exactly this — and `CudaBackend` dequantizes
  it in-kernel per GEMM under the existing `CacheWeightCasts` budget. Bit-exact against the host reference on
  real `qwen3vl_32b_minimax_h3_nvfp4_awq` layers. A **VRAM** win only: no consumer GPU here has FP4 tensor
  cores, so the GEMM still runs in F16. Opt-in per caller, since the eager unpack is what CPU/Vulkan need and
  AWQ layers with `pre_quant_scale` must take it regardless.
- **`Tensor.ReinterpretAs`** — a byte-count-validated, keep-alive-rooted dtype/shape view. `Reshape` could not
  serve: it holds element count fixed, and the whole point here is that one U8 byte becomes two F4E2M1 elements.
- **LTX-2.5 pipeline wiring — the Gemma 4 tower is now driven, not just built.** `Gemma4TextEncoder` and
  `Gemma4Tokenizer` existed and were parity-checked but had no consumers; `LtxVideo2Recipe` still constructed the
  Gemma-3 tower and refused a 2.5 bundle with a targeted error.
  - **`ILtx2TextTower` / `ILtx2PromptTokenizer`** name the contract the pipeline was already relying on
    structurally. Both encoders already exposed `EncodeMultiLayer`/`EnumerateWeights`/`NumLayers` with identical
    signatures, so neither implementation changed. `LtxVideo2Pipeline`'s 3840 caption channels and 49 harvested
    states hold for both families.
  - The recipe branches on `model.layers.0.layer_scalar` — **not** on a missing `v_proj`, which would misclassify
    every Gemma 4 checkpoint, since layer 0 is a sliding layer and has one. Gemma 4 conditions at 1024 tokens
    against Gemma 3's 256; that length is part of the conditioning, because the connector replaces learnable
    registers positionally.
  - The tokenizer is built from the `tokenizer_json` U8 tensor LTX-2.5 embeds **inside** its text encoder — there
    is no side file anywhere to fall back to.
  - **Converter routing**: a standalone Gemma 4 tower ships bare `model.layers.*` keys with no `text_encoder.`
    prefix and was falling through to the DiT mapper. Those now route to the text-encoder bucket, checked after
    the connector rule because `text_embedding_projection.*` lives in the same file but belongs to the connectors.
    Packager-embedded `hf_asset__*` side files (chat template, tokenizer/processor config) are dropped — they are
    not weights. Verified on the real 15.37 GB checkpoint: 328 packed int8 tower weights, zero leaking into the DiT.
  - Real-weight verified: the int8-convrot Gemma 4 encoder produces finite, deterministic, prompt-discriminating
    conditioning on a 4090 — identical tokens give bit-identical states, differing tokens diverge.
  - Unchanged on purpose: the diffusion-decoder guard (decode through the conv VAE — the diffusion decoder is
    managed-only), and both documented divergences from upstream (the 49th state's final norm, and padding side),
    which also affect the shipping 2.3 path.
- **LTX-2.5 support (components complete; end-to-end generation not yet wired).** Every piece is ported and
  checked against the reference, but the LTX-2 pipeline still constructs the Gemma-3 tower and the
  convolutional decoder, so a full 2.5 bundle is now refused with a targeted error instead of being
  mis-decoded. Details and the two open questions are in
  [`docs/Research/LTX_2_5.md`](docs/Research/LTX_2_5.md).
  - **Variant detection** replaces the hardcoded `LtxVideo2Config.V23`: `SafeTensorsLoader` exposes a file's
    `__metadata__`, and `LtxVideo2VariantDetector` resolves a config from it with tensor-key probes as the
    fallback — probing whenever the architecture config did not state a value, since repacks routinely ship
    metadata with no LTX config, and letting key presence win for the keyframe marker as ComfyUI does.
  - **Transformer**: `keyframes_abs_pos_embedding` applied to the first latent frame's tokens in all three
    forward paths including the captured step graph, plus a load-time cross-check that rejects a checkpoint
    contradicting the detected variant. The 2.3→2.5 architecture diff is only two config keys.
  - **`IBackend.Na3d`** — 3D neighborhood attention (NATTEN window semantics: the window slides inward at
    borders rather than truncating), verified against the reference at relL2 < 1e-6.
  - **`LtxVideo25DiffusionDecoder`** — the new diffusion video decoder, relL2 2.3e-7 on final pixels and a real
    decode of the shipped checkpoint. Managed-only, so it is a numerical reference rather than a fast path.
  - **`Gemma4TextEncoder` + `Gemma4Tokenizer`** — per-layer alternating geometry (global layers carry one KV
    head at a different head dim, no `v_proj`, and a 25% partial rotary), RMS weights stored directly rather
    than as Gemma 3's `1+w`, and a rank-merge BPE tokenizer that is bit-exact against the HuggingFace library
    on the real 262k vocab.
  - **Distilled sampling**: a checkpoint's baked-in sigma schedule replaces the dynamic flow-match shift, and
    the unconditional branch is skipped when both guidance scales are 1.
  - **Catalog**: `ltx-2.5` and `ltx-2.5-distilled` as separate ids, because the two checkpoints are
    byte-indistinguishable and only the id can carry which schedule was intended.
- **Device-resident CFG+Euler for the image denoise loops** (CUDA image bring-up): Lance, Lumina2, HiDream,
  F-Lite, Kandinsky5, SD3, and Z-Image's fast path now run guidance + the Euler update in-place on device via
  `CfgEulerStep` and the new fused `CfgRenormEulerStep` (Lance renorm), `CfgNormalizedEulerStep` (Lumina2
  `cfg_normalization`), and `AffineMix`/`MaskedAffineMixInPlace` (SD3 img2img noise + masked-inpaint blend/
  recomposite) — replacing per-step scalar host loops over `DataPointer`. New `MixContract`/`PatchTokenContract`/
  `SplitContract` validate the op geometry identically across backends. BF16 elementwise dispatch (Gelu, Clamp,
  GeGlu, RepeatKvHeads) no longer silently falls through to the F32 kernel — GeGlu gained a real BF16 kernel,
  RepeatKvHeads a 16-bit bit-copy launcher, and unsupported dtypes now throw.
- **Z-Image lifecycle hardening**: Base/Turbo checkpoint-variant detection from the filename, two independent
  prompt-cache layers, a Qwen3 tokenizer rewrite (byte-level encoding gap, `<think>` handling, tokenization-
  boundary fix) golden-tested against HF tokenizers 0.22.2 output, and a CUDA-graph-captured denoise step for
  the packed fast path. Scope note: the packed device-resident loop covers t2i, img2img, and CFG; masked
  inpaint and regional conditioning still run the host-stepped loop (per-step `ApplyZImageCfg`/`scheduler.Step`
  on the CPU) — porting those onto the SD3 `MaskedAffineMixInPlace` pattern is tracked follow-up work.
- **Kernel build reproducibility**: `conv/`, `vision/`, and `wan/` gain the same `build.sh` (nvcc, or the
  committed `nvrtc_compile` fallback) the other kernel domains already had — their shipped PTX previously had
  no scripted rebuild path at all; `dequant/build.sh` now covers `w8a8` (sm_75) and `fp8_quant` (sm_80, per its
  shipped target) and gains the nvrtc fallback; `dit/build.sh` covers `mg3_action`. 13 of 16 artifacts
  reproduce bit-identically from source; `stepcache`/`w8a8`/`wan_vae_norm` are regenerated with the current
  pinned toolchain (all kernel-family GPU tests pass against the regenerated artifacts).

- **Wan 2.2 A14B dual-expert swap through the native contract** (regression restore): `VideoRequest.VideoSwapModel`
  + `VideoSwapPercent` (fraction of steps for the low-noise expert; null = official 0.875/0.9 boundary) →
  `WanVideoRecipe` loads the second DiT and warps the fraction through the flow shift
  (`boundary = s·p/(1+(s−1)·p)`); swap-aware pipeline cache key; CLI `--swap-model`/`--swap-percent`.
- **FLUX.1 Redux through the native contract** (regression restore): `redux.stylemodel`/`redux.multiply`/
  `redux.merge` (number 0..1)/`redux.apply_start` Extra keys drive `ReduxResolver` from `Flux1RecipePipeline`;
  prompt images ride `IpAdapter.PromptImages`; Flux declares `ImageFeatures.IpAdapter` (Redux only — real
  IP-Adapter checkpoints are refused with a clear message). CLI `--style-model` + redux knobs.
- **Wan-Animate driving video**: `VideoRequest.DrivingVideo`/`DrivingPoseVideo`/`DrivingFaceVideo`/
  `DrivingAutoPreprocess` with in-engine YOLO11-pose skeleton render + face crop (ported from the extension's
  dead preprocessors); single-still tiling kept as fallback. CLI `--driving-video`/`--pose-video`/`--face-video`/
  `--no-auto-preprocess`.
- **`VideoFeatures.ReferenceImages/ReferenceVideos/ReferenceAudios/DrivingVideo`** gating bits — closes the
  silent drop of reference conditioning on families that never consumed it.
- **`ImageRequest.InstructPix2PixCfg` wired** to OmniGen2 (default 2.0) and Boogu (default 1.0) dual-CFG edit
  paths; CLI `--ip2p-cfg`.
- **Multi-GPU sharding, placement & parallelism — the full opt-in feature set** (`PlacementConfig` /
  `EngineOptions.Placement`; all-defaults is byte-identical to single-GPU). Works over plain PCIe with
  no P2P/NVLink (host-staged boundaries; P2P used when available). User guide: `docs/MULTI_GPU.md`.
  - **LLM layer split** (`ShardDevices`, or a `"cuda:0+cuda:1"` composite device key / `--device` in the
    CLI): N-way transformer layer split planned from live free VRAM (or explicit `ShardRatios`), per-stage
    asymmetric weight preload, KV cache per-layer on its stage's card, logits/sampler on the last stage.
    Verified: Llama-3.2-1B split = exact token parity vs single-GPU; **Qwen3-32B Q4_K_M (19.8 GB) OOMs a
    24 GB 4090 alone and runs at ~12.1 tok/s split across 4090+3060**. Exclusions: SSM, Gemma-4 PLE, VLM
    sidecars (warned + skipped); CUDA-graph/speculative decode disabled while staged.
  - **DiT block sharding** (`EnableDitSharding`, exactly 2 devices; CLI `--dit-shard-gpu`, extension
    `DitShardGpuId`): diffusion transformer block-range split with pooled (never replicated) weights.
    Verified on real weights: Krea2, Qwen-Image 20B, Flux.1 (plain generations; ControlNet/Kontext/
    inpaint/regional auto-fall-back), Chroma, HunyuanImage 2.1, MiniMax-H3 fp8. Disables step-graph/
    step-cache/block-streaming while sharded; mutually exclusive with CFG-parallel.
  - **Audio-LM layer split + precision policy** (CLI `--lm-shard-gpu`, extension `LmShardGpuId`): YuE's
    7B Stage-1 rides the same layer-split machinery; the load-time Q4_K quantization became a policy
    (`HARTSY_AUDIO_LM_QUANT=q4k|q8|off`) defaulting to **un-quantized bf16 when sharded** — pooled at
    8.7 + 4.3 GB across 4090+3060. `MusicLoadContext` carries shard backends + precision to loaders.
  - **TE/VAE component placement** (`TextEncoderDevice`/`VaeDevice`; CLI `--te-gpu`/`--vae-gpu`): Wan
    TI2V-5B 43.7 s → 32.7 s with umT5 on the second card; SDXL SSIM 0.9998; Flux/Qwen-Image/Chroma/
    HunyuanImage/LTX-1/LTX-2 wired. Composes with DiT sharding.
  - **CFG-branch parallelism** (`CfgParallelDevice`; CLI `--cfg-parallel-gpu`): negative branch runs
    concurrently on a second card with replicated weights (~1.8-1.9× per-step concurrency; Wan +
    Flux true-CFG), observably falling back to sequential when the replica doesn't fit.
  - **Same-GPU dual backends**: two engine instances share one physical GPU with isolated
    streams/caches/mempools; serialized per-ordinal by default (`HARTSY_SAME_GPU_CONCURRENT=1` opts into
    concurrent mode, which has a known allocator issue near VRAM capacity — left off).
  - Verification: `tests/run-multigpu-campaign.sh` (real-weight, fail-on-missing-checkpoint) covering
    every mode; measured tables in `benchmarks/results/2026-08-05_multigpu_speeds.md`.
- **YuE full-quality pipeline activated**: Stage-2 (m-a-p/YuE-s2-1B-general, cb0 → all 8 codebooks) and
  the per-stem 44.1 kHz Vocos vocoders are now weights-catalog entries (auto-download); the vocoder
  torch checkpoints auto-convert to safetensors on first load (`EnsureVocoders`, same pattern as
  x-codec). Without them YuE silently degraded to the vocal-cb0-only 16 kHz draft — the "garbled" mode
  (Whisper transcribes it as nothing; the full pipeline transcribes supplied lyrics near-verbatim).
  CLI `hartsy music` gained `-g|--genre` (YuE's prompt is the LYRICS — `[verse]`/`[chorus]` markers —
  and genre carries the style tags).
- **MiniMax-H3 ("Hailuo 03") — full port, real-weight video + audio verified on a 12 GB RTX 3060.** Single-stream
  packed-token DiT (`[text | cond | audio | video]`, hidden 5376, 50 blocks) denoising 24-channel video and 32-channel
  40 Hz stereo audio jointly, with a ViT3D video VAE, a DAC/BigVGAN audio VAE, and an NVFP4-AWQ Qwen3-VL text encoder.
  Verified end-to-end: 512×288 / 39 frames / 30 steps produces a tracking shot of a dog splashing through a stream with
  a 1.625 s stereo soundtrack at −12.5 dBFS peak. The 66 GB bf16 DiT is mmap-backed and loads at 943 MB RSS.
  Layout, dual-shift schedule, audio row packing and final-layer modulation rows are byte-identical to upstream
  ComfyUI master; the audio VAE decoder matches the reference module shipped inside the checkpoint at relL2 4.9e-6
  (CPU) / 1.3e-3 (CUDA), covered by the new `MiniMaxH3AudioVaeParityTests` + `tests/python-reference/
  minimax_h3_audio_vae_ref.py`.

- **SeedVR2-7B support (v1 NaDiT port).** The 7B checkpoint is the V1 architecture (`models/dit`, not the
  3B's `dit_v2`) — same windowing/attention/AdaSingle, but a different RoPE (pixel-basis freqs
  `linspace(1,128,10)·π` matching the checkpoint's `rope.rope.freqs`, 60 of 128 head dims rotated,
  positions normalized to `linspace(−1,1)` over each window axis, applied to video only — text is never
  rotated), plain GELU-tanh MLP with biases, all 36 blocks fully split (no `mm_layers`), and no tail
  norm/ada or last-layer text shortcut. `SeedVr2Config.Detect` now configures all of this from the
  plain-MLP+no-tail signature instead of throwing. Parity: a v1 tiny-config dump against ByteDance's own
  `models.dit.nadit` (`seedvr2_transformer_v1_parity_dump.py`) passes at blocks ≤9.1e-4 / output 8.96e-4
  (`Dit_TinyConfigV1_ForwardMatchesReference_PerBlock`).
- **SeedVR2 BF16 VAE activations** (CUDA default; `HARTSY_SEEDVR2_VAE_F32=1` reverts): the fp32
  whole-clip activation peak that OOM'd 24 GB at 720p-area is halved — 720p-area now restores a full
  25-frame clip at a measured 13.3 GB peak, 960×540-area at 9.0 GB, and **the 12 GB 3060 runs 960×540-area
  end-to-end (peak 7.8 GB)**. Pixel/latent boundaries stay F32; the mid-block attention runs F32 (the
  known F16-attention precision class). BF16 variants of the five `wan_vae` glue kernels; BF16 admitted
  through `SliceRows`/`Permute0213`. Output vs the f32 path: SSIM 0.9998. 1080p-area still exceeds 24 GB —
  tiled/sliced VAE remains open.

- **LTX-2.3 audio guidance rescale — the near-silent soundtrack is ~14 dB louder.** Root cause measured:
  CFG over-disperses the audio latent (σ 0.89 at guidance 1 → 2.22 at 3 → 2.69 at 7, against the
  checkpoint's own 1.17) and the decoded level falls with it; the audio VAE is not at fault (fed a
  training-distribution latent it decodes to a healthy level). `AudioGuidanceRescale` (default 1.0) applies
  diffusers' `rescale_noise_cfg` to the audio stream, restoring σ to 1.141. Real 20-step generation
  (512×320×25f, seed 42): **peak −43.9 → −28.2 dBFS, RMS −59.4 → −45.4 dBFS**, video unchanged.
  Implemented as an affine transform so only four scalars reach the host and the step stays on-device.
  Also adds `AudioGuidanceScale` (null = follow the video scale, the reference default) with
  `HARTSY_LTX2_AUDIO_CFG` / `HARTSY_LTX2_AUDIO_RESCALE` overrides. **Still ~20 dB below a healthy
  soundtrack** — the reference's STG + modality-isolation guidance remain unimplemented; see
  MODEL_STATUS_VIDEO. Note for anyone tempted: raising audio guidance to the authors' recommended 7.0
  *alone* makes it ~17 dB worse, because that recommendation assumes the rescale/STG stack.
- `HARTSY_LTX2_PROBE=1` now also dumps the audio stages (latent pre/post-denorm, VAE log-mel, vocoder
  waveform) via a shared `ProbeTensor` helper.

### Changed
- **`LlamaStyleEncoder` attention glue is device-resident** — the per-layer CPU reshape/RoPE/GQA-repeat/merge
  `float*` loops are replaced with the existing `Permute0213`/`ApplyRopeSingleHeadMajor`/`RepeatKvHeads`
  kernels (this encoder is shared by Qwen-Image, Z-Image, Krea2, Boogu, Flux.2, Ideogram 4, Lumina2, OmniGen2
  and others); tests assert the D2H sync count, not just numerics.
- **Lumina2 sampling schedule corrected — output images change.** The scheduler previously applied Flux-style
  dynamic shifting derived from the image token count (an experiment its own comment marked VALIDATION-PENDING,
  using Flux's base/max-shift constants); the official Alpha-VLLM/Lumina-Image-2.0 `scheduler_config.json` is
  `shift: 6.0` with `use_dynamic_shifting: false`, so the pipeline now uses the checkpoint's static shift. Same
  seed produces a (correctly) different image than prior releases.
- **SD3 patchify/final-layer/masked-mix run on device**; `flash_attn_v2_tf32` rejects partial query tiles
  (OOB read) and zero-fills its shared-memory K/V tail (stale-value poisoning); MaxPool distinguishes an empty
  window from a valid all-−Inf one; MSDA uses true −Inf softmax init and 64-bit index products; the cuDNN SDPA
  plan cache is keyed by attention scale; the step-cache treats a zero-denominator relative distance as
  Infinity (was a false cache HIT); Sage's F32→F16 V-narrowing is opt-in (`HARTSY_SAGE_UNSAFE_F32_V_NARROW`).
- **SeedVR2 DiT is device-resident** — the bring-up host-math forward (window gather/scatter, rope,
  qk-norm, AdaSingle on CPU spans; ~200 stream drains per forward) is replaced with backend-op
  composition: fused `QkvSplitNorm`, `RowGather`/`RowScatterAdd` window packing over cached per-geometry
  index tensors, `WanRopeInterleaved` with per-token identity-padded tables, and modulation vectors
  precombined per timestep (constant 1000) and cached across chunks (`SeedVr2DevicePlan`). No new
  kernels beyond GPU-resident `SeedVr2PixelShuffle`/`SeedVr2PadBottomRight` (the VAE upsampler/downsampler
  host loops, previously a multi-GB D2H+H2D round trip each). **Measured e2e: 960×540-area 14.5 → 2.7
  s/frame (362 → 68 s); 720p-area 25.7 → 8.4 s/frame (4090, BBB 25 f). The 3060 runs the same clip in
  169 s.** Existing tiny-config parity numbers are unchanged to the printed digit; per-chunk phase timing
  is logged at Debug level.

### Fixed
- **LTX-2.5 ignored the prompt entirely — `prompt_adaln` was driven by the raw flow sigma.** Output was sharp
  and temporally coherent but followed the seed, not the text: two unrelated prompts at one seed differed by
  1.28% of pixel range, guidance 1 and guidance 10 behaved identically, and zeroing the *entire* 1024-row
  conditioning changed nothing. `LtxVideo2Transformer` passed the unscaled sigma (0..1) to
  `prompt_adaln_single` / `audio_prompt_adaln_single`, where the reference passes the same ×1000-scaled
  timestep every other modulator gets. Those modules emit the `shift_kv`/`scale_kv` that modulate the **text
  keys and values** into every block's cross-attention, so evaluating a sinusoidal timestep embedding at t≈1
  instead of t≈1000 left the cross-attention with the right magnitude but no ability to discriminate between
  prompts. A code comment stated the wrong convention as fact, which is what kept it alive. Fixed at all six
  call sites (video + audio); per-block prompt sensitivity against ComfyUI 0.32 went from 8–13× too weak to
  **ratio 1.00**. LTX-2.5 now generates prompt-faithful 704×480×25f clips with a soundtrack in ~80 s on a 4090.
  The whole text path was verified against the reference on the real `int8_lean_convrot` checkpoints on the way
  (tokenizer ids byte-exact, Gemma-4 tower cosine 0.9999–1.0000 per layer, connector within 0.2–0.6%, `attn2`
  within 0.07%) and needed no changes; two research items previously flagged as open — the 49-state stack's
  final norm and the left-vs-right padding side — were both settled as **not** defects, the first of which also
  clears the shipping LTX-2.3 Gemma-3 path.
- **Z-Image Base checkpoints were silently corrupted when the filename carried no variant token.** The official
  Base release ships under the bare family name (`z_image_bf16.safetensors`); variant detection fell through to
  Turbo's policy, whose F16 attention narrowing overflows Base's >83k value-projection range into Inf — a
  garbage image with no error. Bare family naming now positively detects Base, and genuinely ambiguous
  filenames default to the numerically safe Base policy (F32 attention, shift 6) with a loud warning — a
  misfiled Turbo merely runs slower, instead of a misfiled Base corrupting. Verified with a real generation
  from the official Comfy-Org single-file on the exact previously-corrupted filename.
- **Z-Image `ReleaseDeviceCache` could leave the captured denoise step graph pointing at freed memory** — the
  graph bakes the caption-pin and RoPE-table device addresses it frees, so a later same-signature forward
  could replay against freed allocations (CUDA 700 context poison). The release path now invalidates the
  graph first, keeps an invalidation failure as the first error, and continues the rest of cleanup.
- **`CfgNormalizedEulerStep`/`ApplyCfgNormalized` produced NaN for `eps=0` with an all-zero guided row**
  (0/0 in the norm ratio; NaN then poisons `z` through `0·NaN`). A zero denominator now resolves the ratio to
  0 — exact, since an all-zero row contributes nothing — identically in the IBackend fallback, the CUDA
  kernel (`dit_f32.ptx` regenerated), and the host helper; new CPU+CUDA regression test.
- **cuDNN auto-fetch 404'd for CUDA < 12** (NVIDIA publishes no cuDNN 9.21 redist there) — now refused
  up-front with manual-install guidance instead of attempting the download.
- **`Qwen3Tokenizer` hardcoded its chat special-token ids** (`<think>`, `<|im_start|>`, pad/EOS), which only
  fit the embedded artifact — a caller-supplied `tokenizer.json` now has them resolved from its added-token
  table, with a logged fallback when absent.
- **Z-Image rejected legitimate solid-color output** — a uniformly black/white frame now only fails the
  generation when the decoded F32 tensor is actually non-finite; a finite solid frame (valid prompt outcome,
  inpaint over a solid source) is accepted with a log line.
- **CUDA BF16/F16 GroupNorm mis-read non-F32 affine weights.** `CastAffineDownIfF32` only converted an
  F32 affine down to the kernel dtype; an F16-checkpoint affine (e.g. the numz SeedVR2 VAE) was passed
  raw to the BF16 kernel — F16 bits reinterpreted as BF16 → garbage scale/shift and flat-gray output.
  Any affine dtype is now converted to the kernel's dtype. Caught by the SeedVR2 BF16-VAE bring-up: the
  isolated parity test passed against the f32 checkpoint while the pipeline (fp16 catalog checkpoint)
  produced uniform gray.

- **Borrowed views passed as an in-place op's OUTPUT silently discard the write on CUDA** — a hazard class, found via
  MiniMax-H3. The backend binds the result to the borrowed `View`/`RowView` and the dispose callback skips the D2H, so
  the store never lands; the CPU path is unaffected, which is why unit tests passed. In H3 this made RoPE, adaLN
  modulation and the gated residual no-ops across all 50 blocks — `h` never left the patch embedding and every frame
  decoded to a regular-grid mosaic. Fixed by forcing the read-back at the three sites; CPU-vs-CUDA parity went
  0.246 → 4.75e-4 and step-0 velocity rms 7.90 → 2.24. The rest of the repo was swept for the same pattern and is
  clean: the only other borrowed views outside `HartsyInference.Core` feed a host `float*` loop (video-VAE tile
  blending) or are read-only GEMM inputs (LLM stacked-weight slicers).
- **MiniMax-H3 now loads SwarmUI/Comfy's flat checkpoint layout, not just the vendor folder tree.** The vendor
  publishes `transformer/` + `video_vae/` + `audio_vae/` + `text_encoder/` folders; Comfy-Org repackages the same
  weights as one file per component under `diffusion_models/`, `vae/` and `text_encoders/`, and that is what SwarmUI's
  native H3 support downloads — so a Swarm-driven load previously failed looking for a `transformer/` subfolder that
  does not exist. New `MiniMaxH3Assets` resolves both layouts, walking up from the DiT to find components and ranking
  variants so an unloadable `int8_convrot` file never beats a loadable sibling. Nothing is downloaded: re-fetching
  under the engine's own model directory would duplicate the ~5.8 GB of VAEs Swarm already has. Falls back to the
  embedded Qwen BPE and to `MiniMaxH3VideoVaeConfig.Detect` since the flat repack ships no tokenizer or `config.json`.
- **MiniMax-H3 `pruned_fp8_scaled` checkpoints load, and are ~10x faster than bf16.** Two defects blocked them.
  (1) `ThrowIfInt8Convrot` rejected on the *presence* of Comfy's quantization companions, but Comfy tags every
  quantized build with the same `.weight_scale`/`.input_scale`/`.comfy_quant` suffixes — only the `.comfy_quant`
  descriptor distinguishes them, and the fp8 build says `{"format": "float8_e4m3fn"}`. The guard now reads the
  descriptor and rejects only genuine int8-convrot (`MiniMaxH3QuantGuardTests`; an absent/unreadable descriptor still
  rejects conservatively). (2) The converter never called the shared `CheckpointConvertUtils.ApplyFp8ScaledDequant`,
  so the scale companions were routed as unknown weights. **Verified on the real 21 GB
  `minimax_h3_fl2va_pruned_fp8_scaled` checkpoint:** 22 frames at 512x288 / 20 steps produces a coherent tracking shot
  with matched 0.9167 s audio at -21.5 dBFS peak, at **8.6 s/step and 22.5 GB VRAM, fully resident on a 24 GB 4090** —
  versus ~90 s/step for the 66 GB bf16 build, which cannot stay resident and re-reads most of itself from NVMe every
  step. This run is also the first exercise of the *pruned* checkpoint's `adaln_t_table` curve path (`curves=True`).

- **MiniMax-H3 is 25.9x faster: 50.2 -> 1.94 s/step** (512x288, 141 frames, RTX 4090; a full 30-step clip went
  1602 s -> 129 s). ComfyUI does the identical work at 1.67 s/step, so this closes a ~30x gap to ~1.16x.
  The cause was host round-trips, not weight residency or GEMM selection. `View`/`RowView` were built as
  `new Tensor((void*)t.DataPointer, ...)`, and `GpuTransferHelper`'s activation cache is keyed by Tensor object
  reference, so a view can never alias its parent's device buffer — and merely CONSTRUCTING one calls
  `DataPointer` -> `EnsureCpuData` -> cache-evict + `cuStreamSynchronize` + device-to-host copy. The worst
  offender was the QKV split, which host-copied a `[seq, 21504]` tensor three times per attention (~473 MB each
  way per block, ~47 GB/step). Restructured so views are never needed: `SliceLastDim` for the QKV and adaln
  splits, q/k allocated 4-D up front so `RmsNorm` runs in place and `ApplyRopeSingle` consumes them directly,
  the residual stream shaped `[seq, 1, hidden]` so `AffineBroadcastLastDim`/`GatedResidualLastDim` modulate the
  whole packed sequence in a single launch driven by a `RowGather`ed table, and `Concat` for segment assembly.
  **No new kernels.** Acceptance metric: D2H syncs per forward **74 -> 0** (`IBackend.GetD2hSyncCount`, whose own
  doc states a fully GPU-resident denoise loop must stay at ~0). Numerics unchanged — parity holds at video
  relL2 4.752E-004 / audio 6.555E-004.
- **MiniMax-H3 text encoder: ~2x less PCIe traffic per prompt.** The nvfp4 tower dequantized every layer into a
  full F32 weight and uploaded it per call (~97 GB per encode). It now narrows to BF16 inside the dequant loop
  and reuses one shared host scratch buffer instead of ~350 short-lived 200-500 MB allocations, and drops a
  redundant per-call `Sync()` (`FreeWeights` already syncs). BF16 rather than F16 because F16 overflows on the
  SwiGLU gated tensor. Qwen3-VL is BF16-trained, so this is closer to the reference than the old F32/TF32 path.

- **MiniMax-H3 geometry was wrong at its own declared defaults.** Three grids were mis-derived: frame counts must
  snap up onto `17k+5`, video latent frames are `(frames-5)/17*5 + 2` rather than `frames/4`, and pixel axes round to
  32 rather than 16 (a multiple of 16 that is not a multiple of 32 gives an odd latent axis, and the 2x2 patchifier
  silently drops its last row/column). At the shipped defaults `1360x768x121f` that meant 1344x768 output and 102
  delivered frames sized against ~5.0 s of audio — roughly 0.8 s of soundtrack generated and then trimmed away. The
  reference grids now live in `MiniMaxH3Geometry` and are pinned by `MiniMaxH3GeometryTests`, including a round-trip
  asserting the latent count re-expands to exactly the requested frames. Defaults corrected to 1344x768x124f.
- **`CudaBackend` now logs the device name at construction.** `CUDA_VISIBLE_DEVICES` defaults to fastest-first
  ordering, so it does not agree with `nvidia-smi` indices — every perf and VRAM figure from an H3 bring-up run was
  initially attributed to the wrong GPU because only the ordinal was logged.

## [2.0.0-alpha.8] — 2026-08-01

### Fixed
- **A short soundtrack silently dropped trailing video frames.** Muxers cut to the shorter stream
  (ffmpeg `-shortest`), and LTX-2.3's audio-latent count rounds down: a real 25-frame @24fps clip
  (1.0417s) came back with 1.010s of audio, so the muxed mp4 contained **24 frames, not 25**.
  `VideoAudioResolver` now fits the track to the clip in both directions — trim if long, silence-pad
  (`AudioBuffer.PadTo`) if short — so frame count is preserved; a shortfall over 0.25s still warns,
  since that indicates the wrong track rather than latent rounding. Verified on a real LTX-2.3
  generation: audio 1.0417s, muxed mp4 keeps all 25 frames, and the generated samples are
  bit-identical to the pre-fix run with the padding appended as pure silence.
  Found by the e2e run after alpha.7 was cut, hence the separate version.

### Note
- `2.0.0-alpha.7` was tagged but never appeared on nuget.org (both the flat-container and registration
  indexes still topped out at alpha.6 more than 30 minutes after publish). Consume alpha.8 instead.

## [2.0.0-alpha.7] — 2026-08-01

Video gets its sound back: generated audio now reaches the caller (closes `TODO(E-IMG-4/5)`), plus the
LTX-2 split-checkpoint decode fix.

### Added
- **`AudioBuffer`** (`Engine.Requests`) — engine-native raw planar-float PCM, the decoded counterpart to
  `AudioClip` (encoded in) and `AudioResult` (encoded out). Mono/stereo conversion + duration trim; the
  shared currency for moving a waveform between components in any modality.
- **`VideoGenerationResult`** — frames plus the soundtrack that belongs with them.
- **`VideoAudioResolver`** — one place that decides which track ships with a generation: what the pipeline
  attached beats `VideoRequest.VideoAudioInput` pass-through, then the track is trimmed to video length.
  `VideoAudioReference` is deliberately not a fallback (it is conditioning; a family that means it to be
  heard attaches it itself).
- `AudioClipCodec` is now public and gained `DecodeNative` (native rate/channels, no resample) and an
  `EncodeWav(AudioBuffer)` overload.
- REST `/v1/native/video/stream` emits an `audio` SSE event (base64 WAV + rate/channels).

### Fixed
- **LTX-2 split-checkpoint output was checkerboard garbage** (the known-broken `hartsy video -m ltx-2`
  path, which SwarmUI also hits). The split VAE file ships bare keys, and the converter's bare-key router
  only recognized `decoder.`/`encoder.`/`latents_` as VAE keys — `per_channel_statistics.{mean-of-means,
  std-of-means}` fell into the Transformer bucket, so latent denormalization silently became an identity
  no-op. With std-of-means as low as 0.074, the decoder received channels up to ~13× too hot; the up-stack
  amplified that to ±943 and the RGB clamp saturated to checkerboard. One added route in
  `LtxVideo2CheckpointConverter.RouteKey` fixes it: decode now lands in [-1,1] and the catalog path produces
  coherent frames (verified 512×320×25f, seed 42; transformer Sha256 pinned). Bundled single-file
  checkpoints were never affected.
- **LTX-2.3's generated soundtrack was dropped**, not muxed — `LtxVideo2RecipePipeline` logged a warning
  and discarded it because the pipeline contract carried frames only. It is now attached and muxed.
- **Wan2.2-S2V's driving speech was not muxed either.** The mux moved to the Engine when the extension was
  thinned to a wrapper, but was never implemented there; `VideoRequest.VideoAudioInput` was documented as a
  mux track with no consumer. S2V now attaches the speech it consumed, at source rate rather than the 16 kHz
  mono conditioning downmix.
- The SwarmUI extension's ffmpeg audio mux (`VideoOutputEncoder.AudioTrack`, `FormatSupportsAudio`) was
  unreachable dead code — never constructed, never passed. Reconnected, with a warning when the chosen
  container (gif/webp) cannot carry a track.

### Changed
- **Breaking:** `IVideoRecipePipeline.Generate` returns `VideoGenerationResult` instead of
  `IReadOnlyList<VideoFrame>`, and `IVideoService.GenerateAsync` returns `Task<VideoGenerationResult>`
  instead of `IAsyncEnumerable<VideoFrame>`. The enumerable never streamed — it awaited the full frame list
  before yielding — so no delivery behaviour is lost. Replaced rather than added alongside: a second
  frame-only overload would silently drop audio, which is the bug being fixed.
- `hartsy video` writes `audio.wav` beside the frame directory when a generation produces sound.

## [2.0.0-alpha.6] — 2026-08-01

SeedVR2 video/image restoration — a new modality, end to end.

### Added
- **SeedVR2 one-step video restoration** (`Modality.Restore`, catalog ids `seedvr2-3b`/`seedvr2-7b`):
  NaDiT windowed MM-DiT + s8c16t4 causal video VAE ported to pure C#, every stage parity-gated against
  the ByteDance reference — window partition **exact** (2,490 slices), preprocessing maxAbs **2.3e-6**,
  VAE relL2 **≤2.9e-6** vs real weights, full-model E2E **SSIM 0.99950 / 56.6 dB PSNR** vs the Python
  pipeline with injected reference noises. Surfaces: `hartsy restore <video|image>` (PNG frames + H.264
  MP4 out), `--restore` chain on `hartsy video`, REPL `/mode restore`, `POST /v1/native/restore[/stream]`,
  and the SwarmUI extension's "Video Restore" param group. 7-clip real-footage matrix verified on the
  4090 (USIA Reagan '87, NASA Apollo 11, JFK '61, Steamboat Willie, Prelinger '62, Big Buck Bunny
  ground-truth, still-image t==1 branch): 25-frame clips at 960×540-area, **~14 s/frame, 17.1 GB peak,
  zero OOM**. Ground-truth profile matches the paper: pixel metrics prefer bicubic (SSIM −0.05) but
  **LPIPS improves 26–28%** (0.735→0.541 extreme; 0.448→0.324 mild) — it repaints, it doesn't
  reconstruct; `--strength` guards oversharpening.
- **`FfmpegProcessDecoder`** (ffmpeg/ffprobe child processes) — first video-INPUT path in the engine;
  `VideoClip`/`RestoreRequest` DTOs; `TorchResize` (torchvision-exact antialiased bicubic, a=−0.5
  float32 weights — two silent-divergence bugs caught by parity, see PARITY_VERIFICATION).
- **Reference quirks ported deliberately** (SEEDVR2_ARCHITECTURE.md §2.5): the tail `vid_out_ada`
  cache-collision (uses the ATTN emb slice — the code as written is dimensionally impossible), last-layer
  `vid_only` semantics incl. the txt self-residual doubling, per-frame VAE GroupNorm stats, asymmetric
  (0,1,0,1) downsampler padding, MAGViT `(x y z c)` pixel-shuffle dropping output frame index 1.

### Known limitations
- fp32 whole-clip VAE activations cap restoration at ~960×540-area on 24 GB (5-frame chunks); 720p+
  needs bf16 activations or tiled VAE — tracked in MODEL_STATUS_VIDEO remaining work.
- DiT window gather/scatter and RoPE run host-side (bring-up shape): ~14 s/frame. Residency/CUDA-graph
  optimization is the follow-up perf pass.
- Catalog DiT + VAE download from the community safetensors mirror `numz/SeedVR2_comfyUI` (verbatim
  original state-dict keys, fp16; Sha256 pinned from a verified download → convert → restore run, and
  the fp16 output is visually equivalent to fp32 — remaining delta is generative high-frequency repaint).
  Only the 1.2 MB frozen pos/neg embeddings ship from `HartsyAI/SeedVR2-safetensors` (upstream has them
  as torch-pickle `.pt` only); until published, place `seedvr2_embeddings.safetensors` under
  `Models/Video/SeedVr2/`.
- **seedvr2-7b is catalog-registered but BLOCKED**: its smoke run revealed the 7B is the **v1 NaDiT**
  (`models/dit`, `qk_rope`/`shared_qkv`) whose state-dict keys coincide with v2 — it loaded and produced
  plausible-but-wrong mud (GT SSIM 0.71 vs 3B's 0.88). `SeedVr2Config.Detect` now throws on the v1
  signature instead of running it; the v1 port is tracked in MODEL_STATUS_VIDEO.

## [2.0.0-alpha.5] — 2026-07-27

Low-VRAM generation, a GPU-memory leak fix, and selectable devices.

### Added
- **Low-VRAM weight streaming across the image fleet** (`HARTSY_LOWVRAM`, three-state: `auto` default /
  `on` / `off`). The sliding-window machinery (`BlockStreamingController`) already existed but only one of
  ~25 image pipelines used it. **Four models that could not run on a 12 GB card now do**, all 1024²,
  quality-gate clean: **HunyuanImage-2.1** (19.7 s), **Ideogram 4** (205 s — a *pair* of 9.3 GB DiTs
  needing 19.7 GB against 9.2 GB available), **Qwen-Image** (231 s, 20B MMDiT), **Krea2** (71 s).
  `off` is a real escape hatch: the same request succeeds under `on` and raises `OutOfVramException`
  under `off`.
- **`VramPlanner`** — one place that decides resident-vs-streamed per generation phase, carries the
  `HARTSY_KEEP_MODELS` residency short-circuit, and logs the **weights-vs-activations split** (streaming
  can only move the weight term, so a phase dominated by activations needs a smaller working set, not a
  sliding window).
- **Selectable CUDA device**: `cuda:1`-style backend selectors, `InferenceEngine(selector, ordinal)`, and
  a real `GPU_ID` in the SwarmUI backend — previously logged and ignored. Verified by memory delta.
  Note the ordinal is CUDA's (fastest-first), which need not match `nvidia-smi`'s PCI order.
- **SD3.5 modular component loading** — CLIP-L / CLIP-G / T5-XXL / VAE each resolve independently when the
  checkpoint does not bundle them, which is the standard SD3.x distribution format. SD3.5-Medium now
  generates end-to-end; it previously threw before any sampling.

### Fixed
- **GPU memory leak on OOM.** `CudaBackend.PreloadWeights` had no exception path, so a mid-load OOM left
  already-uploaded weights registered against a model that would never finish — unreachable, therefore
  unfreeable. The process held ~11.5 GB with nothing running and **starved other processes on the same
  card**, including a separate ComfyUI. Now: typed `OutOfVramException`, per-batch rollback, and reclaim at
  both the generate and construct boundaries. An OOM'd process now holds **152 MiB instead of ~11.5 GB**,
  and a sequential multi-model sweep survives an OOM (3/3 models succeeded after one).
- **Streaming was inert for every GGUF model.** `DType.Q4_K.SizeInBytes` is 0 (a K-quant has no
  per-element size), so `ElementCount * SizeInBytes` totalled block weights to **zero bytes** and the
  "fits resident?" test was always trivially true. Fixed in four block implementations.
- **Lens rendered solid black** (16/16). SageAttention's INT8 path materializes V as F16; Lens does not
  RMS-norm V, and `max|V|` crossed F16's 65504 mid-generation. Verified against ComfyUI's own reference
  implementation on the same checkpoint — an engine bug, not a port bug.
- **Anima was 19-63× slower than ComfyUI**: 792 host round-trips per denoise step (14 per block × 28
  blocks × 2 CFG passes). Now **3**. Warm step 15,279 ms → 519 ms. Its documented "1024² hangs" was never
  a hang.
- **The VRAM planner under-reported free memory by ~4.6 GB**, because `cuMemGetInfo` counts the
  stream-ordered pool's reservations as used. The error is asymmetric — it biases toward streaming, which
  costs 5-8× — so a large card could silently take the slow path for a model that fits.
- `TextService.PrimaryDeviceKey()` hardcoded `"cuda:0"`, so a `cuda:1` engine would have rendered images on
  one GPU while its LLM landed on another.
- Lumina2's on-disk checkpoint was the wrong variant (`cap_embedder.*` naming vs the diffusers
  `time_caption_embed.*` the converter expects). Correct weights now load and generate — though this
  revealed a **separate, previously unreachable conditioning bug**: output is coherent but off-prompt.

### Changed
- `Chroma` checkpoint conversion is now streaming per tensor (removes a GC-timing dependence from the
  peak). **The documented "host RAM OOM" does not reproduce** — it peaks at 9.1 GB anon and completes;
  the reported 25 GB was total RSS including reclaimable file-backed page cache.

## [1.0.0-alpha.48]

Production-readiness push: closes the throughput gap toward python inference stacks (vLLM/TGI-class) and
adds the serving infrastructure a real deployment needs. Full technical detail in
[`docs/Checklists/LLM_DECODE_PERF_GRIND.md`](docs/Checklists/ROADMAP.md)'s dated status
updates; this is the release-notes-level summary.

### Added
- **Fused GEMV kernels for Q4_0 and Q5_K** quantization formats — the last two of the six original
  quant types without a fused decode kernel; both previously fell to the ~10-20x-slower
  dequant-to-F16-then-cuBLAS path.
- **On-device repetition penalty for CUDA-graph decode.** Graph decode was previously greedy-only with a
  raw unpenalized argmax — a request with `RepetitionPenalty > 1.0` and graph decode enabled silently
  ignored the penalty. Fixed with two new device-resident kernels chained into the existing captured graph.
- **`/v1/chat/completions`** (OpenAI-compatible, streaming and non-streaming) on `HartsyInference.Server` —
  the server previously had no LLM chat endpoint at all (image generation only). Includes structured
  request logging (queue depth, prompt/completion tokens, latency, tokens/sec) and real cancellation that
  stops in-flight generation, not just the HTTP connection.
- **Paged KV cache** (`PagedKvPool`/`PagedKvCache`) — replaces the single-sequence `FixedKvCache` (hard
  `batch=1` restriction) with pages allocated on demand from a pool shared across sequences.
- **True continuous batching** (`DynamicBatchScheduler`/`IBatchScheduler`) — requests admit dynamically at
  any time and batch together into shared decode rounds; each sequence evicts the instant it
  finishes/stops/cancels. Replaces the old static-batch `ContinuousBatchScheduler` (fixed request list up
  front, zero production callers, removed). Backend-exclusivity is preserved via an injected gate so LLM
  batching never races with diffusion image generation on the shared GPU backend instance.
- **JSON-mode constrained decoding** (`response_format: {"type":"json_object"}`) — masks every candidate
  token so generation can only produce syntactically valid JSON. The richer `json_schema` mode is not
  implemented and is rejected with a clear 400 rather than silently ignored.
- Server integration test suite (`ChatCompletionsIntegrationTests`, in-process via `WebApplicationFactory`)
  covering chat-completions request validation — previously zero automated coverage on this HTTP surface.

### Changed
- `IBackend.SliceTimeRange` — new primitive (host default + CUDA kernel) extracting a contiguous
  time-range from a KV-shaped tensor; used by the paged KV cache.
- `GenericTransformer.ForwardBatchDecode`'s cache parameter widened from `FixedKvCache[]` to `IKvCache[]`.
- Chat-completions request validation now checks pure request-shape issues (empty messages, unsupported
  `response_format`) before consulting server state (is the model loaded) — fails fast on a malformed
  request regardless of what's currently loaded.

### Fixed
- Two real bugs in the new JSON-grammar state machine, both caught by unit tests before ever touching a
  live model: object keys didn't set the post-string parse transition (would have broken any JSON with a
  key — i.e. almost all real JSON); the state's `Clone()` was missing two fields added after it was first
  written (every candidate-token check clones the state, so this would have corrupted the container stack
  on every single trial in production).
- `ModelManager`'s diffusion-vs-LLM checkpoint routing no longer speculatively attempts the LLM loader on
  an unrecognized GGUF — a prior version of this logic (try-LLM-then-catch-fallback) fully materialized a
  multi-GB diffusion checkpoint's tensors before the fallback path could fire, causing a real OOM.
- Paged KV cache's VRAM footprint is now sized from a configurable byte budget
  (`HartsyInferenceServerOptions.KvPoolBytesBudget`, default 512MB) scaled to each loaded model's actual KV
  dimensions, replacing a fixed page count that comfortably fit a narrow-KV-dim model but eagerly
  pre-allocated several GB for a wider one — caught loading gemma-3 during a broader architecture sweep.

### Deferred (explicitly, not attempted)
- Prefix/prompt caching (share identical-content KV pages across sequences) — real additional scope (page
  reference-counting, prefix hashing, copy-on-write on divergence).
- Speculative decoding — a true stretch item, orthogonal to everything else in this release.
- `json_schema`-constrained decoding (schema-aware, not just syntax-valid JSON).
- Wider quant kernel coverage (Q2_K/Q3_K/IQx formats) — no template to adapt from, genuinely new kernel
  design (lookup-table dequant for IQx specifically).

## [1.0.0-alpha.47] and earlier

Not individually itemized here — see `git log` for the full history prior to this changelog's introduction.
