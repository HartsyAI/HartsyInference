# Shared architecture and task routing

Read this with [code style](../CODE_STYLE.md). Load only the linked sections your task needs.

## Task routing

<task_routing>
Read the file that matches your task before you start. A task that spans two areas, such as adding a model and writing
its kernel, loads both files.

| Task | File |
|---|---|
| Add a model (any modality) | `ADD_MODEL.md` |
| Non-model feature (engine, CLI, API, extension) | `BUILD_FEATURE.md` |
| Review or audit code | `AUDIT.md` |
| GPU math, SIMD/PTX/SPIR-V kernels, performance | `KERNEL.md` |
| Research a topic before implementing | `RESEARCH.md` |
| Cleanup, doc upkeep, NuGet packaging | `CLEANUP.md` |
</task_routing>

## Shipping a change

<shipping>
The author is `kalebbroo <kalebbroo@gmail.com>`, alone. Leave out co-author, session and tool-attribution trailers from
commit messages, PR bodies and the changelog.

Work through these steps in order:

1. **Branch, then push as soon as you have commits.** Unpushed non-`main` branches get reaped. The checkout is shared
   with other agents, so run `git branch --show-current` before every commit. If a commit lands on someone else's
   branch, rescue it with a cherry-pick. Never rewrite their branch.
2. **Open a draft PR early**, before the work is finished, so reviewers and the bots can see progress.
3. **Finish the implementation and test it against the PR.**
   - Run the CPU lane and the tests that cover the changed area. The trait filters are in [testing](../CODE_STYLE.md#testing).
     A docs-only change needs a link and claim check instead.
   - Run GPU suites one at a time. Never run one alongside another suite, a benchmark or a generation: concurrent runs
     create contention failures that bury real ones. Record failures by test name, not by count.
   - For a CUDA or Vulkan change, run the whole `Category=GpuIntegration` category for the affected backends, not only
     the tests the change seems to touch. A dispatch-gate change must run on every weight of that dtype.
   - Anything that can change output or speed needs a real-generation A/B. Build `origin/main` fresh (PTX included)
     and the branch, then run them by hand on the same prompts, seeds and settings, alternating the arms so clock and
     thermal drift hit both equally, at least two repetitions each, keeping every run's output. Report wall clock,
     per-step ms, VRAM, and SSIM or a raw-pixel digest for each pair. State before you run it whether the expectation
     is identical (a refactor) or bounded (a numerics change, with a floor and a reason). Put the table in the PR body.
     CUDA picks the F32 attention route and tile height (and MiniMax-H3 its chunk rows) from free VRAM. A change that
     frees memory earlier can therefore move bits on a card near the threshold. Before you call an identity failure a
     bug, rerun both arms with `vram.sdpaForceTiled=true`, `numerics.sdpaTile=256` and `vram.h3ChunkRows=1000000` pinned.
4. **Final pass.** Clean up comments and docs. Bump the version only for a new feature or a large bug fix. Such a bump
   needs a `## alpha.N` section in `CHANGELOG.md` and the matching `VersionSuffix` in `Directory.Build.props`, one bump
   per PR. Small changes, docs-only ones included, skip both. A merge to `main` that touches `src/` publishes whatever
   version is already set, so a small change ships with the next bump and its changelog section.
5. **Mark the PR ready for review.** `claude-code-review.yml` reviews on open, on each push and on ready. The Codex
   connector reviews separately. If Codex reports a usage limit, no review happened: comment `@claude review pr` on the
   PR and wait for `claude[bot]`.
6. **Resolve every comment.** Reviews and comments are separate API objects, so check all three surfaces:

   ```bash
   gh api --paginate repos/HartsyAI/HartsyInference/issues/<N>/comments --jq '.[]|"\(.user.login) \(.created_at)"'
   gh api --paginate repos/HartsyAI/HartsyInference/pulls/<N>/reviews  --jq '.[]|"\(.user.login) \(.state)"'
   gh api --paginate repos/HartsyAI/HartsyInference/pulls/<N>/comments --jq '.[]|"\(.path):\(.line)"'
   ```

   Reply to each thread, then resolve it with the GraphQL `resolveReviewThread` mutation. `gh pr` has no resolve command.
7. **Merge one PR at a time**, once CI is green and every thread is resolved. Each merge makes the others' version bump
   and changelog insert stale. A single push that carries several merges publishes only the top version.

Report in plain language. Keep answers short, do not restate what the diff already shows, and close a long answer or a
research session with a short plain-English TL;DR.
</shipping>

## Shared design rules

<design_rules>
These apply to every agent. Specialist files add only task-specific rules.

**Pure C# runtime and model implementation.** Do not add Python or C++ inference wrappers, ONNX Runtime, or managed GPU
frameworks; the engine stays native. The existing boundaries are vendor driver and compute-library P/Invoke, CUDA
sources compiled to PTX, and offline Python reference tools.

**Eager model execution.** Backend CUDA graph capture and replay is a backend optimization. It is not a model graph.

**No GC on hot paths.** Allocate no managed memory during inference. Use `NativeMemory.AlignedAlloc(byteCount, 64)` for
tensor storage, and `ArrayPool<T>.Shared` only for managed metadata. `TensorPool` has no production call site yet, so
adopt it or retire it (see ROADMAP). Do not cite it as established practice.

**Backends behind `IBackend`.** Model code never calls CPU, CUDA or Vulkan directly. Each backend delegates to static
kernels.

**Package boundaries.** Each NuGet package gets one folder under `src/`. Dependencies run one way:
`Core` ← modality packages ← `Engine` ← CLI, API and extension. Keep CUDA and Vulkan out of the CPU-only packages; GPU
code sits behind `IBackend`. When you are unsure where code belongs, match the package a sibling model or feature
already uses.

**Reuse before writing anything.** The backend is modular so that models share it. Before you add any helper, inline or
shared, search for an existing primitive. Check the `IBackend` operations first (`Transpose2D`, `Conv1d`/`ConvTranspose1d`,
`Snake`, `Silu`, `GroupNorm`, `ScaledDotProductAttention`, and others), then the shared statics below. For example, a
`[1,C,T]↔[1,T,C]` layout transpose is `backend.Transpose2D(out, in, d1, d2)`, not a hand-written loop.

| Package | Shared statics |
|---|---|
| `Core` | `TensorCasts` host dtype casts (`EnsureF32`, `LoadF32`/`LoadF32Opt`, `F32ToBf16Bits`, `RelabelRank2Copy`), `ByteFormat` for VRAM log lines, `Runtime/MonotonicClock` (monotonic ns + absolute-deadline sleep), `Runtime/RealtimeScheduling` (`TryEnterFifo`, `TryPinToCpu`), `Runtime/SpscRing<T>` (lock-free single-producer/single-consumer ring), `Numerics/LatencyHistogram` (fixed µs buckets, p50/p99/max), `CpuParallel.InlineScope` (thread-local "never fan out"), `Rope/` RoPE tables (`RopeFrequencyPlan` per-axis or sectioned (M-RoPE) frequencies, `MultiAxisRopeTables` pairs/rows/tensors from positions, `RopeTableExpander` for the rotate-half and interleaved layouts, `RopeTableCache` device rows per backend, or host-staged rows for callers without one, with leases; `MRopePositions` / `MRopePairAxes` / `MRopeTable` for Qwen-VL multimodal RoPE). A model owns only its position rules. |
| `Diffusion` | `DiTUtils` (denoiser blocks), `CfgHelper` (`SliceBatchElement`, `ApplyCfg`, `ConcatLastDim`), `DtypeCastHelper` (backend-routed, source-disposing casts), `Img2ImgSetup.Prepare`, `Schedulers/SchedulerFactory.Create`, `NoiseSchedule`, `VaeOps`/`MageVaeOps`, `WeightBytes`; in-context references: `TokenSegmentLayout` + `SegmentModulation` (per-segment adaLN), `ReferenceKvCache` (prepass keys/values appended each step, through an `IKvRewrite` on the shared `AttentionLayer`), `Flux2ReferenceEncoder`, `ReferenceCachePolicy` |
| `ModelAssets` | `CheckpointConvertUtils` (key remaps, quant-aware QKV splits), `VisionPlaceholderPrompt` + `VisionTokenSplice` (Qwen-VL image placeholders and their embedding splice) |
| `Engine` | `ReferenceImageResolver` + `ReferenceImagePolicy` (an edit request's references, sized for the VAE and the vision tower) |
| `Audio` | `WhisperOps.ProjectLinear`, `Layers/Activations` (`ErfGelu`, `SigmoidS`), `RnnOps`, `VqOps`, `WeightNormFusion.LoadFused`, `LogitSampling`, `SignalPadding`, `IStft`, `Dsp/NsfVocoderDsp` (NSF source, STFT/iSTFT head, pad, scale), `Dsp/DeterministicRng`, `Streaming/SentenceChunkedSynthesis` (sentence streaming over any whole-utterance synth; Piper and Kokoro use it), `Frontends/SentenceSplitter` + `StreamingSentenceSplitter` + `SplitClauses`, `Frontends/SpokenTextNormalizer`, `Io/G711` (μ-law/A-law), `Models/Wake/IVadModel` |

Choose a shared helper by ownership, not by name. `TensorCasts` works on the host, while `DtypeCastHelper` routes through
the backend and disposes its source. Keep new shared code generic. When two or more callers need the same operation,
write one helper parameterized by the differences: a few extra parameters or a `switch` beats a dozen near-identical
methods. When you add a model, audit it against the models already built and fold the shared parts into common code.
Re-run the affected models' tests afterwards, because shared code is load-bearing.
</design_rules>

## Engine patterns

Tensor ownership, CUDA launch and PTX, config, streaming, error handling, video planning, GPU weight and activation
management, and diffusion pipeline conventions are in [ENGINE_PATTERNS.md](ENGINE_PATTERNS.md). Read it when you touch
engine, backend or pipeline code. The design rules above stand on their own for everything else.

Ownership in one line: the creator owns and disposes a `Tensor`. A `TensorView` or `TensorRef` is borrowed and must never
outlive the storage behind it.
