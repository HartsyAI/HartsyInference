# Shared architecture and task routing

Read with [code style](../CODE_STYLE.md); load only the linked sections a task needs.

## Task Routing

Read the file matching your task before starting. A task spanning two (add a model *and* write its kernel) loads both.

| Task | File |
|---|---|
| Add a model (any modality) | `ADD_MODEL.md` |
| Non-model feature (engine, CLI, API, extension) | `BUILD_FEATURE.md` |
| Review or audit code | `AUDIT.md` |
| GPU math, SIMD/PTX/SPIR-V kernels, performance | `KERNEL.md` |
| Research a topic before implementing | `RESEARCH.md` |
| Cleanup, doc upkeep, NuGet packaging | `CLEANUP.md` |

## Shipping a change

Author is `kalebbroo <kalebbroo@gmail.com>` alone: no co-author, session, or tool-attribution trailers in any
commit message or PR body, and none in the changelog.

1. Branch and **push as soon as commits exist** — unpushed non-`main` branches get reaped. Check
   `git branch --show-current` before every commit; the checkout is shared with other agents and a commit
   landing on someone else's branch is rescued with a cherry-pick, never by rewriting theirs.
2. Open a **draft PR** early, before the work is finished.
3. Finish implementation and all testing against it:
   - Run the CPU lane and the tests covering the changed area; see [testing](../CODE_STYLE.md#testing) for the
     trait filters. A docs-only change needs a link and claim check instead.
   - GPU suites run one at a time, never alongside another suite, benchmark, or generation — concurrent runs
     manufacture contention failures that bury real ones. Record failures by test name, not count.
   - A CUDA/Vulkan change runs the whole `Category=GpuIntegration` category for the affected backends, not the
     tests it appears to touch: a dispatch-gate change runs on every weight of that dtype.
   - Anything that can move output or speed needs a real-generation A/B: `origin/main` built fresh (PTX
     included) against the branch, interleaved on the same seeds, reporting wall clock, per-step ms, VRAM and
     SSIM or raw-pixel digest per pair, via `tests/regression-ab.sh`. State up front whether the expectation
     is identical (refactor) or bounded (numerics change, with a floor and a reason). Table goes in the PR body.
4. Final pass: comment/doc cleanup, a `## alpha.N` section in `CHANGELOG.md`, and the matching
   `Directory.Build.props` `VersionSuffix` bump — one per PR, docs-only PRs skip both. The bump is not
   optional for code: the engine version is what the SwarmUI extension's NuGet pin resolves against.
5. Mark ready for review. `claude-code-review.yml` reviews on open, on each push and on ready; the Codex
   connector reviews independently. A Codex usage-limit note means no review happened — comment
   `@claude review pr` on the PR and wait for `claude[bot]`.
6. Resolve every comment. Reviews and comments are different API objects, so check all three surfaces:
   ```bash
   gh api repos/HartsyAI/HartsyInference/issues/<N>/comments --jq '.[]|"\(.user.login) \(.created_at)"'
   gh api repos/HartsyAI/HartsyInference/pulls/<N>/reviews  --jq '.[]|"\(.user.login) \(.state)"'
   gh api repos/HartsyAI/HartsyInference/pulls/<N>/comments --jq '.[]|"\(.path):\(.line)"'
   ```
   Reply per thread, then resolve it with the GraphQL `resolveReviewThread` mutation — `gh pr` has no resolve verb.
7. Merge once CI is green and every thread is resolved, one PR at a time: each merge stale-dates the
   others' version bump and changelog insert, and a single push carrying several merges publishes only the
   top version.

Report in plain language. Keep answers terse, skip restating what the diff already shows, and close a long
answer or a research session with a short plain-English TL;DR.

## Shared Design Rules

These apply to ALL agents; specialized files only add task-specific rules.

**Pure C# runtime and model implementation** — no Python/C++ inference wrappers, ONNX Runtime, or managed GPU
frameworks. Vendor driver/compute-library P/Invoke, CUDA sources compiled to PTX, and offline Python reference
tools are existing boundaries.

**Eager model execution** — backend CUDA graph capture/replay is a backend optimization, not a model graph.

**Zero GC on hot paths** — no managed allocations during inference. `NativeMemory.AlignedAlloc(byteCount, 64)`;
`ArrayPool<T>.Shared` for managed metadata only. (`TensorPool` still has no production call site — adopt or
retire it, see ROADMAP; do not cite it as established practice.)

**IBackend abstraction** — model code never calls CPU/CUDA/Vulkan directly. Each backend delegates to static kernels.

**Package boundaries** — one folder per NuGet package under `src/`; dependencies run one way
(`Core` ← modality packages ← `Engine` ← CLI/API/extension). No CUDA/Vulkan in CPU-only packages — GPU code lives
behind `IBackend`. When unsure, match the package a sibling model or feature already lives in.

**Reuse before writing anything.** The backend is modular so models share it. Before adding ANY helper, inline or
shared, grep for an existing primitive: `IBackend` ops first (`Transpose2D`, `Conv1d`/`ConvTranspose1d`, `Snake`,
`Silu`, `GroupNorm`, `ScaledDotProductAttention`, …), then the shared statics below. A `[1,C,T]↔[1,T,C]` layout
transpose is `backend.Transpose2D(out, in, d1, d2)`, never a hand-rolled loop.

| Package | Shared statics |
|---|---|
| `Core` | `TensorCasts` host dtype casts (`EnsureF32`, `LoadF32`/`LoadF32Opt`, `F32ToBf16Bits`, `RelabelRank2Copy`), `ByteFormat` for VRAM log lines |
| `Diffusion` | `DiTUtils` (denoiser blocks), `CfgHelper` (`SliceBatchElement`, `ApplyCfg`, `ConcatLastDim`), `DtypeCastHelper` (backend-routed, source-disposing casts), `Img2ImgSetup.Prepare`, `Schedulers/SchedulerFactory.Create`, `NoiseSchedule`, `VaeOps`/`MageVaeOps`, `WeightBytes` |
| `ModelAssets` | `CheckpointConvertUtils` (key remaps, quant-aware QKV splits) |
| `Audio` | `WhisperOps.ProjectLinear`, `Layers/Activations` (`ErfGelu`, `SigmoidS`), `RnnOps`, `VqOps`, `WeightNormFusion.LoadFused`, `LogitSampling`, `SignalPadding`, `IStft`, `Dsp/NsfVocoderDsp` (NSF source, STFT/iSTFT head, pad, scale), `Dsp/DeterministicRng` |

Pick by ownership, not by name: `TensorCasts` is host-side, `DtypeCastHelper` routes through the backend and
disposes its source. New shared code stays generic: when two or more callers need an operation, hoist ONE helper
**parameterized by the differences** — a few extra parameters or a `switch` beats a dozen near-identical methods.
Adding a model includes auditing it against models already built and folding the shared parts; re-run the
affected models' tests afterwards, because shared code is load-bearing.

## Engine Patterns

Tensor ownership, CUDA launch/PTX, config, streaming, error handling, video planning, GPU weight and activation
management, and diffusion pipeline conventions live in [ENGINE_PATTERNS.md](ENGINE_PATTERNS.md). Read it when
touching engine, backend or pipeline code; the rules above stand on their own for everything else.

Ownership in one line: `Tensor` is owned and disposed by its creator, `TensorView` and `TensorRef` are borrowed
and must never outlive the storage behind them.
