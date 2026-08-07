# Contributing to HartsyInference

Thanks for your interest in contributing to HartsyInference.

## Getting Started

You need the [.NET 10 SDK](https://dotnet.microsoft.com/download) and git. An NVIDIA GPU with CUDA 12.x
or 13.x is optional — the CPU and Vulkan backends work without one.

```bash
git clone https://github.com/your-org/HartsyInference.git
cd HartsyInference
dotnet build
```

## Running Tests

```bash
dotnet test                                        # unit lane — no GPU, no checkpoints (the default)
dotnet test --filter "Category=Integration"        # requires GPU + model files
dotnet test --filter "FullyQualifiedName~Parity"   # all shared-component parity tests
```

**Before adding a test, read [`docs/CODE_STYLE.md`](docs/CODE_STYLE.md) §Testing.** A test earns its place
only if its failure would be *silent*. Do not add one that proves a model works end to end — a broken model
is visible the moment anyone uses it. Test what breaks quietly: kernel numerics, cross-device and
cross-backend equivalence, quantization and codec round-trips, tensor lifetime and concurrency,
padding/tiling geometry, format and key mapping. Shared-component parity goes in `tests/<Project>/Parity/`
and must end in `*ParityTests`.

## Project Structure

One folder per NuGet package under `src/`, with a matching test project under `tests/`. The dependency
direction is one-way: `Core` ← modality packages ← `Engine` ← CLI/API/extension. Code should stay inside
its package's responsibility, and CPU-only packages must never take a dependency on CUDA or Vulkan — GPU
code lives behind `IBackend` in the backend packages.

`HartsyInference.Engine` owns "load a model + generate". The CLI, the HTTP API, and the SwarmUI extension
are thin wrappers over it; don't re-implement load/generate orchestration in a consumer.

## Submitting Changes

Fork, branch from `main`, make your change, keep the tests passing, and open a pull request with a clear
description. Include your .NET version, GPU model, CUDA version, and OS on any issue report, plus a minimal
reproduction where you can.

## Coding Standards

[`docs/CODE_STYLE.md`](docs/CODE_STYLE.md) is the mandatory, authoritative reference — read it before your
first change. The rules that catch people out most often:

- **Pure C#** — no native shared libraries, no Python, no C++ wrappers
- **Unmanaged memory** for tensor data — `NativeMemory.AlignedAlloc`, never managed arrays on hot paths
- **`IDisposable`** on anything holding unmanaged resources
- **`IBackend` abstraction** — model code never calls CPU, CUDA, or Vulkan kernels directly
- **File-scoped namespaces**, `sealed` by default, `readonly` where possible
- **XML doc comments** on all public APIs
- **No warnings** — `TreatWarningsAsErrors` is on

Public signatures are effectively append-only: the SwarmUI backend extension pins a *published* engine
version, so a renamed or changed public signature stays invisible until it is republished and re-pinned.
Add an overload instead.

### Kernels

Every kernel needs a scalar fallback, FP32 accumulation even for FP16 inputs, and validation against a
reference implementation before it ships. AVX2 is the baseline SIMD target and AVX-512 is optional; PTX
targets `sm_80` minimum. `.cu` and `.comp.glsl` sources are the source of truth — the committed `.ptx` and
`.spv` files are build artifacts, never hand-edited.

See [`docs/Agents/KERNEL.md`](docs/Agents/KERNEL.md) for the full kernel reference, including the launch
pattern, the toolchain gotchas, and the validation tolerances.

## Architecture and Background

- [`docs/README.md`](docs/README.md) — the docs map: what each folder is for, and the rule for adding to it
- [`docs/Agents/AGENTS.md`](docs/Agents/AGENTS.md) — shared design rules and core engine patterns (the
  architecture single source of truth)
- [`docs/Checklists/TROUBLESHOOTING.md`](docs/Checklists/TROUBLESHOOTING.md) — bring-up debugging reference;
  read it first when a model is wrong, crashes, or is slow
- [`docs/Checklists/ROADMAP.md`](docs/Checklists/ROADMAP.md) — cross-cutting open work
- [`docs/Research/`](docs/Research/) — per-model and per-technique research notes

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
