# Contributing to HartsyInference

Thanks for your interest in contributing to HartsyInference.

## Getting Started

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- NVIDIA GPU with CUDA 12.x (optional — CPU backend works without it)
- Git

### Building

```bash
git clone https://github.com/your-org/HartsyInference.git
cd HartsyInference
dotnet build
```

### Running Tests

```bash
# Unit tests (no GPU, no checkpoints) — this is the default lane
dotnet test

# Integration tests (requires GPU + model files)
dotnet test --filter "Category=Integration"

# All shared-component parity tests
dotnet test --filter "FullyQualifiedName~Parity"
```

**Before adding a test, read `docs/CODE_STYLE.md` §Testing.** A test earns its place only if its
failure would be *silent*. Do not add one that proves a model works end to end — a broken model is
visible the moment anyone uses it. Test what breaks quietly: kernel numerics, cross-device and
cross-backend equivalence, quantization and codec round-trips, tensor lifetime and concurrency,
padding/tiling geometry, format and key mapping. Shared-component parity goes in
`tests/<Project>/Parity/` and must end in `*ParityTests`.

## Project Structure

See [File Structure](docs/Design/FILE_STRUCTURE.md) for the full layout.

Each NuGet package lives in its own folder under `src/`. Tests are in `tests/` with a matching project name. See [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) for package boundaries — code should stay within its package's responsibility.

## How to Contribute

### Reporting Issues

- Use GitHub Issues
- Include .NET version, GPU model, CUDA version, and OS
- Include a minimal reproduction if possible

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes following the coding standards below
4. Write or update tests
5. Ensure all tests pass
6. Submit a pull request with a clear description

### Coding Standards

- **Pure C#** — no native shared libraries, no Python, no C++ wrappers
- **Unmanaged memory** for tensor data — `NativeMemory.AlignedAlloc`, never managed arrays on hot paths
- **IDisposable** on anything holding unmanaged resources
- **IBackend abstraction** — model code never calls CPU or CUDA kernels directly
- **File-scoped namespaces**, `sealed` classes by default, `readonly` where possible
- **XML doc comments** on all public APIs
- **No warnings** — `TreatWarningsAsErrors` is enabled

See [Builder Agent](docs/Agents/BUILDER.md) for the full coding standards reference.

### Kernel Contributions

SIMD CPU kernels and PTX GPU kernels have additional requirements:

- Every kernel must have a scalar fallback
- AVX2 is the baseline SIMD target; AVX-512 is optional
- PTX targets `sm_80` minimum (Ampere+)
- Every kernel must be validated against a reference implementation

See [Kernel Agent](docs/Agents/KERNEL.md) for the full kernel standards reference.

### Testing Requirements

- Every public API method needs at least one test
- Every kernel needs a correctness test against reference values
- Pipeline changes need integration tests with fixed seeds
- See [Validation Strategy](docs/Design/VALIDATION_STRATEGY.md) for reference implementations and tolerances

## Architecture Decisions

Before proposing architectural changes, read:

- [Core Design](docs/Design/CORE_DESIGN.md) — design pillars and key decisions
- [Implementation Details](docs/Design/IMPLEMENTATION_DETAILS.md) — why things are built the way they are
- [Build Order](docs/Design/BUILD_ORDER.md) — phase dependencies

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
