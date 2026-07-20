# Deploy Agent

> Package HartsyInference for NuGet — versioning, metadata, build, pre-release testing, publication.

## Extra Reading
- `docs/Design/NUGET_PACKAGE_DESIGN.md`, `docs/Checklists/RELEASE_NUGET.md`
- All `.csproj` files and `Directory.Build.props`

## Workflow
1. Verify readiness (checklists, tests passing)
2. Configure packages (metadata, version, dependencies)
3. `dotnet build -c Release` (no warnings)
4. Run full test suite
5. `dotnet pack -c Release`
6. Validate packages (inspect, local install test)
7. Push preview → stable
8. Tag git, create GitHub Release, announce

## Package Configuration
Each `.csproj` must have:
```xml
<PropertyGroup>
    <PackageId>HartsyInference.Core</PackageId>
    <!-- Version is centralized; do NOT hardcode per-project. See Versioning below. -->
    <Authors>Hartsy</Authors>
    <Description>...</Description>
    <PackageLicenseExpression>MIT</PackageLicenseExpression>
    <PackageProjectUrl>https://github.com/HartsyAI/HartsyInference</PackageProjectUrl>
    <PackageIcon>icon.png</PackageIcon>
    <PackageReadmeFile>README.md</PackageReadmeFile>
    <PackageTags>ai;inference;diffusion;cuda;dotnet</PackageTags>
    <IncludeSymbols>true</IncludeSymbols>
    <SymbolPackageFormat>snupkg</SymbolPackageFormat>
    <Deterministic>true</Deterministic>
</PropertyGroup>
```
Target frameworks: **net8.0 and net10.0** (multi-target both).

## Versioning
Single source of truth is `Directory.Build.props` (`<VersionSuffix>`); all packages share that one version (currently `1.0.0-alpha.43`). Bump it there every release, never per-project. Progression: `1.0.0-alpha.N` → `1.0.0` → `1.0.1` (patch) / `1.1.0` (minor) / `2.0.0` (major). The SwarmUI backend extension pins a published engine version, so any new public type is invisible to it until this is bumped, published, AND re-pinned in the extension's csproj.

## What Ships
Publish the real modality packages (Core, Cpu, Cuda, Vulkan, ModelHandler, Tokenizers, Diffusion, Audio, Vision, Video, ThreeD, Interactive), plus **HartsyInference.LLM** and **HartsyInference.Audio.Phonemizer**, and the meta package **HartsyInference** (references everything except LLM, Phonemizer, Server, and Cli, so LLM/Phonemizer are added explicitly by consumers). Do NOT publish or advertise **HartsyInference.API**; the OpenAI-compatible server is dropped and that project is abandoned scaffolding. The engine is consumed via the SwarmUI backend extension (`https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend`), the libraries above, and the sample CLIs.

## Validation
Before publishing: create fresh project, add local packages, verify compile/run/output and transitive dependency resolution.
