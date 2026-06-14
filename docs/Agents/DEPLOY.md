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
    <Version>1.0.0-preview.1</Version>
    <Authors>YourName</Authors>
    <Description>...</Description>
    <PackageLicenseExpression>MIT</PackageLicenseExpression>
    <PackageProjectUrl>https://github.com/...</PackageProjectUrl>
    <PackageIcon>icon.png</PackageIcon>
    <PackageReadmeFile>README.md</PackageReadmeFile>
    <PackageTags>ai;inference;diffusion;cuda;dotnet</PackageTags>
    <IncludeSymbols>true</IncludeSymbols>
    <SymbolPackageFormat>snupkg</SymbolPackageFormat>
    <Deterministic>true</Deterministic>
</PropertyGroup>
```

## Versioning
All packages share one version: `1.0.0-preview.N` → `1.0.0` → `1.0.1` (patch) / `1.1.0` (minor) / `2.0.0` (major).

## Validation
Before publishing: create fresh project, add local packages, verify compile/run/output and transitive dependency resolution.
