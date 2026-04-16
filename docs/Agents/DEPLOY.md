# Deploy Agent

> **Role:** Package SharpInference for NuGet publication. Handle versioning, package metadata, build configuration, pre-release testing, and publication.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — package list, dependencies, boundaries
- `docs/Checklists/RELEASE_NUGET.md` — the full release checklist
- All `.csproj` files in `src/` — current package configuration
- `Directory.Build.props` — shared build settings

## Your Workflow

1. **Verify readiness** — check phase checklists, all tests passing
2. **Configure packages** — metadata, version, dependencies
3. **Build release** — `dotnet build -c Release` with no warnings
4. **Run full test suite** — all unit + integration tests pass
5. **Pack** — `dotnet pack -c Release` produces correct .nupkg files
6. **Validate packages** — inspect contents, test local install
7. **Pre-release** — push preview packages to NuGet.org
8. **Stable release** — bump version, push stable packages
9. **Post-release** — tag git, create GitHub Release, announce

## Package Configuration Checklist

Each `.csproj` must have:

```xml
<PropertyGroup>
    <PackageId>SharpInference.Core</PackageId>
    <Version>1.0.0-preview.1</Version>
    <Authors>YourName</Authors>
    <Description>Core tensor types and backend abstractions for SharpInference AI inference engine</Description>
    <PackageLicenseExpression>MIT</PackageLicenseExpression>
    <PackageProjectUrl>https://github.com/yourorg/SharpInference</PackageProjectUrl>
    <PackageIcon>icon.png</PackageIcon>
    <PackageReadmeFile>README.md</PackageReadmeFile>
    <PackageTags>ai;inference;diffusion;cuda;dotnet</PackageTags>
    <IncludeSymbols>true</IncludeSymbols>
    <SymbolPackageFormat>snupkg</SymbolPackageFormat>
    <Deterministic>true</Deterministic>
</PropertyGroup>
```

## Versioning Strategy

- **Pre-release:** `1.0.0-preview.N` — for early adopter testing
- **Stable:** `1.0.0` — first production release
- **Patches:** `1.0.1` — bug fixes only
- **Minor:** `1.1.0` — new features, backward compatible
- **Major:** `2.0.0` — breaking API changes

All packages share the same version number for simplicity.

## Package Validation

Before publishing:
1. Create a fresh .NET project
2. Add package references from local NuGet feed
3. Write a minimal program that uses the package
4. Verify it compiles, runs, and produces correct output
5. Verify transitive dependencies resolve without conflicts

## Related Docs
- `docs/Checklists/RELEASE_NUGET.md` — full release checklist (follow this step by step)
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — what packages to publish
- `docs/Agents/TESTER.md` — testing standards before release
