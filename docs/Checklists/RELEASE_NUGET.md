# Release — NuGet Publication

> **Goal:** Publish all SharpInference packages to NuGet.org.

---

## 1. Pre-Release Preparation

**Branding:** Package icon, descriptions, README, project URL, license expression, tags. Verify/reserve IDs on NuGet.org.

**Versioning:** SemVer, initial version (1.0.0-preview.1 or 0.1.0), `Directory.Build.props` version props, CI auto-increment.

**Documentation:** Main README with quickstart, CONTRIBUTING.md, API docs (XML comments), samples project, getting-started guide, docs site.

## 2. Quality Gates

**Code:** All public APIs documented, no compiler warnings (treat as errors), static analysis clean, no known vuln deps, coverage >80% (Core/ModelHandler/Cpu), >70% (Diffusion/Audio/Vision).

**Testing:** All unit + integration + golden reference tests pass, 24hr memory soak, cross-platform build (Windows + Linux), benchmarks documented.

**Compatibility:** .NET 10 target builds, transitive deps resolve, minimum install scenarios work, OpenAI Python SDK compat, SwarmUI compat.

## 3. Package Build

- [ ] All .csproj metadata complete (PackageId, Version, Authors, Description, License, Icon, Tags, Symbols)
- [ ] Source Link + .snupkg configured, deterministic builds
- [ ] PTX files + tokenizer vocab included as resources
- [ ] `dotnet build -c Release` (no warnings), `dotnet test -c Release` (all pass), `dotnet pack -c Release`
- [ ] Verify .nupkg contents and dependency graph
- [ ] Local NuGet feed install test

## 4. Pre-Release

- [ ] Push 1.0.0-preview.1 to NuGet.org
- [ ] Fresh project install test from NuGet.org
- [ ] Gallery pages look correct
- [ ] Community feedback, fix issues

## 5. Stable Release

- [ ] Bump to stable (1.0.0), final pack, push
- [ ] Git tag `v1.0.0`, GitHub Release with changelog
- [ ] Announce

## 6. Post-Release

- [ ] Monitor downloads/errors, GitHub issue templates, Dependabot, plan next release
