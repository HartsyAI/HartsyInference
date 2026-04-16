# Release — NuGet Publication

> **Goal:** Publish all SharpInference packages to NuGet.org as production-ready releases.

---

## 1. Pre-Release Preparation

### Branding & Metadata
- [ ] Finalize package icon (consistent across all packages)
- [ ] Write NuGet package descriptions for each package
- [ ] Write package README for NuGet gallery page
- [ ] Set up project URL (GitHub repo)
- [ ] Set up license expression in all .csproj files
- [ ] Add package tags for discoverability (ai, inference, diffusion, cuda, etc.)
- [ ] Verify package IDs are available on NuGet.org
- [ ] Reserve package name prefixes on NuGet.org

### Versioning
- [ ] Decide versioning strategy (SemVer)
- [ ] Set initial version (1.0.0-preview.1 or 0.1.0)
- [ ] Configure `Directory.Build.props` with version properties
- [ ] Set up version auto-increment for CI builds

### Documentation
- [ ] Write main README.md with quickstart guide
- [ ] Write CONTRIBUTING.md
- [ ] Write API documentation (XML doc comments on all public APIs)
- [ ] Create samples project with working examples
- [ ] Write migration/getting-started guide
- [ ] Publish documentation site (GitHub Pages or similar)

## 2. Quality Gates

### Code Quality
- [ ] All public APIs have XML documentation comments
- [ ] No compiler warnings (treat warnings as errors)
- [ ] Static analysis clean (dotnet analyzers)
- [ ] No known security vulnerabilities in dependencies
- [ ] Code coverage > 80% for Core, ModelHandler, Cpu packages
- [ ] Code coverage > 70% for Diffusion, Audio, Vision packages

### Testing
- [ ] All unit tests pass
- [ ] All integration tests pass on GPU CI
- [ ] Golden reference validation tests pass for all supported models
- [ ] Memory leak tests pass (24-hour soak test)
- [ ] Cross-platform build verification (Windows, Linux)
- [ ] Performance benchmarks documented and acceptable

### Compatibility
- [ ] Verify .NET 10 target framework builds correctly
- [ ] Verify transitive dependency resolution (no conflicts)
- [ ] Test minimum install scenarios (each use case from NuGet Package Design)
- [ ] Test with OpenAI Python SDK (server compatibility)
- [ ] Test SwarmUI extension with latest SwarmUI release

## 3. Package Build

### NuGet Package Configuration
- [ ] Each .csproj has correct `<PackageId>`, `<Version>`, `<Authors>`, `<Description>`
- [ ] Each .csproj has `<PackageLicenseExpression>`, `<PackageProjectUrl>`, `<PackageIcon>`
- [ ] Each .csproj has `<PackageTags>` and `<PackageReadmeFile>`
- [ ] Source Link configured for debugger source stepping
- [ ] Symbol packages (.snupkg) configured for NuGet symbol server
- [ ] `<IncludeSymbols>true</IncludeSymbols>` and `<SymbolPackageFormat>snupkg</SymbolPackageFormat>`
- [ ] Deterministic builds enabled (`<Deterministic>true</Deterministic>`)
- [ ] PTX files included as embedded resources in Cuda package
- [ ] Tokenizer vocab/merges files included as embedded resources

### Build & Pack
- [ ] `dotnet build -c Release` succeeds with no warnings
- [ ] `dotnet test -c Release` all pass
- [ ] `dotnet pack -c Release` produces all .nupkg files
- [ ] Verify .nupkg contents (correct files, no extra junk)
- [ ] Verify package dependency graph is correct in each .nuspec
- [ ] Test local NuGet feed install — create new project, add packages, verify they work

## 4. Pre-Release Testing

- [ ] Push packages to NuGet.org as pre-release (e.g., 1.0.0-preview.1)
- [ ] Install from NuGet.org in a fresh project — verify packages resolve correctly
- [ ] Run quickstart sample against NuGet packages (not project references)
- [ ] Verify NuGet gallery pages look correct (description, icon, README)
- [ ] Collect community feedback on pre-release
- [ ] Fix any issues found during pre-release

## 5. Stable Release

- [ ] Bump version to stable (e.g., 1.0.0)
- [ ] Final `dotnet pack -c Release`
- [ ] Push all packages to NuGet.org
- [ ] Verify all packages appear on NuGet.org
- [ ] Tag release in git (`v1.0.0`)
- [ ] Create GitHub Release with changelog
- [ ] Announce release (blog post, social media, relevant communities)

## 6. Post-Release

- [ ] Monitor NuGet download counts and error reports
- [ ] Set up GitHub issue templates (bug report, feature request)
- [ ] Set up automated dependency update (Dependabot)
- [ ] Plan next release cycle (Phase 2 models, performance improvements)
- [ ] Document known limitations and roadmap
