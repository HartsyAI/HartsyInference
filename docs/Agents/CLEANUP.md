# Cleanup & Format Agent

> Refactor for real reasons, keep docs/checklists in sync with the code, and handle NuGet packaging.
> Assumes you've read `AGENTS.md` + `docs/CODE_STYLE.md`. Code is the source of truth; docs follow it.

## Refactor discipline

```text
✅ valid motivations: a profiled perf bottleneck · genuine code duplication (2+ call sites) ·
   a package-boundary violation (route it through IBackend) · hot-path temps that should use TensorPool
❌ invalid: "could be cleaner" · a hypothetical future abstraction (YAGNI) · preference renames ·
   non-functional file shuffling
```

Safety rules: verify test coverage first (write the test if it's missing); **one change per commit**, never
mixed with feature work; all tests pass with **numerically identical** results within the existing tolerance;
don't change a public signature (see `BUILD_FEATURE.md` — the extension pins a published version). If the
motivation is perf, baseline-benchmark before and after.

```csharp
// ✅ the canonical refactor: hoist duplication into ONE parameterized shared helper
//    src/HartsyInference.Diffusion/Utilities/CfgHelper.cs — "Centralizes the batch-slice + CFG-combine
//    duplicated across SD1.5, SDXL, SD3, and every CFG pipeline." (also DtypeCastHelper, Img2ImgSetup)
CfgHelper.ApplyCfg(...); DtypeCastHelper.EnsureF32(...);
// ❌ copying the same CFG-slice / dtype-cast / img2img-prep loop into each new pipeline
```

## Docs & status sync

- Scan for drift (docs vs. code); keep `README.md` short — description, NuGet quickstart, minimal examples;
  try to compile any doc code snippet. Present tense, active voice.
- **Model-status drift** is the highest-value catch: a model verified end-to-end but still shown `🔧` in its
  `MODEL_STATUS_*` doc → update that table and record the evidence in `PARITY_VERIFICATION.md`.

## Checklist hygiene

- Right list: cross-cutting work → `ROADMAP.md`; per-model work → a `MODEL_STATUS_*` doc's `Remaining work`
  section. When an item ships, **delete the line** — the surviving docs track *open* work and *verified*
  status, not a running history (git is the archive). Move any generalizable bug lesson into
  `TROUBLESHOOTING.md`. An item is done only when merged **and** the status table reflects it.

## NuGet packaging

```xml
<!-- ✅ version is a single source of truth in Directory.Build.props; every project inherits it -->
<VersionPrefix>2.0.0</VersionPrefix><VersionSuffix>alpha.5</VersionSuffix>
<!-- ❌ a hardcoded <Version> in an individual .csproj -->
```

- Multi-target **net8.0 + net10.0**; build Release with no warnings; validate with a fresh consumer project
  before pushing. What ships: the modality packages + `HartsyInference.LLM` + `HartsyInference.Audio.Phonemizer`
  + the meta `HartsyInference` package; `HartsyInference.API` is `IsPackable=false` (a runnable app).
- **Publish order** (get this wrong and the extension build breaks): bump the engine version → publish to
  nuget.org → let it land → *then* re-pin the version in the SwarmUI extension's csproj (its default build
  restores from nuget.org; local dev uses `-p:UseLocalHartsy=true`).
