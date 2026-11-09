# Cleanup and documentation

Use this file for refactors, doc upkeep and packaging.

- Refactor only for a measured cost, real duplication, or a boundary or ownership defect. Preserve public compatibility
  and numerical tolerances. Skip speculative abstractions and cosmetic churn.
- Check coverage that fits the change. Benchmark performance changes before and after. Documentation-only edits need
  link and claim checks, not new tests that mirror the implementation.
- `TensorPool` has no established production adoption. Do not prescribe it just because an older agent file did.
- Put each fact in its owning document, following the [documentation map](../README.md): open work in ROADMAP, model gaps
  in the modality status, reference evidence in PARITY_VERIFICATION, recurring traps in TROUBLESHOOTING. Delete completed
  task journals once their unique evidence and decisions are preserved elsewhere.
- Claim verification only to the level you actually demonstrated. Synthetic tests, component parity, real generation,
  consumer verification and full numerical parity are five different claims.
- Keep examples short, current and compilable. Agent files add task-specific constraints and do not repeat the core
  and style rules.

## Packaging

The version comes from `Directory.Build.props`. Libraries target net8.0 and net10.0. The API is not packable. Before you
change release steps, read the actual package metadata and the publishing workflow.

Validate Release builds and a fresh consumer. Publish the engine version and confirm it is available before you repin the
SwarmUI extension. Use `UseLocalHartsy=true` for local extension development. Do not claim a version is publicly
available from the repository version alone.
