# Cleanup and documentation

- Refactor for measured cost, real duplication or a boundary/ownership defect. Preserve public compatibility and numerical tolerances; avoid speculative abstractions and cosmetic churn.
- Check coverage appropriate to the change. Benchmark performance changes before/after. Documentation-only edits need link/claim checks, not new implementation-mirroring tests.
- TensorPool has no established production adoption; do not prescribe it merely because an older agent file did.
- Follow [documentation ownership](../README.md): cross-cutting open work in ROADMAP; model gaps in the modality status; reference evidence in PARITY_VERIFICATION; recurring traps in TROUBLESHOOTING. Delete completed task journals after preserving unique evidence/decisions.
- Update verification claims only to the level actually demonstrated: synthetic, component parity, real generation, consumer verification and full numerical parity differ.
- Keep examples short, current and compilable. Agent files add task-specific constraints rather than repeating core/style rules.

## Packaging

Version comes from Directory.Build.props. Libraries target net8.0/net10.0; API is not packable. Check actual package metadata and the publishing workflow before changing release steps.
Validate Release builds and a fresh consumer. Publish the engine version and verify availability before repinning the SwarmUI extension; local extension development uses UseLocalHartsy=true. Do not claim a version is publicly available from the repository version alone.
