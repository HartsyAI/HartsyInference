# Audit

Use this file to review existing code. Check the affected path against the [shared architecture](AGENTS.md),
[engine patterns](ENGINE_PATTERNS.md), [code style](../CODE_STYLE.md) and any matching
[known failures](../Checklists/TROUBLESHOOTING.md).

- Trace inputs through the Engine, ownership, backend execution, result delivery, cancellation and disposal. Check
  package boundaries, shape and dtype validation, path handling and resource limits.
- Borrowed `TensorView` and `TensorRef` values must not outlive their storage. Check preload and CPU disposal, GPU
  callbacks, stream ordering, partial construction and exception cleanup.
- Check hot-path allocations, unnecessary device transfers, SIMD tails and native status handling. Back every
  performance finding with a measurement.
- Reproduce numerical failures with saved reference inputs, then isolate the first divergent layer. INF can indicate F16
  overflow, and finite garbage can indicate a layout error. Neither symptom proves a unique cause.
- Report each actionable finding with its source location, trigger, impact, evidence and remaining uncertainty. Mark
  confirmed bugs separately from hypotheses and from gaps in verification.
- Fix root causes with minimal changes and keep the existing tolerances. Add regression coverage for every behaviour
  change. Leave unrelated code alone during a diagnosis, because a refactor there hides the cause you are chasing.
