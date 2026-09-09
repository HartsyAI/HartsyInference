# Audit

Review the affected path against [shared architecture](AGENTS.md), [style](../CODE_STYLE.md), and relevant [known failures](../Checklists/TROUBLESHOOTING.md).

- Trace inputs through Engine, ownership, backend execution, result delivery, cancellation and disposal. Check package boundaries, shape/dtype validation, path handling and resource limits.
- Borrowed TensorView/TensorRef values must not outlive storage. Check preload/CPU disposal, GPU callbacks, stream ordering, partial construction and exception cleanup.
- Check hot-path allocations, unnecessary device transfers, SIMD tails and native status handling. Prove performance findings with measurements.
- Reproduce numerical failures using saved reference inputs; isolate the first divergent layer. INF can indicate F16 overflow; finite garbage can indicate layout errors. Neither symptom proves a unique cause.
- Report actionable findings with source location, trigger, impact, evidence and remaining uncertainty. Distinguish confirmed bugs from hypotheses and missing verification.
- Fix root causes minimally; preserve tolerances. Add meaningful regression coverage for behavior changes; do not refactor unrelated code during diagnosis.
