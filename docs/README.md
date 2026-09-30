# Documentation map

Keep what code cannot recover: design constraints, upstream provenance, format traps, reference disagreements,
verification evidence and unresolved work. Not implementations, file trees, task journals or benchmark tables —
git preserves removed history.

| Need | Read |
|---|---|
| Coding rules | [CODE_STYLE.md](CODE_STYLE.md) |
| Architecture, task routing, PR workflow | [Agents/AGENTS.md](Agents/AGENTS.md) |
| Tensor, CUDA, planning and pipeline patterns | [Agents/ENGINE_PATTERNS.md](Agents/ENGINE_PATTERNS.md) |
| Current model support and model-specific gaps | [Checklists/MODEL_STATUS.md](Checklists/MODEL_STATUS.md) |
| Which models have been run on Vulkan | [Checklists/VULKAN_STATUS.md](Checklists/VULKAN_STATUS.md) |
| Cross-cutting open work | [Checklists/ROADMAP.md](Checklists/ROADMAP.md) |
| Real-weight numerical evidence | [Checklists/PARITY_VERIFICATION.md](Checklists/PARITY_VERIFICATION.md) |
| Debugging traps | [Checklists/TROUBLESHOOTING.md](Checklists/TROUBLESHOOTING.md) |
| Upstream constants, formats, methods, unresolved research | [Research/](Research/) |
| Multi-GPU configuration | [MULTI_GPU.md](MULTI_GPU.md) |
| Settings: where they live, how to change them | [SETTINGS.md](SETTINGS.md) |
| Reproducible community performance | [Benchmark guide](../benchmarks/README.md) |

## Maintenance

- Read only the relevant document or section; search before loading a long reference. One fact, one home — link
  to measurements instead of copying them into a README or plan.
- Status rows carry a verdict and a details link. Evidence records checkpoint/variant, backend, reference,
  metric/tolerance, date and limitations. A coherent output is not full numerical parity.
- Quote a measurement as an absolute number against an external baseline ("1.671 s/step against ComfyUI's
  1.660 s/step"), never against our own past, and never present an estimate as a measurement.
- A fixed bug earns doc space only for the diagnostic a future reader would reuse — not the incident, the dates
  or the commit that caused it. Default to writing nothing; the changelog already has the history.
- Delete completed checklist items once their unique evidence lives in the matching status or parity doc. Remove
  implementation walkthroughs after the code ships, keep exact constants, layouts, source links and
  disagreements, and keep unresolved questions explicit.
- Research dates describe the source snapshot, not implementation status: check code and status before treating
  an old build plan as open work. New research follows [RESEARCH.md](Agents/RESEARCH.md).
- Generated reports, local run artifacts, tokenizer data and fixtures are not agent onboarding material.
