# Documentation map

Keep information that code cannot recover: design constraints, upstream provenance, format traps,
reference disagreements, verification evidence, and unresolved work. Do not duplicate implementations,
file trees, completed-task journals, or benchmark tables. Git preserves removed history.

| Need | Read |
|---|---|
| Coding rules | [CODE_STYLE.md](CODE_STYLE.md) |
| Architecture and task-specific instructions | [Agents/AGENTS.md](Agents/AGENTS.md) |
| Current model support and model-specific gaps | [Checklists/MODEL_STATUS.md](Checklists/MODEL_STATUS.md) |
| Cross-cutting open work | [Checklists/ROADMAP.md](Checklists/ROADMAP.md) |
| Real-weight numerical evidence | [Checklists/PARITY_VERIFICATION.md](Checklists/PARITY_VERIFICATION.md) |
| Debugging traps | [Checklists/TROUBLESHOOTING.md](Checklists/TROUBLESHOOTING.md) |
| Upstream constants, formats, methods, and unresolved research | [Research/](Research/) |
| Multi-GPU configuration | [MULTI_GPU.md](MULTI_GPU.md) |
| Environment controls and unsafe switches | [ENV_VARS.md](ENV_VARS.md) |
| Measured performance | [Scoreboards](../benchmarks/scoreboards/) |

## Maintenance

- Read only the relevant document/section; search before loading long references.
- One fact, one home. Link to measurements instead of copying them into READMEs or plans.
- Status rows contain a verdict and a details link. Evidence records checkpoint/variant, backend,
  reference, metric/tolerance, date, and limitations. A coherent output is not full numerical parity.
- Delete completed checklist items after preserving unique evidence in the appropriate status or parity doc.
- Research dates describe the source snapshot, not current implementation status. Check code and status
  before treating an old build plan as open work. A provenance note does not imply the model is verified.
- Keep exact constants, tensor/key layouts, source links, disagreements, and diagnostic lessons.
  Remove implementation walkthroughs after the code ships. Keep unresolved questions explicit.
- New research follows [RESEARCH.md](Agents/RESEARCH.md). Avoid estimated speedups presented as measurements.
- Generated reports, local run artifacts, tokenizer data, and fixtures are not agent onboarding material.
