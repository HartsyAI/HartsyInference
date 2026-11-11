# Documentation map

Keep what code cannot recover: design constraints, upstream provenance, format traps, reference disagreements,
verification evidence and unresolved work. Not implementations, file trees, task journals or benchmark tables —
git preserves removed history.

| Need | Read |
|---|---|
| Coding rules | [CODE_STYLE.md](CODE_STYLE.md) |
| Attention semantics, numerical permissions and kernel planning | [ATTENTION_CONTRACT.md](ATTENTION_CONTRACT.md) |
| Generic layout ops: strided views, strided copy and accumulate, pad | [BACKEND_OPS.md](BACKEND_OPS.md) |
| Current model support and model-specific gaps | [Checklists/MODEL_STATUS.md](Checklists/MODEL_STATUS.md) |
| Which models have been run on Vulkan | [Checklists/VULKAN_STATUS.md](Checklists/VULKAN_STATUS.md) |
| Vulkan LLM and MoE: measured gap to CUDA and the port plan | [VULKAN_MOE_PLAN.md](VULKAN_MOE_PLAN.md) |
| Cross-cutting open work | [Checklists/ROADMAP.md](Checklists/ROADMAP.md) |
| Comfy-Org repo rollout: pinned inventory, phases, target and current state per repo | [Checklists/COMFY_ORG_ROLLOUT.md](Checklists/COMFY_ORG_ROLLOUT.md) |
| Real-weight numerical evidence | [Checklists/PARITY_VERIFICATION.md](Checklists/PARITY_VERIFICATION.md) |
| DeepSeek-V4.1-Flash rented-GPU runbook and frozen gates | [Checklists/DSV41_RENTAL_RUNBOOK.md](Checklists/DSV41_RENTAL_RUNBOOK.md) |
| CoLa3D geometry decoder and prompt inputs: key groups, shape VAE relation, unverified prompt conventions | [Research/COLA3D_GEOMETRY_DECODER.md](Research/COLA3D_GEOMETRY_DECODER.md) |
| Hunyuan3D-2.1 shape model: source spike, MoE and reuse map, licence | [Research/HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md](Research/HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md) |
| Hunyuan3D shape models: pins, licence terms, upstream-to-C# stage map, findings | [Research/HUNYUAN3D_ARCHITECTURE.md](Research/HUNYUAN3D_ARCHITECTURE.md) |
| Mesh post-processing and PBR baking toolkit: algorithms, conventions, limits against cumesh and xatlas | [Research/MESH_POSTPROCESS.md](Research/MESH_POSTPROCESS.md) |
| MiniMax-H3-Image architecture: header facts, ComfyUI reference constants, plan corrections, open items | [Research/MINIMAX_H3_IMAGE_ARCHITECTURE.md](Research/MINIMAX_H3_IMAGE_ARCHITECTURE.md) |
| Debugging traps | [Checklists/TROUBLESHOOTING.md](Checklists/TROUBLESHOOTING.md) |
| Multi-GPU configuration | [MULTI_GPU.md](MULTI_GPU.md) |
| Embedding the engine in a host application: the public entry points | [HOST_INTEGRATION.md](HOST_INTEGRATION.md) |
| Proving a numerical change on a GPU: op fixtures and generation captures | [VERIFICATION.md](VERIFICATION.md) |
| Settings: where they live, how to change them | [SETTINGS.md](SETTINGS.md) |
| Reproducible community performance | [Benchmark guide](../benchmarks/README.md) |

## Maintenance

- Read only the relevant document or section; search before loading a long reference. One fact, one home — link
  to measurements instead of copying them into a README or plan.
- Agent files (`AGENTS.md`, `docs/Agents/`, skills under `docs/Agents/skills/`) target current Claude models. Write them
  to the [Claude prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices):
  state the goal and the reason for each rule, phrase rules positively, separate sections with XML tags, and drop
  CRITICAL or MUST emphasis, which current models tend to over-apply.
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
  an old build plan as open work.
- Generated reports, local run artifacts, tokenizer data and fixtures are not agent onboarding material.
