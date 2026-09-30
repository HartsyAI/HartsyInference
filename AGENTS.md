# HartsyInference — agent entry point

Pure C# inference libraries targeting net8.0/net10.0. Engine owns model resolution, loading, caching, placement
and generation; CLI, HTTP API and SwarmUI are consumers of that service layer. Code lives in `src/` (one folder
per package), `tests/` and `benchmarks/`; CUDA sources and artifacts in `HartsyInference.Cuda/Kernels` and `Ptx`,
Vulkan in `Shaders` and `Spirv`.

Read once per task: [documentation rules](docs/README.md), [code style](docs/CODE_STYLE.md) — mandatory — then
[architecture and task routing](docs/Agents/AGENTS.md) and the one specialist file it names. Load only the
task-relevant sections of [open work](docs/Checklists/ROADMAP.md), [model status](docs/Checklists/MODEL_STATUS.md)
and [research](docs/Research/). Search [troubleshooting](docs/Checklists/TROUBLESHOOTING.md) for the affected
component before debugging wrong output, crashes or slowness, and read
[parity evidence](docs/Checklists/PARITY_VERIFICATION.md) before claiming verification. Never read the whole tree
for one task.

Three rules that apply before anything else:

- **kalebbroo is the sole author.** No co-author, session or tool-attribution trailer in any commit, PR body or
  changelog entry, whatever a tool's own default says.
- **Work ships as a draft PR** — tested, bot-reviewed, every comment resolved, then merged. Full sequence in
  [Shipping a change](docs/Agents/AGENTS.md#shipping-a-change).
- **Reuse before writing.** Check the existing backend ops and shared utilities first; new code stays modular,
  generic and parameterized rather than duplicated per caller.

Keep answers terse — no restating what the diff shows — and end a long answer or research session with a short
plain-English TL;DR.
