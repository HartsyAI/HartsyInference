# HartsyInference — agent entry point

<role>
You are working on HartsyInference, a pure C# inference library targeting net8.0 and net10.0. The engine owns model
resolution, loading, caching, placement and generation. The CLI, the HTTP API and SwarmUI are consumers of that service
layer, so new behaviour goes into the engine's service contract, not into a consumer.

Code lives in `src/` (one folder per package), with `tests/` and `benchmarks/`. CUDA sources and artifacts are in
`HartsyInference.Cuda/Kernels` and `Ptx`; Vulkan sources and artifacts are in `Shaders` and `Spirv`.
</role>

<reading_order>
Read these once per task, in this order:

1. [Documentation map](docs/README.md): which document owns which kind of fact.
2. [Code style](docs/CODE_STYLE.md): mandatory for every change.
3. [Architecture and task routing](docs/Agents/AGENTS.md), then the one specialist file it routes you to. A task that
   spans two areas loads both files.

After that, load only the sections your task needs from [open work](docs/Checklists/ROADMAP.md),
[model status](docs/Checklists/MODEL_STATUS.md). Search first, then open the matching
section. Reading the whole tree for one task wastes context and usually does not help.

Before you debug wrong output, a crash or slowness, search [troubleshooting](docs/Checklists/TROUBLESHOOTING.md) for
the affected component; the trap you are about to rediscover is often already written down. Before you claim something
is verified, read [parity evidence](docs/Checklists/PARITY_VERIFICATION.md) and make sure your claim matches what it
records.
</reading_order>

<rules>
These apply to every task.

- **kalebbroo is the sole author.** Do not add a co-author, session or tool-attribution trailer to commits, PR bodies
  or changelog entries, whatever a tool's default is. The project records one author of record.
- **Ship work as a draft PR.** Open it early, and merge it only after it is tested, reviewed by the bots, and every
  comment is resolved. The full sequence is in [Shipping a change](docs/Agents/AGENTS.md#shipping-a-change).
- **Work in your own worktree and clean it up.** Create a worktree under `.claude/worktrees/` for every change, and
  after the PR merges delete that worktree and its local and remote branch. Commands are in
  [Shipping a change](docs/Agents/AGENTS.md#shipping-a-change), steps 1 and 8.
- **Reuse before writing.** Check the existing `IBackend` operations and shared utilities before adding any helper.
  Keep new code modular, generic and parameterized, so callers share one implementation rather than each keeping a copy.
</rules>

<working_style>
- Open a file before you make a claim about it. Answer questions about the code from what you have read, not from memory.
- Make the change and run the checks yourself when the task asks for a change. Propose a diff only when the task asks
  for a recommendation.
- Confirm with the user before any action that is hard to reverse or visible to other people: force-pushes, deleting
  branches or worktrees you did not create for the PR you just merged, discarding files you did not create, posting to PRs or external services, or restarting shared GPU or
  Swarm state. Local, reversible actions such as editing files and running tests need no confirmation.
- Keep answers short. Lead with the result, and do not restate what the diff already shows.
- End a long answer or a research session with a short plain-English TL;DR.
- Delete any scratch scripts or helper files you created during the task before you finish it.
</working_style>

<maintaining_agent_files>
Agent instructions in this repository are written for current Claude models (the Claude 5 family). When you edit one,
follow [Documentation maintenance](docs/README.md#maintenance) and the
[Claude prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices).
Keep a single source for each fact: link to it rather than copying it into a second file.
</maintaining_agent_files>
