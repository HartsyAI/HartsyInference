# SharpInference — Claude Code Instructions

## Project Overview

SharpInference is a pure C#/.NET 10 AI inference engine for non-LLM modalities — image generation (diffusion), speech-to-text, text-to-speech, voice conversion, vision, object detection, and video generation. It pairs with dotLLM for LLM inference.

**Before doing anything, read these files to understand the project:**

1. `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (read this first, follow it always)
2. `docs/Design/CORE_DESIGN.md` — architecture overview, design pillars, key decisions
3. `docs/Design/BUILD_ORDER.md` — phase dependencies and sequencing
4. `docs/Design/FILE_STRUCTURE.md` — where everything lives
5. `docs/Design/NUGET_PACKAGE_DESIGN.md` — package boundaries and dependencies

## How to Work on This Project

### Step 1: Identify What You're Doing

| If asked to... | Use this agent |
|---|---|
| Research a topic or fill out a research doc | `docs/Agents/RESEARCH.md` |
| Plan an implementation approach | `docs/Agents/ARCHITECT.md` |
| Write implementation code | `docs/Agents/BUILDER.md` |
| Write SIMD kernels or PTX GPU code | `docs/Agents/KERNEL.md` |
| Write or run tests | `docs/Agents/TESTER.md` |
| Review code for issues | `docs/Agents/REVIEWER.md` |
| Update docs or README | `docs/Agents/DOCS.md` |
| Update checklists and track progress | `docs/Agents/CHECKLIST.md` |
| Run benchmarks or compare performance | `docs/Agents/BENCHMARK.md` |
| Convert model formats or quantize | `docs/Agents/CONVERT.md` |
| Package and publish to NuGet | `docs/Agents/DEPLOY.md` |
| Debug a failing test or runtime issue | `docs/Agents/DEBUG.md` |
| Refactor or optimize existing code | `docs/Agents/REFACTOR.md` |
| Build server/API endpoints | `docs/Agents/API.md` |
| Wire up cross-package integration | `docs/Agents/INTEGRATION.md` |

### Step 2: Read the Agent Instructions

Open the agent file listed above. It contains:
- What to read before starting
- The workflow to follow
- Quality standards and output expectations
- Related docs to reference

### Step 3: Check the Relevant Checklist

Before starting work, check `docs/Checklists/` for the current phase checklist. Mark items as you complete them.

## Key Rules

- **Pure C# only** — no native shared libraries, no Python, no C++ wrappers
- **CUDA via PTX** — all GPU code is PTX embedded as resources, JIT-compiled via CUDA Driver API P/Invoke
- **Eager execution** — no computation graphs, ops execute immediately
- **Unmanaged memory** — tensor storage via `NativeMemory.AlignedAlloc` or mmap, never managed arrays on hot paths
- **Zero GC pressure** — no allocations on inference hot paths
- **Validate against references** — every component must match a Python/C++ reference implementation within documented tolerances
- **Package boundaries matter** — respect the NuGet package design, don't leak dependencies across packages

## Project Structure

```
SharpInference/
├── CLAUDE.md                          ← You are here
├── docs/
│   ├── Design/                        Architecture and design documents
│   ├── Research/                      Research notes (read before implementing)
│   ├── Checklists/                    Phase progress tracking
│   └── Agents/                        Agent instruction files
├── src/                               Source code (one folder per NuGet package)
├── tests/                             Test projects
├── samples/                           Example applications
├── benchmarks/                        Performance benchmarks
└── native/cuda/                       CUDA C++ source for PTX generation
```

## Current Phase

Check `docs/Checklists/` to determine which phase is currently active. Start with the earliest phase that has unchecked items.
