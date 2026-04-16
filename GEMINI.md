# SharpInference — Gemini Agent Instructions

## Project Overview

SharpInference is a pure C#/.NET 10 AI inference engine for non-LLM modalities — image generation (diffusion), speech-to-text, text-to-speech, voice conversion, vision, object detection, and video generation. It pairs with dotLLM for LLM inference.

**Before doing anything, read these files to understand the project:**

1. `docs/Design/CORE_DESIGN.md` — architecture overview, design pillars, key decisions
2. `docs/Design/BUILD_ORDER.md` — phase dependencies and sequencing
3. `docs/Design/FILE_STRUCTURE.md` — where everything lives
4. `docs/Design/NUGET_PACKAGE_DESIGN.md` — package boundaries and dependencies

## How to Work on This Project

Identify the task you've been given, then read the matching agent instruction file from `docs/Agents/`:

| Task | Agent File |
|---|---|
| Research a topic | `docs/Agents/RESEARCH.md` |
| Plan an implementation | `docs/Agents/ARCHITECT.md` |
| Write code | `docs/Agents/BUILDER.md` |
| Write SIMD/PTX kernels | `docs/Agents/KERNEL.md` |
| Write or run tests | `docs/Agents/TESTER.md` |
| Review code | `docs/Agents/REVIEWER.md` |
| Update documentation | `docs/Agents/DOCS.md` |
| Track progress | `docs/Agents/CHECKLIST.md` |
| Benchmark performance | `docs/Agents/BENCHMARK.md` |
| Convert model formats | `docs/Agents/CONVERT.md` |
| Package for NuGet | `docs/Agents/DEPLOY.md` |
| Debug failures | `docs/Agents/DEBUG.md` |
| Refactor/optimize | `docs/Agents/REFACTOR.md` |
| Build API endpoints | `docs/Agents/API.md` |
| Cross-package wiring | `docs/Agents/INTEGRATION.md` |

Follow the agent instructions. They contain what to read, the workflow, and quality standards.

## Key Rules

- **Pure C#** — no native shared libraries, no Python, no C++ wrappers
- **CUDA via PTX** — embedded as resources, JIT-compiled via CUDA Driver API P/Invoke
- **Eager execution** — no computation graphs
- **Unmanaged memory** — `NativeMemory.AlignedAlloc` or mmap for tensors, never managed arrays on hot paths
- **Validate against references** — every component must match Python/C++ reference within tolerances
- **Respect package boundaries** — see `docs/Design/NUGET_PACKAGE_DESIGN.md`

## Current Phase

Check `docs/Checklists/` to determine which phase is active. Start with the earliest phase that has unchecked items.
