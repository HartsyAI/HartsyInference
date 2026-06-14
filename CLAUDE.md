# HartsyInference — Claude Code Instructions

## Project Overview

HartsyInference is a pure C#/.NET 10 AI inference engine for non-LLM modalities — image generation (diffusion), speech-to-text, text-to-speech, voice conversion, vision, object detection, and video generation. It pairs with dotLLM for LLM inference.

## Before Any Task

1. Read `docs/CODE_STYLE.md` — **MANDATORY** code style (read first, follow always)
2. Read `docs/Agents/AGENTS.md` — shared design rules, dotLLM patterns, and task routing table
3. Read the specialized agent file matching your task (routing table is in `AGENTS.md`)
4. Check `docs/Checklists/` — find the active phase (earliest with unchecked items)

For deeper context on specific areas, read:
- `docs/Design/CORE_DESIGN.md` — architecture overview, design pillars
- `docs/Design/BUILD_ORDER.md` — phase dependencies and sequencing
- `docs/Design/FILE_STRUCTURE.md` — where everything lives
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — package boundaries and dependencies

## Key Rules

- **Pure C# only** — no native shared libraries, no Python, no C++ wrappers
- **CUDA via PTX** — all GPU code is PTX loaded from disk, JIT-compiled via CUDA Driver API P/Invoke
- **Eager execution** — no computation graphs, ops execute immediately
- **Unmanaged memory** — tensor storage via `NativeMemory.AlignedAlloc` or mmap, never managed arrays on hot paths
- **Zero GC pressure** — no allocations on inference hot paths
- **Validate against references** — every component must match a Python/C++ reference within documented tolerances
- **Package boundaries matter** — respect the NuGet package design, don't leak dependencies across packages

## Project Structure

```
HartsyInference/
├── CLAUDE.md                          ← You are here
├── docs/
│   ├── CODE_STYLE.md                  Mandatory code style
│   ├── Design/                        Architecture and design documents
│   ├── Research/                      Research notes (read before implementing)
│   ├── Checklists/                    Phase progress tracking
│   └── Agents/                        AGENTS.md (core) + specialized agent files
├── src/                               Source code (one folder per NuGet package)
├── tests/                             Test projects
├── samples/                           Example applications
├── benchmarks/                        Performance benchmarks
└── native/cuda/                       CUDA C++ source for PTX generation
```
