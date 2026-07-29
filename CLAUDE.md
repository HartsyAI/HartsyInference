# HartsyInference — Claude Code Instructions

## Project Overview

HartsyInference is a pure C#/.NET AI inference engine (targets net8.0 and net10.0) covering LLM text generation, image generation (diffusion), speech-to-text, text-to-speech, voice conversion, music, vision, object detection, video generation, 3D mesh, and interactive world models. LLM inference is native in the `HartsyInference.LLM` package (dotLLM is no longer a dependency). The recommended way to run the engine is the SwarmUI backend extension (https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend); it is also consumed as NuGet libraries and via the sample CLIs.

**`HartsyInference.Engine` is the service layer / single source of truth** for "load a model + generate": it owns model lifecycle (registry, HF download, cache), the `InferenceEngine` facade + per-modality dispatch/handlers, the backend factory, and native request/result DTOs. The CLI, the HTTP server, and the SwarmUI extension are thin wrappers over it — do not re-implement load/generate orchestration in a consumer.

## Before Any Task

1. Read `docs/CODE_STYLE.md` — **MANDATORY** code style (read first, follow always)
2. Read `docs/Agents/AGENTS.md` — shared design rules, core engine patterns, and task routing table
3. Read the specialized agent file matching your task (routing table is in `AGENTS.md`)
4. Check `docs/Checklists/` — cross-cutting open work is in `ROADMAP.md`; per-model open work is in the
   `Remaining work` section of the matching `MODEL_STATUS_*` doc; `TROUBLESHOOTING.md` is the model
   bring-up debugging reference (read it first when a model is wrong, crashes, or is slow)

To see which models are built vs **verified end-to-end**, read the per-modality status docs indexed in
`docs/Checklists/MODEL_STATUS.md` (Image / Audio / Video / World / 3D / Vision / LLM). The cross-modality
real-weight parity authority is `docs/Checklists/PARITY_VERIFICATION.md`.

For deeper architecture context, read `docs/Agents/AGENTS.md` (Shared Design Rules + Core Engine Patterns —
the architecture single source of truth) and the `src/` tree, which has one folder per NuGet package.

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
│   ├── Checklists/                    Model status, roadmap & troubleshooting
│   └── Agents/                        AGENTS.md (core) + specialized agent files
├── src/                               Source code (one folder per NuGet package)
├── tests/                             Test projects
├── samples/                           Example applications
├── benchmarks/                        Performance benchmarks
└── native/cuda/                       CUDA C++ source for PTX generation
```
