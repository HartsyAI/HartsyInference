# Vision & Goals

## Why HartsyInference exists

Every serious open AI inference stack today is Python plus native C++/CUDA libraries. That is fine for
research, but it is a heavy, fragile dependency to embed inside a .NET application: you ship a Python
runtime, a virtual environment, a native inference library per platform, and a subprocess boundary to
marshal across. HartsyInference removes all of that.

HartsyInference is a **pure C#/.NET inference engine**. It loads `.safetensors`, `.gguf`, and PyTorch
`.pt`/`.ckpt` checkpoints directly and runs them on CUDA, Vulkan, or CPU with no Python, no C++
wrappers, and no bundled native inference library. GPU kernels are PTX (CUDA Driver API) and SPIR-V
(Vulkan), JIT-compiled at runtime. The whole thing is NuGet packages.

The engine covers the modalities Python stacks spread across a dozen projects: LLM text generation,
diffusion image generation, speech-to-text, text-to-speech, music, vision (embeddings, detection,
segmentation), video generation, 3D mesh generation, and real-time interactive world models.

## The SwarmUI angle (how people actually use it)

HartsyInference is an **engine, not an application**. We are deliberately **not building a first-party
front-end**. Instead, the recommended way to run it is inside
[SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) through the
[SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)
extension.

The extension registers HartsyInference as a SwarmUI backend, a pure-C# alternative to SwarmUI's
ComfyUI backend. That gives users:

- SwarmUI's mature generation UI, model browser, queue, and parameter controls, with **no Python
  environment to install**.
- Per-architecture model loaders, video output (Wan, LTX) with ffmpeg muxing, audio/music
  (ACE-Step), LoRA passthrough, live previews, and automatic checkpoint conversion.
- The exact engine and kernels this repository builds, consumed as pinned `HartsyInference` NuGet
  packages.

This split keeps our surface small and focused: SwarmUI owns the product experience, HartsyInference
owns correct, fast, dependency-free inference. Developers who want to embed the engine directly still
can, through the per-modality NuGet libraries.

## Goals

1. **Pure managed .NET.** No Python, no native shared inference libraries, no external processes. GPU
   access is PTX/SPIR-V via P/Invoke only.
2. **Broad, correct model coverage.** Match a Python/C++ reference within documented tolerances for
   every model. Correctness is verified against real weights, not just "finite floats", and tracked in
   [`../Checklists/PARITY_VERIFICATION.md`](../Checklists/PARITY_VERIFICATION.md).
3. **The best pure-C# performance we can reach.** We are transparent that we are not yet as fast as the
   fastest native runners (see [the benchmarks](../../benchmarks/README.md)); closing that gap
   (flash-attention, CUDA graphs, F16 activation paths) is an ongoing, in-the-open effort.
4. **First-class SwarmUI integration.** The SwarmUI extension is the primary way users touch the
   engine; new model support is not "done" until it runs end-to-end through SwarmUI.
5. **Modular packaging.** Pull in only the modality and backend you need.
6. **Zero-GC hot paths.** Tensor storage in unmanaged aligned memory, memory-mapped weights,
   `Span<T>` throughout, no allocations on inference hot paths.

## Non-goals

- **A first-party UI / web app / desktop app.** SwarmUI is the front-end; we build the backend for it.
- **An OpenAI-compatible REST server as a product.** This was previously scoped (old "Phase 7") and has
  been **dropped**. The engine is consumed via the SwarmUI extension, the NuGet libraries, and the
  bundled sample CLIs. The `HartsyInference.Server` project remains in the tree only as abandoned
  scaffolding and is not advertised or supported.
- **A dependency on dotLLM.** Early designs treated LLM inference as an external concern handled by
  dotLLM. That is no longer the case: LLM text generation is **native** in the `HartsyInference.LLM`
  package (a config-driven generic decoder transformer). The
  [`../Research/DOTLLM_ARCHITECTURE.md`](../Research/DOTLLM_ARCHITECTURE.md) note is retained only as a
  historical study that informed the native design.
- **Training.** HartsyInference is an inference engine. Fine-tuning and training are out of scope.

## Audience

- **SwarmUI users** who want a pure-C#, no-Python backend for image/video/audio/LLM generation.
- **.NET developers** embedding AI inference into their own applications without a Python sidecar.
- **Contributors** porting new model architectures; see [`BUILD_ORDER.md`](BUILD_ORDER.md) and the
  agent instruction files under [`../Agents/`](../Agents/).

## Related documents

- [Core Design](CORE_DESIGN.md) — architecture overview and design pillars.
- [Features](FEATURES.md) — the full capability list across modalities.
- [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) — the model support plan.
- [NuGet Package Design](NUGET_PACKAGE_DESIGN.md) — package boundaries and dependencies.
- [Benchmarks](../../benchmarks/README.md) — how we measure and where we stand.
