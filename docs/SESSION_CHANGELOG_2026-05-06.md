# Session Changelog — 2026-05-06

> **Scope:** Phase 4 polish + closeout work. Everything in this changelog is on the dev tree, uncommitted. Review and split into logical commits before merging.

## Summary

| Area | Files added | Files modified | Tests added |
|---|---|---|---|
| Quality presets / FP8 wiring | 5 | 0 | 11 (Quality + Loaders) |
| Adapter loaders (ControlNet/IpAdapter) | 4 | 0 | 8 |
| Native FP8 GEMM (cuBLASLt) | 2 | 1 (CudaBackend) | 7 |
| SSIM / numerical gates | 3 | 2 (Sdxl + Flux pipelines, request type) | 5 (skip cleanly) |
| GGUF K-quant fix + Q5_K | 0 | 3 (DType, GgufLoader, GgufDequantizer) | 7 |
| Docs (T5 mem, FP8 perf, benchmarking) | 3 | 2 (Phase 4 checklist, CUDA_PERFORMANCE) | — |
| CI workflows | 2 | 0 | — |
| Python reference dumps | 3 | 0 | — |
| **Totals** | **22** | **8** | **38** |

All tests pass on .NET 8 + 10. Full solution builds clean (0 warnings, 0 errors).

## Changes by area

### Quality presets + per-pipeline applier

**New:**
- [`src/HartsyInference.Diffusion/Quality/QualityPreset.cs`](../src/HartsyInference.Diffusion/Quality/QualityPreset.cs) — 5-value enum (Maximum/High/Medium/Low/Custom)
- [`src/HartsyInference.Diffusion/Quality/QualityProfile.cs`](../src/HartsyInference.Diffusion/Quality/QualityProfile.cs) — record with BackboneDType / TextEncoderDType / VaeDType + Validate()
- [`src/HartsyInference.Diffusion/Quality/QualityProfileApplier.cs`](../src/HartsyInference.Diffusion/Quality/QualityProfileApplier.cs) — generic dict-mutating cast helper that skips norms/biases/pos_embeds when target is FP8
- [`src/HartsyInference.Diffusion/Quality/FluxQualityLoader.cs`](../src/HartsyInference.Diffusion/Quality/FluxQualityLoader.cs) — Flux-specific applier over `FluxCheckpointConverter.ConvertedWeights`
- [`src/HartsyInference.Diffusion/Quality/SdxlQualityLoader.cs`](../src/HartsyInference.Diffusion/Quality/SdxlQualityLoader.cs) — same shape for SDXL

**Tests:** `QualityProfileTests` covers preset → dtype mapping, FP8 VAE validation rejection, F32→F16/FP8 cast paths with norm-skip rule, and the Flux/Sdxl loader composition. 11 tests.

### Adapter loaders

**New:**
- [`src/HartsyInference.Diffusion/Adapters/ControlNetFile.cs`](../src/HartsyInference.Diffusion/Adapters/ControlNetFile.cs) + [`ControlNetLoader.cs`](../src/HartsyInference.Diffusion/Adapters/ControlNetLoader.cs)
- [`src/HartsyInference.Diffusion/Adapters/IpAdapterFile.cs`](../src/HartsyInference.Diffusion/Adapters/IpAdapterFile.cs) + [`IpAdapterLoader.cs`](../src/HartsyInference.Diffusion/Adapters/IpAdapterLoader.cs)

Auto-detects base model (Sd15 / Sdxl / Flux from key signatures + tensor shapes) and mode/variant (filename keywords + key inspection). Returns owned `*File` wrapping the SafeTensorsLoader. Downstream `ControlNet.LoadWeights` / `IpAdapter.LoadWeights` are still stubs (deferred to v2 — block-mirroring forward pass).

**Tests:** `AdapterLoaderTests` builds synthetic safetensors files with hand-crafted JSON headers and asserts detection. 8 tests.

### Native FP8 GEMM (cuBLASLt)

**New:**
- [`src/HartsyInference.Cuda/CublasLtApi.cs`](../src/HartsyInference.Cuda/CublasLtApi.cs) — P/Invoke for cuBLASLt (handle, matmul desc, layout, preference, dispatch)
- [`src/HartsyInference.Cuda/Fp8GemmExecutor.cs`](../src/HartsyInference.Cuda/Fp8GemmExecutor.cs) — single-handle wrapper, `IsSupported` gates SM ≥ 8.9

**Modified:**
- [`src/HartsyInference.Cuda/CudaBackend.cs`](../src/HartsyInference.Cuda/CudaBackend.cs) — added `EnableNativeFp8Gemm` opt-in flag, `Fp8Executor` lazy property, and gated dispatch in `Linear` (FP8×FP8 → F16 with native GEMM when enabled + supported)

**Tests:** `Fp8GemmExecutorTests` — 7 gating tests confirm Ampere reports `IsSupported=false` and `Run` throws on unsupported hardware. **Native path is wired but not validated end-to-end on Ada GPU** (dev box is RTX 3060 / SM 8.6).

### SSIM / numerical gates

**New:**
- [`tests/HartsyInference.Diffusion.Tests/SdxlSsimTests.cs`](../tests/HartsyInference.Diffusion.Tests/SdxlSsimTests.cs) — strict gate 0.90 when `init_noise_seed42.bin` present, loose 0.30 fallback
- [`tests/HartsyInference.Diffusion.Tests/FluxSsimTests.cs`](../tests/HartsyInference.Diffusion.Tests/FluxSsimTests.cs) — Dev (10 step) + Schnell (4 step) at strict 0.85 / loose 0.30
- [`tests/HartsyInference.Diffusion.Tests/T5EncoderDiffTests.cs`](../tests/HartsyInference.Diffusion.Tests/T5EncoderDiffTests.cs) — CPU avg_err < 1e-4, GPU avg_err < 1e-3 vs HuggingFace transformers reference

**Modified:**
- [`src/HartsyInference.Diffusion/Requests/TextToImageRequest.cs`](../src/HartsyInference.Diffusion/Requests/TextToImageRequest.cs) — added `Tensor? InitialNoise` (pipeline takes ownership, disposes after use; eliminates RNG mismatch with PyTorch for SSIM tests)
- [`src/HartsyInference.Diffusion/Pipelines/SdxlPipeline.cs`](../src/HartsyInference.Diffusion/Pipelines/SdxlPipeline.cs) — `BuildInitialLatent` now uses `TakeOrCreateNoise` helper that consumes injected noise when present
- [`src/HartsyInference.Diffusion/Pipelines/FluxPipeline.cs`](../src/HartsyInference.Diffusion/Pipelines/FluxPipeline.cs) — same wiring on the unpacked-noise path

All SSIM tests skip cleanly when reference images aren't present (Python scripts haven't been run yet).

### GGUF K-quant fix + Q5_K

**Modified:**
- [`src/HartsyInference.Core/Tensors/DType.cs`](../src/HartsyInference.Core/Tensors/DType.cs) — added `Q5_K` (176 bytes / 256 elements)
- [`src/HartsyInference.ModelHandler/Gguf/GgufLoader.cs`](../src/HartsyInference.ModelHandler/Gguf/GgufLoader.cs) — corrected `MapGgufType`: was mapping `14 → Q4_K` (wrong; 14 is Q6_K). Now `12 → Q4_K, 13 → Q5_K, 30 → BF16` per ggml.h. Throws on unknown IDs.
- [`src/HartsyInference.ModelHandler/Gguf/GgufDequantizer.cs`](../src/HartsyInference.ModelHandler/Gguf/GgufDequantizer.cs) — fixed Q4_K scale extraction (was `dmin * (scales[2*j+8] >> 4)` due to operator precedence bug; now uses canonical ggml `get_scale_min_k4` static helper). Added Q5_K dequantizer using both low-nibble and high-bit reconstruction.

**Tests:** `GgufDequantizerTests` — 7 tests with hand-built canonical block bytes verify Q8_0 round-trip, Q4_K known-block reconstruction, Q5_K low+high bit packing, F16 path, and exception cases.

**Impact:** unblocks the §5b common blocker — AuraFlow Q4_K, ERNIE-Image Q4_K_M / Q5_K_M, Chroma Q4/Q5 are now feasible on 12 GB GPUs (each still needs a per-format GGUF→diffusers key remapper).

### Other config additions

**Modified:**
- [`src/HartsyInference.Diffusion/Models/Denoisers/QwenImageConfig.cs`](../src/HartsyInference.Diffusion/Models/Denoisers/QwenImageConfig.cs) — new `V1` preset matching actual `Qwen/Qwen-Image` (60 layers, hidden=3072, 24 heads). `V1_7B` retained as alias; clarified the suffix was a misnomer.
- [`src/HartsyInference.Diffusion/Models/TextEncoders/LlamaStyleEncoderConfig.cs`](../src/HartsyInference.Diffusion/Models/TextEncoders/LlamaStyleEncoderConfig.cs) — added `Qwen2_5_VL_7B` preset (28 layers, hidden=3584, GQA 28:4) for Qwen-Image's text encoder.

### CI

**New:**
- [`.github/workflows/ci-cpu.yml`](../.github/workflows/ci-cpu.yml) — Ubuntu + Windows × .NET 8 + 10, runs unit tests filtered to skip GPU/integration suites
- [`.github/workflows/ci-gpu.yml`](../.github/workflows/ci-gpu.yml) — self-hosted `cuda` runner, fast lane (smoke + diff/SSIM) on every push, slow lane (full generation) on manual dispatch

### Python reference dumps

**New:**
- [`tests/python-reference/dump_sdxl_reference_image.py`](../tests/python-reference/dump_sdxl_reference_image.py)
- [`tests/python-reference/dump_flux_reference_image.py`](../tests/python-reference/dump_flux_reference_image.py)
- [`tests/python-reference/dump_t5_xxl_hidden_states.py`](../tests/python-reference/dump_t5_xxl_hidden_states.py)

Both image scripts emit PNG (human inspection) + raw `.rgb` (C# consumption without a PNG decoder dep) + `init_noise_seed{N}.bin` for noise-injection tests.

### Documentation

**New:**
- [`docs/Research/T5_MEMORY_STRATEGY.md`](Research/T5_MEMORY_STRATEGY.md) — per-pipeline FP8/Q8 sizing for 12 GB GPUs + eviction discipline
- [`docs/Research/BENCHMARKING.md`](Research/BENCHMARKING.md) — procedure for HartsyInference vs ComfyUI it/s comparison
- This file (`SESSION_CHANGELOG_2026-05-06.md`)

**Modified:**
- [`docs/Research/CUDA_PERFORMANCE.md`](Research/CUDA_PERFORMANCE.md) — added Native FP8 GEMM section
- [`docs/Checklists/PHASE_4_MODEL_BREADTH.md`](Checklists/PHASE_4_MODEL_BREADTH.md) — checked off ~12 items based on what was already built (Chroma was 100% built but checklist showed it as pending), added GGUF K-quant resolution, scoped Qwen-Image with VRAM blocker

## What's still open

- **Qwen-Image full transformer port** — VRAM-blocked for end-to-end on 12 GB; deferred behind GGUF Q4_K loader for Qwen-Image checkpoints. Config + presets ready.
- **End-to-end visual validation against actual checkpoints** for AuraFlow / ERNIE / Chroma — requires download (~10-17 GB each) + first-run debug pass.
- **`EnableNativeFp8Gemm` validation on Ada hardware** — code wired but untested, gated off by default.
- **GPU CI lighting up** — workflow yml ships but needs a self-hosted runner with the `cuda` label registered.
- **Phase 4 §7 closeout** — code review (this doc), benchmark collection (procedure documented; data still needs gathering on hardware), merge to main (user action).

## Suggested commit split

If splitting into focused commits:

1. `feat(gguf): fix Q4_K scale extraction; add Q5_K`
2. `feat(quality): add QualityPreset/QualityProfile + per-pipeline appliers`
3. `feat(adapters): add ControlNetLoader / IpAdapterLoader auto-detection`
4. `feat(cuda): wire native FP8 GEMM via cuBLASLt (opt-in)`
5. `feat(pipelines): plumb initialNoise injection through Sdxl/Flux pipelines`
6. `test: SSIM gates + T5 numerical diff tests + Python ref dumps`
7. `ci: add CPU + GPU workflow`
8. `docs: T5 memory strategy, benchmarking guide, Phase 4 checklist updates`
