# Phase 1 — Foundation (Core + ModelHandler + Cpu)

> **Goal:** Tensor types work, models can be loaded from disk, basic CPU math is operational.
> **Packages:** SharpInference.Core, SharpInference.ModelHandler, SharpInference.Cpu

---

## 1. Research

- [x] Complete [SAFETENSORS_FORMAT.md](../Research/SAFETENSORS_FORMAT.md) research — done and verified
- [x] Complete [GGUF_FORMAT.md](../Research/GGUF_FORMAT.md) research — done and verified
- [x] Complete [IM2COL_CPU.md](../Research/IM2COL_CPU.md) research — done and verified
- [x] Complete [GROUPNORM_MATH.md](../Research/GROUPNORM_MATH.md) research — done and verified
- [x] Complete [SIMD_INTRINSICS_DOTNET.md](../Research/SIMD_INTRINSICS_DOTNET.md) research — done and verified
- [x] Complete [FLASH_ATTENTION.md](../Research/FLASH_ATTENTION.md) research — done and verified
- [x] Review dotLLM source for tensor and SIMD patterns to follow — done in DOTLLM_ARCHITECTURE.md

## 2. Planning

- [x] Finalize `Tensor` class API surface (constructors, indexers, slicing, disposal) — done in CORE_DESIGN.md and IMPLEMENTATION_DETAILS.md
- [x] Finalize `IBackend` interface method signatures — done in CORE_DESIGN.md and IMPLEMENTATION_DETAILS.md
- [x] Define `DType` enum values and byte-size mappings — done in CORE_DESIGN.md and IMPLEMENTATION_DETAILS.md
- [x] Plan SIMD dispatch strategy (AVX2 baseline, AVX-512 optional, NEON for ARM) — done in CORE_DESIGN.md and IMPLEMENTATION_DETAILS.md
- [x] Plan memory-mapped file lifecycle management — done in IMPLEMENTATION_DETAILS.md
- [x] Plan thread pool design for CPU parallelism — done in IMPLEMENTATION_DETAILS.md
- [x] Write CLAUDE.md agent instructions for Phase 1 — done, CLAUDE.md and 15 agent files exist

## 3. Project Setup

- [x] Create `SharpInference.slnx` solution file — done and verified
- [x] Create `Directory.Build.props` (net10.0, nullable, implicit usings) — done and verified
- [x] Create `Directory.Packages.props` (central package management) — done and verified
- [x] Create `SharpInference.Core.csproj` — done and verified
- [x] Create `SharpInference.ModelHandler.csproj` — done and verified
- [x] Create `SharpInference.Cpu.csproj` — done and verified
- [x] Create `SharpInference.Core.Tests.csproj` — done and verified
- [x] Create `SharpInference.ModelHandler.Tests.csproj` — done and verified
- [x] Create `SharpInference.Cpu.Tests.csproj` — done and verified
- [x] Set up CI pipeline (build + test on push) — done, GitHub Actions workflow at .github/workflows/ci.yml
- [x] Add `.gitignore` for .NET — done and verified
- [x] Add `LICENSE` file — done and verified

## 4. Implementation — SharpInference.Core

- [x] `DType.cs` — enum with F32, F16, BF16, Q8_0, Q4_K, I8, U8 — done and verified (81 lines)
- [x] `TensorShape.cs` — shape + stride metadata, up to 6D — done and verified (194 lines)
- [x] `NativeBuffer.cs` — wrapper over `NativeMemory.AlignedAlloc` with IDisposable — done and verified
- [x] `MmapHandle.cs` — `MemoryMappedFile` lifetime manager — done and verified
- [x] `Tensor.cs` — core tensor type with unmanaged storage, shape, dtype, device — done and verified (207 lines)
- [x] `TensorView.cs` — non-owning ref struct view with offset/shape — done and verified
- [x] `TensorPool.cs` — thread-safe buffer pool for temp allocations — done and verified
- [x] `DeviceKind.cs` — Cpu, Cuda enum — done and verified
- [x] `IBackend.cs` — full interface (matmul, conv2d, groupnorm, layernorm, sdpa, activations, elementwise) — done and verified (103 lines)
- [x] `BackendCapabilities.cs` — what ops a backend supports — done and verified
- [x] `IDiffusionPipeline.cs`, `IAudioPipeline.cs`, `IVisionPipeline.cs` — done and verified
- [x] `IPipelineRequest.cs` — base request record — done and verified
- [x] `IScheduler.cs` — SetTimesteps, Step, AddNoise — done and verified
- [x] `IModel.cs`, `ModelConfig.cs`, `ModelFormat.cs` — done and verified
- [x] `Logs.cs` — static logging class — done and verified
- [x] `SharpInferenceException.cs`, `OutOfVramException.cs`, `UnsupportedModelException.cs` — done and verified

## 5. Implementation — SharpInference.ModelHandler

- [x] `SafeTensorsLoader.cs` — mmap + JSON header parse → dictionary of TensorView — done and verified (169 lines)
- [x] `SafeTensorsWriter.cs` — save tensors to safetensors format — done and verified
- [x] `SafeTensorsShardLoader.cs` — multi-shard loading with unified index — done and verified
- [x] `GgufLoader.cs` — GGUF v3 header + metadata + tensor mmap — done and verified (300 lines)
- [x] `GgufDequantizer.cs` — Q4_0, Q8_0, Q4_K_M dequantization — done and verified
- [x] `GgufMetadata.cs` — typed metadata access — done and verified
- [x] `ModelRegistry.cs` — in-memory loaded model cache — done and verified
- [x] `ModelCacheStore.cs` — disk cache at `~/.sharpinference/models/` — done and verified
- [x] `ModelInfo.cs` — name, format, architecture, size, path — done and verified
- [x] `HuggingFaceClient.cs` — search, download, resolve variants — done and verified
- [x] `HuggingFaceModelIndex.cs` — parse model card metadata — done and verified

## 6. Implementation — SharpInference.Cpu

- [x] `SimdDispatch.cs` — runtime AVX2/AVX-512/NEON detection — done and verified (62 lines)
- [x] `CpuBackend.cs` — `IBackend` implementation routing to kernels — done and verified (202 lines)
- [x] `MatMulKernels.cs` — GEMM, GEMV, batched matmul (AVX2 + AVX-512) — done and verified (131 lines)
- [x] `Conv2DKernels.cs` — im2col + GEMM, 1×1, 3×3, depthwise — done and verified (204 lines)
- [x] `NormKernels.cs` — GroupNorm, LayerNorm, RMSNorm, InstanceNorm — done and verified (327 lines)
- [x] `AttentionKernels.cs` — tiled SDPA (flash-attention style) — done and verified (244 lines)
- [x] `ActivationKernels.cs` — GELU, SiLU, GELU-approx, Mish, Swish — done and verified (54 lines)
- [x] `UpDownSampleKernels.cs` — nearest/bilinear upsample, strided downsample — done and verified (210 lines)
- [x] `AudioKernels.cs` — FFT (Cooley-Tukey), STFT, mel filterbank — done and verified (223 lines)
- [x] `ElementWiseKernels.cs` — Add, Mul, Scale, Clamp, Concat, Split — done and verified (306 lines)
- [x] `ComputeThreadPool.cs` — zero-alloc work-stealing thread pool — done and verified (60 lines)
- [x] `NumaAffinity.cs` — NUMA + P-core pinning — done and verified (45 lines)

## 7. Testing

- [x] `TensorTests.cs` — create, index, slice, reshape, dispose — done (12 tests passing)
- [x] `TensorShapeTests.cs` — stride computation, broadcasting — done (15 tests passing)
- [x] `NativeBufferTests.cs` — alloc, free, alignment, pool round-trip — done (10 tests passing)
- [x] `SafeTensorsLoaderTests.cs` — load synthetic safetensors file, verify tensor values — done (7 tests passing)
- [x] `GgufLoaderTests.cs` — load synthetic GGUF file, verify parsed values — done (7 tests passing)
- [x] `MatMulKernelTests.cs` — compare against known-good values — done (6 tests passing)
- [x] `Conv2DKernelTests.cs` — compare against hand-computed reference — done (4 tests passing)
- [x] `NormKernelTests.cs` — compare against hand-computed norm values — done (5 tests passing)
- [x] `AttentionKernelTests.cs` — compare against hand-computed SDPA — done (4 tests passing)
- [ ] All tests pass on CI

## 8. Review & Merge

- [x] Code review — memory safety (no leaks, proper disposal) — done, 11 issues found and fixed across 9 files
- [x] Code review — SIMD correctness (edge cases at vector boundaries) — done, 4 issues fixed
- [x] Benchmark key kernels (matmul, conv2d, groupnorm) against baseline — done, benchmarks/SharpInference.Benchmarks/
- [x] Document any deviations from design plan — done, see docs/PHASE_1_DEVIATIONS.md
- [ ] Merge to main branch
