# Kernel & GPU-Performance Agent

> Write and optimize compute kernels — CPU SIMD, CUDA PTX, Vulkan SPIR-V — and the GPU math behind them.
> Assumes you've read `AGENTS.md` + `docs/CODE_STYLE.md`. Kernel pitfalls are catalogued in
> `docs/Checklists/TROUBLESHOOTING.md`; the open kernel-perf backlog is in `docs/Checklists/ROADMAP.md §2`.
> Research references: `docs/Research/{CUDA_AND_PTX,CONV2D_CUDA,CUDA_PERFORMANCE,SIMD_INTRINSICS_DOTNET,
> SPIRV_COMPUTE_SHADERS,VULKAN_COMPUTE_API,VULKAN_MEMORY_MANAGEMENT}.md`.

Every kernel needs a **scalar fallback** and **FP32 accumulation even for FP16 inputs**. Validate every new
kernel against the CPU reference within tolerance before shipping.

## CUDA / PTX

```csharp
// ✅ launch args are LOCALS whose addresses are stable; handle is an nint field; PTX loaded from disk
void** args = stackalloc void*[4];
args[0] = &outArg; args[1] = &aArg; args[2] = &bArg; args[3] = &countArg;   // CudaKernels.cs:786
CudaDriverApi.cuLaunchKernel(func, grid,1,1, 256,1,1, 0, stream, (nint)args, 0).ThrowOnError();
// ❌ taking the address of a field/property, or storing handles in Dictionary<string,nint>
void** bad = stackalloc void*[] { &this.OutputPtr };   // unstable address → corruption
```

- `.cu` sources live in `src/HartsyInference.Cuda/Kernels/{attention,conv,dit,lm,dequant,vision,wan,audio}/`; compile
  `nvcc -ptx -arch=compute_80` (target `sm_80` minimum) and **ship the `.ptx` as a content file**
  (`<Content Include="Ptx\*.ptx" CopyToOutputDirectory="PreserveNewest" .../>`), loaded at runtime via
  `CudaModule.LoadFromFile` → `GetFunction` — never an embedded resource. Store each handle as an `nint`
  field. Verify every emitted PTX starts `.version 9.0` (driver JIT caps there — see TROUBLESHOOTING §Toolchain).

```c
// ✅ 64-bit indexing for any spatial kernel at ≥1024² (C·kH·kW·outH·outW overflows u32 → illegal address)
size_t idx = (size_t)c * kH * kW * outH * outW + ...;
// ✅ a gated activation splits on the LAST dim — decompose the flat index; test a multi-row [2,2,2D] tensor
int outerIdx = i / D, d = i % D; float x = in[outerIdx*2*D + d], g = in[outerIdx*2*D + d + D];
// ❌ flat-midpoint split (in[i], in[i+N]) — correct only for one row, garbles everything else
```

- Stream/memory: non-blocking streams race with a synchronous `cuMemcpyHtoD`; `cuMemFreeAsync` is deferred,
  so `Sync()` at stage boundaries + add an OOM-retry; before re-`CacheActivation` on an **in-place** op, null
  `_gpuSyncCallback`/`_gpuDisposeCallback` first. Never read `weight.DataPointer` after preload.

## CPU SIMD & Vulkan SPIR-V

- **SIMD**: `Vector512` → `Vector256` → scalar via `SimdDispatch`; always handle the tail; use
  `TensorPrimitives` where it exists; `[AggressiveInlining]` on small helpers. Watch AVX-512 downclocking.
- **SPIR-V**: target Vulkan 1.3; **pin `requiredSubgroupSize`** (8–64 across vendors — a cross-subgroup
  reduction silently drops partials otherwise); `barrier(); memoryBarrierShared();` after shared writes;
  push constants ≤128 B; per-buffer `VkBufferMemoryBarrier2` (sync2). One `matmul_tiled.comp.glsl` with spec
  constants replaces every `cublasGemmEx`. **Kernel dtype selection follows the OUTPUT tensor, not the
  inputs** (an F16-only guard leaves an F32-output pipeline running the scalar path). Tile-size starting
  points are per-vendor — see `SPIRV_COMPUTE_SHADERS.md § Tile-size tuning`.
- **`src/HartsyInference.Vulkan/Shaders/*.comp.glsl` is the single source of truth; `Spirv/*.spv` is a
  checked-in BUILD ARTIFACT of it — never hand-edit a `.spv`.** Same convention as CUDA's `.cu`→`.ptx`
  (§ above), colocated the same way after the kernel-directory reorg (`Shaders/`+`Spirv/` under the
  package, matching `Kernels/`+`Ptx/`). After ANY `.comp.glsl` edit or new shader, rebuild via `bash
  src/HartsyInference.Vulkan/Shaders/build.sh` and commit the resulting `.spv` in the same change — a new
  shader also needs adding to `build.sh`'s `DTYPE_KERNELS`/`SINGLE_KERNELS` arrays, or it silently never
  builds (a `.comp.glsl` file existing is not enough). `VulkanShaderDriftTests.
  CommittedSpirv_MatchesFreshRebuildFromSource` (in `HartsyInference.Vulkan.Tests`) rebuilds every shader
  and byte-diffs it against the committed `Spirv/` dir to catch exactly this drift — run it after any
  shader change. **Toolchain gotcha**: Ubuntu's `glslang-tools` apt package cannot compile
  `matmul_int8.comp.glsl` (`GL_EXT_integer_dot_product` unsupported in its GLSL frontend) — use the
  LunarG Vulkan SDK's `glslangValidator` instead (`GLSLANG=<path> bash build.sh`; no install needed, see
  `TROUBLESHOOTING.md`). The drift test skips (doesn't fail) when no compiler is resolvable, or when the
  resolved one can't build the current shader set — a toolchain gap reads as inconclusive, not a pass.

## Performance

- **Profile before optimizing** — don't guess the bottleneck; instrument *every* op (an uninstrumented
  Concat or host glue op has hidden the real wall repeatedly). Memory access dominates compute; optimize the
  hot inner loop, not setup. Isolated-kernel `ncu` "Est. Speedup" over-states co-limited kernels — confirm
  end-to-end it/s or tok/s. Pin the GPU: `CUDA_DEVICE_ORDER=PCI_BUS_ID` (not just `CUDA_VISIBLE_DEVICES`, which
  defaults to fastest-first and silently picks the wrong card).
- Priority order: kernel fusion (GroupNorm+SiLU, Conv+bias+act ~1.5–2×) → FP16 / Tensor-Cores ~1.5–2× →
  activation `cuMemPool` → CUDA-graph capture (only helps when host-launch-bound) → FlashAttention-style SDPA.

```csharp
// ✅ benchmarks are tagged so CPU-only CI skips them; results are archived, not just printed
[Trait("Category", "GpuIntegration")] // or "Slow" / a bench-specific trait
// BenchmarkDotNet: [MemoryDiagnoser], [SimpleJob(RuntimeMoniker.Net100)], [Params] over sizes.
// Write results to benchmarks/results/YYYY-MM-DD_component.md with hw/driver/.NET + gap-vs-reference.
```

Validation tolerances: CPU FP32 `1e-5` / FP16 `1e-3`; CUDA-vs-CPU same; SPIR-V-vs-CUDA `1e-3`;
fused-vs-unfused `1e-3`.
