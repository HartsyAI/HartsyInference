# Implementation Details

> Per-component technical approach. For shared patterns (tensor types, P/Invoke, disposal, CUDA launch), see `docs/CODE_STYLE.md` and `docs/Agents/AGENTS.md`.

## Component Overview

| Component | Approach | Key Risk |
|---|---|---|
| Tensor/TensorShape/TensorRef/TensorView/DType | Pure C# sealed class + readonly record struct | None |
| SafeTensors Loader | Pure C# mmap + JSON header | Multi-shard edge cases |
| GGUF Loader | Clean-room impl (dotLLM is GPLv3) | GGUF v3 dequant complexity |
| CPU Kernels | `System.Runtime.Intrinsics` AVX2/AVX-512/NEON | AVX-512 not universal |
| CUDA Backend | PTX via CUDA Driver API P/Invoke | PTX syntax curve |
| Vulkan Backend | SPIR-V via Vulkan API P/Invoke | Vulkan verbosity |
| CLIP/T5 Text Encoder | Pure C# transformer + kernels | Tokenizer must match exactly |
| UNet/DiT | Pure C# using op set | Cross-attention correctness |
| VAE Decoder | Pure C# Conv2D + GroupNorm | Tiled decode seam blending |
| Schedulers | Pure C# math | FP reproducibility |
| LoRA/ControlNet | Delta weights / separate UNet residuals | Flux format differs |
| Whisper STT / Kokoro TTS | Encoder-decoder / HiFiGAN vocoder | Autoregressive + timestamps |
| LLM text generation | Config-driven generic decoder transformer + GGUF quant + KV cache | GPU-resident decode loop |

---

## Core — Tensor & Backend

**IBackend** — op-dispatch interface. Domain ops beyond memory management:
```csharp
void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);
void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);
void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);
void Fft(Tensor output, Tensor input);
void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);
void MelFilterbank(Tensor output, Tensor input, Tensor filters);
Tensor AllocateOnDevice(DeviceKind device, TensorShape shape, DType dtype);
void CopyTo(Tensor source, Tensor destination);
```

**Adaptive Thread Pool:** Custom `ComputeThreadPool` with `delegate*` dispatch. SpinWait (~100ns) for latency-critical (denoising), EventBased for throughput (loading). Auto-switches.

---

## Model Handler — Safetensors

**Format:** 8-byte LE header size → UTF-8 JSON header (dtype, shape, data_offsets) → tensor data.

**Loading:** `MemoryMappedFile.CreateFromFile` → parse JSON header → build index → return `TensorView`s into mmap. Multi-shard: sequential load, unified index with adjusted offsets.

---

## CPU — SIMD Kernels

| Kernel | Approach |
|---|---|
| Conv2D | im2col → GEMM. 1x1 degenerates to direct GEMM |
| GroupNorm | Split channels into groups; `TensorPrimitives` with SIMD dispatch |
| SDPA | Tiled attention (L2-cache-sized tiles), O(N) memory, ~1KB stack per head |
| FFT | Cooley-Tukey radix-2. Butterfly loop uses AVX2 complex multiply |
| Weight repacking | R4-style interleave for `Vector256<T>` sequential reads at load time |
| SIMD dispatch | `Vector512` → `Vector256` → scalar. Mandatory scalar fallback |

---

## CUDA — PTX Backend

See `docs/CODE_STYLE.md` for P/Invoke conventions and `docs/Agents/AGENTS.md` for CUDA launch pattern.

**Kernel fusion:** Minimize memory bandwidth. Key fusions: GroupNorm+SiLU, Conv2D+bias+activation, fused attention. Quantize activations (small) rather than dequantize weights (large).

**Conv2D:** cuDNN first for correctness; custom PTX later. 1x1 → cuBLAS HGEMM.

**cuBLAS:** `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for FP16-in/FP32-accumulate, auto Tensor Cores on Ampere+. Handle once per context.

---

## Vulkan — SPIR-V Backend

> Full research: [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md), [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md), [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md). Build order: [PHASE_3_5_VULKAN_BACKEND.md](../Checklists/PHASE_3_5_VULKAN_BACKEND.md).

Same P/Invoke-to-driver philosophy as CUDA. `[LibraryImport("vulkan-1")]` over the ~55 functions enumerated in [VULKAN_COMPUTE_API.md § P/Invoke Function List](../Research/VULKAN_COMPUTE_API.md#pinvoke-function-list-phase-35-minimum-surface). Loader resolution: `libvulkan.so.1` (Linux), `vulkan-1.dll` (Windows), `libvulkan.1.dylib` / `libMoltenVK.dylib` (macOS).

**Target API:** Vulkan 1.3 — promotes `VK_KHR_synchronization2`, `VK_EXT_subgroup_size_control`, FP16 + subgroup-arithmetic features into core. Required device features at create time: `shaderFloat16`, `storageBuffer16BitAccess`, `subgroupSizeControl`, `computeFullSubgroups`, `synchronization2`. Required subgroup ops: `ARITHMETIC | SHUFFLE | SHUFFLE_RELATIVE`. Fail fast (`UnsupportedDeviceException`) if missing.

**Shader management:** `.glsl` → `.spv` via `glslangValidator --target-env vulkan1.3 -S comp -V -O`. Loaded via `vkCreateShaderModule` → `vkCreateComputePipelines` with `VkSpecializationInfo` for tile sizes / dtype / op variants and `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` to pin wave size per vendor. `VkPipelineCache` persisted to `~/.cache/hartsyinference/vulkan/<deviceUUID>.pipeline_cache` to skip ~50–500 ms re-JIT on cold start.

**Memory:** [Slab allocator](../Research/VULKAN_MEMORY_MANAGEMENT.md#sub-allocation-strategy) with two block sizes: 256 MB for weights / large activations, 16 MB for small tensors. Sub-allocates from `vkAllocateMemory` blocks because the spec only guarantees 4096 simultaneous allocations. ReBAR fast path: when `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory type exists, write weights directly without staging. Deferred-free list keyed by timeline-semaphore value — analogue of `cuMemFreeAsync`.

**GPU weight cache:** `Dictionary<Tensor, VulkanBuffer>` with reference equality, exactly mirroring CUDA's `GpuTransferHelper`. `PreloadWeights` batches uploads (one staging buffer per ~64 MB) to keep SDXL preload under 30 s. Weight cache survives CPU `Tensor.Dispose()` because lookup uses object identity.

**Descriptors:** ~10–12 distinct `VkDescriptorSetLayout` shapes (`L_2SSBO` … `L_5SSBO`, `L_3SSBO_QKV`) deduped at startup. Each pipeline layout pre-built once. Two strategies based on driver: (a) `VK_KHR_push_descriptor` when supported — descriptors written directly into command buffer, no pool, no allocation; (b) descriptor-pool ring fallback — two pools, alternate per phase boundary, reset between phases (`MAX_SETS_PER_POOL = 4096`).

**Push constants:** all small per-dispatch parameters (shapes, strides, eps, scale). 128-byte hard cap (Vulkan minimum). Pinned `stackalloc` — zero allocation per dispatch.

**Synchronization:** **Sync2 + timeline semaphores.** One logical "stream" = one command-buffer chain + one timeline semaphore counter. Each `vkCmdDispatch` followed by a per-buffer `VkBufferMemoryBarrier2` covering the output buffer's range. Lazy-sync activation cache (port of CUDA's `CacheActivation` / `_gpuSyncCallback`): every cached activation records `(srcStage, srcAccess)` from its producer; the consumer emits the barrier on first read.

**Linear / Conv2D / SDPA — no cuBLAS.** A single `matmul_tiled.comp.glsl` shader replaces every `cublasGemmEx` call. CUTLASS-style 3-level tiling (workgroup 128×128×16, subgroup 64×32, thread 8×8 micro-tile). Spec consts toggle FP32/FP16, transpose flags, bias, fused activation. Conv2D = im2col + tiled GEMM + col2bias_add (or fused). SDPA = naive Q×Kᵀ → softmax → ×V (FlashAttention-2 deferred to Phase 4). Performance gate: ≥ 60% of cuBLAS HGEMM on the same NVIDIA HW.

**Subgroup size handling.** Variable per vendor: 32 (NVIDIA / Intel Arc), 32 or 64 (AMD RDNA), 64 (AMD GCN), 8–32 (Intel iGPU). Always pin via `requiredSubgroupSize` at pipeline create time and shadow into a SPIR-V spec constant so reductions can hard-code the stride. Cross-warp reduce uses subgroup arithmetic + `shared` memory exactly like the CUDA `__shfl_xor_sync` + `__shared__` pattern.

**Vendor coverage (Phase 3.5 targets):** NVIDIA Pascal+ (Vulkan 1.3 since R465), AMD RDNA / RDNA2 / RDNA3 (Mesa RADV 23+), AMD GCN5+ (Vega / older), Intel Arc / Xe / UHD (Mesa ANV). Apple MoltenVK and Mali / Adreno deferred.

**Validation:** every kernel asserted within 1e-3 of CPU reference (FP16) and within 1e-3 of CUDA reference on the same NVIDIA hardware. End-to-end SD1.5 same-seed → SSIM > 0.99 vs CUDA.

**Key differences from CUDA:** No cuBLAS — tiled GEMM via subgroup ops. Subgroup size varies (32/64/8-32). Explicit sync (timeline semaphores, pipeline barriers per output buffer). Allocation count limit forces sub-allocator. Descriptor-set bookkeeping (or push descriptors) instead of raw kernel-arg pointers. JIT-compile via `vkCreateComputePipelines` instead of `cuModuleLoadData`. No `cuMemFreeAsync` — emulated with deferred-free list keyed by timeline value.

---

## Diffusion — Pipelines

**Pipeline factory** — `PipelineFactory.LoadAuto` is scaffolding (throws with a list of unresolved design questions); callers construct pipelines directly today. Pipelines report progress via `Action<GenerationProgress>?` callbacks (not `IAsyncEnumerable`).

**UNet (SD1.5):** 4 down, 1 mid, 4 up. ResNetBlock: `GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv+residual`. CrossAttentionBlock: `LayerNorm→self-attn→cross-attn→FFN`. Timestep: sinusoidal→MLP→FiLM addition.

**VAE Tiled Decode:** Split latent into overlapping tiles, decode independently, blend overlaps with linear fade mask.

**LoRA:** Delta `dW = B × A × scale`. Applied in-place or kept additive for multi-LoRA.

---

## Audio — Whisper

**Preprocessing:** 16kHz PCM → 25ms Hann frames (10ms hop) → FFT → mel (80 bins) → log → normalize → `[1, 80, T]`.

**Encoder:** Two Conv1D (stride 1, stride 2) → positional encoding → N transformer blocks.

**Decoder:** Autoregressive transformer with cross-attention to encoder. KV-cache. Token IDs with optional timestamps.

---

## LLM — Native Text Generation

One config-driven `GenericTransformer` (Qwen2/Qwen3/Llama/Mistral) drives decode (causal + KV cache) and also backs bidirectional text encoders. GGUF quantized inference uses fused mul_mat_vec decode kernels (Q4_K/Q6_K/Q8_0) plus a quantized LM head. The per-token loop keeps activations and the KV cache device-resident so only the next token id crosses the PCIe boundary. Full design: [LLM_LANGUAGE_PACKAGE.md](LLM_LANGUAGE_PACKAGE.md).

## Server — dropped

The `HartsyInference.Server` ASP.NET scaffolding remains in `src/` but is **abandoned**. There is no OpenAI-compatible server product and none is planned. The engine is consumed via the SwarmUI backend extension, NuGet libraries, and sample CLIs.
