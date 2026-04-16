# NuGet Package Design

> Back to [Core Design](CORE_DESIGN.md)

---

## Why Multiple Packages?

A single monolithic NuGet is the wrong choice because:

- A SwarmUI extension only doing image generation should not pull in audio and video code
- CUDA dependencies (PTX compilation, `nvcuda.dll` P/Invoke) should be opt-in
- The server package adds ASP.NET dependencies that desktop apps don't want
- Teams can develop and version individual packages independently
- Users on CPU-only machines still get full functionality without the CUDA package

---

## Package List

### Core Packages

| Package | Description | Key Dependencies |
|---|---|---|
| **SharpInference.Core** | Tensor types, IBackend, IScheduler, IModel, all shared interfaces, enums, Logs class | .NET 10 only |
| **SharpInference.ModelHandler** | Safetensors loader, GGUF loader, model registry, HuggingFace Hub download, model caching | Core |
| **SharpInference.Tokenizers** | BPE/CLIP tokenizer, T5 tokenizer, Whisper tokenizer. Does NOT duplicate dotLLM's tokenizers — wraps them or provides diffusion-specific variants. | Core, Microsoft.ML.Tokenizers |
| **SharpInference.Cpu** | CPU backend: Conv2D, GroupNorm, SDPA, RMSNorm, matmul — all SIMD via `System.Runtime.Intrinsics` and `TensorPrimitives` | Core |
| **SharpInference.Cuda** | CUDA backend: PTX kernel management, cuBLAS HGEMM, cuDNN optional, cross-device tensor copy, memory pool. Follows dotLLM's `CudaDriverApi.cs` P/Invoke pattern exactly | Core, Cpu (fallback) |
| **SharpInference.Vulkan** | Vulkan compute backend: SPIR-V shader management, compute pipeline caching, Vulkan memory allocator, cross-device tensor copy. Extends dotLLM's P/Invoke-to-driver-API philosophy to Vulkan for AMD/Intel/NVIDIA support | Core, Cpu (fallback) |

### Domain Packages

| Package | Description | Key Dependencies |
|---|---|---|
| **SharpInference.Diffusion** | SD1.5/SDXL/SD3/Flux pipelines, UNet, DiT, VAE, CLIP encoder, schedulers, LoRA, ControlNet, samplers | Core, ModelHandler, Tokenizers |
| **SharpInference.Audio** | Whisper STT, Kokoro/Parler TTS, voice conversion, audio preprocessing (STFT, mel), vocoder | Core, ModelHandler |
| **SharpInference.Vision** | CLIP image encoder, text embeddings, YOLO detection, SAM segmentation, face detection | Core, ModelHandler |
| **SharpInference.Video** | LTX-Video, Wan, temporal attention, video VAE — Phase 3 | Diffusion |

### Application Packages

| Package | Description | Key Dependencies |
|---|---|---|
| **SharpInference.Server** | ASP.NET endpoints: OpenAI image/audio API, model management, SSE streaming, auth, rate limiting | All above + ASP.NET Core |
| **SharpInference.SwarmUI** | SwarmUI backend extension that registers SharpInference as an in-process backend. | Server (or direct), SwarmUI SDK |

### Utility Packages (not shipped to production)

| Package | Description |
|---|---|
| **SharpInference.Benchmarks** | BenchmarkDotNet performance benchmarks for kernels and pipelines |
| **SharpInference.Diagnostics** | Activation capture, latent visualization, generation metrics, intermediate layer inspection |
| **SharpInference.Convert** | Model conversion: `.ckpt` → `.safetensors`, FP32→FP16, FP16→Q8_0 |

---

## Dependency Graph

```
                        SharpInference.Core
                    /        |        |        \
     SharpInference.   SharpInference.  SharpInference.  SharpInference.
      ModelHandler         Cpu            Cuda             Vulkan
            |                \      |      /
     SharpInference.          (all three)
      Tokenizers
            |
   ┌────────┼──────────────────────┐
   │        │                      │
 Diffusion  Audio                Vision
   │        │                      │
   └────┬───┘──────────────────────┘
        │
   Video (future)
        │
   Server ─── SwarmUI
```

**Backend selection is runtime — users install whichever GPU backend matches their hardware:**
- NVIDIA GPU → `SharpInference.Cuda` (PTX + cuBLAS, best performance)
- AMD/Intel GPU → `SharpInference.Vulkan` (SPIR-V compute, cross-vendor)
- CPU-only → `SharpInference.Cpu` (always available as fallback)
- Both CUDA + Vulkan can be installed simultaneously — backend auto-selected per device

---

## Minimum Install Examples

### Image generation on NVIDIA GPU

```xml
<PackageReference Include="SharpInference.Diffusion" />
<PackageReference Include="SharpInference.Cuda" />
<PackageReference Include="SharpInference.ModelHandler" />
<!-- SharpInference.Core is a transitive dep, automatically included -->
```

### Image generation on AMD GPU

```xml
<PackageReference Include="SharpInference.Diffusion" />
<PackageReference Include="SharpInference.Vulkan" />
<PackageReference Include="SharpInference.ModelHandler" />
<!-- Same pipeline code, different GPU backend — IBackend abstraction handles it -->
```

### SwarmUI backend extension

```xml
<PackageReference Include="SharpInference.SwarmUI" />
<!-- Pulls in the full stack automatically -->
```

### Audio-only voice assistant

```xml
<PackageReference Include="SharpInference.Audio" />
<PackageReference Include="SharpInference.Cpu" />
<!-- CPU inference is fine for Whisper tiny/base -->
```

### Embedding search service (cross-vendor GPU)

```xml
<PackageReference Include="SharpInference.Vision" />
<PackageReference Include="SharpInference.Vulkan" />
<!-- Works on AMD, Intel, and NVIDIA GPUs -->
```

### Maximum compatibility (auto-detect best GPU)

```xml
<PackageReference Include="SharpInference.Diffusion" />
<PackageReference Include="SharpInference.Cuda" />
<PackageReference Include="SharpInference.Vulkan" />
<PackageReference Include="SharpInference.ModelHandler" />
<!-- Runtime: prefers CUDA if NVIDIA detected, falls back to Vulkan, then CPU -->
```
