# NuGet Package Design

> Back to [Core Design](CORE_DESIGN.md)

## Why Multiple Packages?

Monolithic NuGet is wrong because:
- SwarmUI extensions shouldn't pull audio/video code
- CUDA dependencies should be opt-in
- Server adds ASP.NET deps desktop apps don't want
- CPU-only users shouldn't need CUDA

## Package List

### Core

| Package | Description | Dependencies |
|---|---|---|
| **Core** | Tensor types, `IBackend`, `IScheduler`, `IModel`, interfaces, enums, logs | .NET 10 only |
| **ModelHandler** | Safetensors/GGUF loaders, registry, HF Hub download, caching | Core |
| **Tokenizers** | BPE (CLIP), SentencePiece (T5), Whisper tokenizer | Core, Microsoft.ML.Tokenizers |
| **Cpu** | SIMD kernels: Conv2D, GroupNorm, SDPA, matmul via `System.Runtime.Intrinsics` | Core |
| **Cuda** | PTX management, cuBLAS HGEMM, cuDNN optional, memory pool | Core, Cpu (fallback) |
| **Vulkan** | SPIR-V shaders, compute pipelines, memory allocator, cross-vendor GPU | Core, Cpu (fallback) |

### Domain

| Package | Description | Dependencies |
|---|---|---|
| **Diffusion** | SD1.5/SDXL/SD3/Flux pipelines, UNet, DiT, VAE, CLIP, schedulers, LoRA, ControlNet | Core, ModelHandler, Tokenizers |
| **Audio** | Whisper STT, Kokoro/Parler TTS, voice conversion, STFT/mel, vocoder | Core, ModelHandler |
| **Vision** | CLIP image encoder, embeddings, YOLO, SAM, face detection | Core, ModelHandler |
| **Video** | LTX-Video, Wan, Lance video, Cosmos-Predict V2W. 3D causal VAE, temporal attention, packed/varlen attention, distilled schedulers, discrete video tokenizers (Cosmos DV / VQ-GAN). Hosts the shared infra that Interactive consumes. | Diffusion |
| **Interactive** | Action-conditioned, real-time, frame-by-frame world models (Matrix-Game 2/3, Oasis, Hunyuan-GameCraft). `IInteractiveSession` streaming loop, `IActionEncoder` abstractions, memory-augmented cross-attn primitive. Strictly user-driven runtime — does not appear in offline pipelines. | Video |

### Application

| Package | Description | Dependencies |
|---|---|---|
| **Server** | ASP.NET OpenAI API, SSE, model mgmt, auth, rate limiting | All above + ASP.NET Core |
| **SwarmUI** | In-process SwarmUI backend extension | Server, SwarmUI SDK |

### Utility (not shipped)

| Package | Description |
|---|---|
| **Benchmarks** | BenchmarkDotNet for kernels/pipelines |
| **Diagnostics** | Activation capture, latent viz, metrics |
| **Convert** | `.ckpt`→`.safetensors`, FP32→FP16, FP16→Q8_0 |

## Dependency Graph

```
              Core
       /  |      |  \
  ModelHandler  Cpu  Cuda  Vulkan
       |          \  |  /
   Tokenizers    (all three)
       |
  Diffusion  Audio  Vision
       |       |       |
       \_______|_______/
               |
          Video (future)
               |
        Interactive (Phase 10 — world models)
               |
          Server → SwarmUI
```

**Runtime backend selection:**
- NVIDIA → Cuda (best perf)
- AMD/Intel → Vulkan
- CPU-only → Cpu (always fallback)
- Cuda + Vulkan can coexist — auto-detected per device

## Minimum Install Examples

**NVIDIA image gen:**
```xml
<PackageReference Include="HartsyInference.Diffusion" />
<PackageReference Include="HartsyInference.Cuda" />
<PackageReference Include="HartsyInference.ModelHandler" />
```

**AMD image gen:**
```xml
<PackageReference Include="HartsyInference.Diffusion" />
<PackageReference Include="HartsyInference.Vulkan" />
<PackageReference Include="HartsyInference.ModelHandler" />
```

**SwarmUI backend:**
```xml
<PackageReference Include="HartsyInference.SwarmUI" />
```

**Audio-only (CPU):**
```xml
<PackageReference Include="HartsyInference.Audio" />
<PackageReference Include="HartsyInference.Cpu" />
```

**Max compatibility:**
```xml
<PackageReference Include="HartsyInference.Diffusion" />
<PackageReference Include="HartsyInference.Cuda" />
<PackageReference Include="HartsyInference.Vulkan" />
<PackageReference Include="HartsyInference.ModelHandler" />
```

**Interactive world model (game-engine integration):**
```xml
<PackageReference Include="HartsyInference.Interactive" />
<PackageReference Include="HartsyInference.Cuda" />
<!-- Brings Video + Diffusion + ModelHandler transitively -->
```
