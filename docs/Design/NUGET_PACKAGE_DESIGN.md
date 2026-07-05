# NuGet Package Design

> Back to [Core Design](CORE_DESIGN.md)

## Why Multiple Packages?

Monolithic NuGet is wrong because:
- SwarmUI extensions and desktop apps shouldn't pull audio/video code they don't use
- CUDA dependencies should be opt-in
- LLM text generation is its own package, so an image-only consumer can depend on `HartsyInference.Diffusion` directly without pulling it in
- CPU-only users shouldn't need CUDA

A single meta-package (`HartsyInference`) is published for consumers who want the whole modality stack in one reference; it bundles every modality package, LLM included (see the meta-package note below). Consumers who want only one modality reference that package directly instead.

## Package List

### Core

| Package | Description | Dependencies |
|---|---|---|
| **Core** | Tensor types, `IBackend`, `IScheduler`, `IModel`, interfaces, enums, logs | net8.0 / net10.0 |
| **ModelHandler** | Safetensors/GGUF/PyTorch-pickle loaders, registry, HF Hub download, caching | Core |
| **Tokenizers** | BPE (CLIP), SentencePiece (T5), Whisper tokenizer | Core, Microsoft.ML.Tokenizers |
| **Phonemizer** | Pure-C# espeak-ng-style G2P / IPA phonemization for phoneme-input TTS | Core |
| **Cpu** | SIMD kernels: Conv2D, GroupNorm, SDPA, matmul via `System.Runtime.Intrinsics` | Core |
| **Cuda** | PTX management, cuBLAS HGEMM, cuDNN optional, memory pool | Core, Cpu (fallback) |
| **Vulkan** | SPIR-V shaders, compute pipelines, memory allocator, cross-vendor GPU | Core, Cpu (fallback) |

### Domain

| Package | Description | Dependencies |
|---|---|---|
| **LLM** | Native LLM text generation: config-driven generic decoder transformer (Qwen2/Qwen3/Llama/Mistral), GGUF quantized inference, device-resident KV cache, sampler chain, chat templates. Also backs text encoders. | Core, ModelHandler, Tokenizers |
| **Diffusion** | SD1.5/SDXL/SD3/Flux pipelines, UNet, DiT, VAE, CLIP, schedulers, LoRA, ControlNet | Core, ModelHandler, Tokenizers |
| **Audio** | Whisper STT, Kokoro/Parler TTS, voice conversion, STFT/mel, vocoder | Core, ModelHandler |
| **Vision** | CLIP image encoder, embeddings, YOLO, SAM, face detection | Core, ModelHandler |
| **Video** | LTX-Video, Wan, Lance video, Cosmos-Predict V2W. 3D causal VAE, temporal attention, packed/varlen attention, distilled schedulers, discrete video tokenizers (Cosmos DV / VQ-GAN). Hosts the shared infra that Interactive consumes. | Diffusion |
| **Interactive** | Action-conditioned, real-time, frame-by-frame world models (Matrix-Game 2/3, Oasis, Hunyuan-GameCraft). `IInteractiveSession` streaming loop, `IActionEncoder` abstractions, memory-augmented cross-attn primitive, CameraNet. Strictly user-driven runtime — does not appear in offline pipelines. | Video |
| **ThreeD** | 3D asset generation (image→mesh/splat). Representation-agnostic foundation: marching cubes, glTF/OBJ/PLY export, triplane/grid sampling, geometry types. Models: Hunyuan3D-2 (flow-match DiT + ShapeVAE) and TripoSR (triplane/NeRF LRM). | Diffusion, Vision |

### Meta package

| Package | Description | References |
|---|---|---|
| **HartsyInference** | Dependencies-only meta-package: one reference pulls the whole modality stack | Core, Cpu, Cuda, Vulkan, ModelHandler, Tokenizers, Phonemizer, LLM, Diffusion, Audio, Vision, Video, Interactive, ThreeD |

> **Meta-package note.** `HartsyInference` explicitly references all 14 libraries above, including **LLM** and **Phonemizer**, so a single `dotnet add package HartsyInference` gives you native LLM text generation and phoneme-input TTS. Only **Server** (abandoned scaffolding) and **Cli** (a sample/validation tool, not a library) are excluded. Consumers who want just one modality reference that package directly instead of the meta.

### Consuming the engine

The engine is consumed three ways, in priority order:

1. **SwarmUI backend extension (recommended):** the [SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend) repo. It is a SwarmUI extension (not a package in this repo) that registers HartsyInference as an alternative to the ComfyUI backend and drives the pipelines.
2. **NuGet libraries:** reference the meta-package or individual modality packages.
3. **Sample CLIs:** the per-modality apps under `samples/` and `src/HartsyInference.Cli` (developer and verification tools).

> **Dropped: first-party server.** `HartsyInference.Server` physically exists in `src/` as abandoned ASP.NET scaffolding. It is **not** a supported or advertised path: there is no OpenAI-compatible server product, and none is planned. Do not depend on it.

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
  LLM   Diffusion  Audio  Vision
        |       |       |
        \_______|_______/
                |
           Video (Phase 9)         ThreeD (Phase 11 — Diffusion + Vision)
                |
         Interactive (Phase 10 — world models)

  Consumers: SwarmUI backend extension (external repo) · CLIs · apps
```

LLM depends only on Core + ModelHandler + Tokenizers (not on the visual stack). Phonemizer depends on Core.

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

**SwarmUI backend:** installed as a SwarmUI extension from [SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend), not a NuGet reference in your own project. The extension pins the engine packages it needs.

**Native LLM text generation (NVIDIA):**
```xml
<PackageReference Include="HartsyInference.LLM" />
<PackageReference Include="HartsyInference.Cuda" />
<PackageReference Include="HartsyInference.ModelHandler" />
<PackageReference Include="HartsyInference.Tokenizers" />
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
