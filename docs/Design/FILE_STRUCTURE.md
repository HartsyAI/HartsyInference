# File Structure

> Back to [Core Design](CORE_DESIGN.md)

---

## Full Project Layout

```
SharpInference/
│
├── SharpInference.slnx                    Solution file
├── Directory.Build.props                  Global project settings (TargetFramework, Nullable, etc.)
├── Directory.Packages.props               Central package version management
├── CLAUDE.md                              Agent instructions for Claude Code
├── GEMINI.md                              Agent instructions for Gemini
├── README.md
├── LICENSE
├── CONTRIBUTING.md
│
├── src/
│   ├── SharpInference.Core/
│   ├── SharpInference.ModelHandler/
│   ├── SharpInference.Tokenizers/
│   ├── SharpInference.Cpu/
│   ├── SharpInference.Cuda/
│   ├── SharpInference.Vulkan/
│   ├── SharpInference.Diffusion/
│   ├── SharpInference.Audio/
│   ├── SharpInference.Vision/
│   ├── SharpInference.Video/
│   └── SharpInference.Server/
│
├── tests/
├── samples/
├── benchmarks/
├── docs/
└── native/
```

---

## src/SharpInference.Core/

```
SharpInference.Core/
├── SharpInference.Core.csproj
├── Tensors/
│   ├── Tensor.cs                  Core tensor type (unmanaged memory, N-D shape)
│   ├── TensorShape.cs             Shape + stride metadata
│   ├── TensorView.cs              Non-owning view, Dispose() is no-op (dotLLM pattern)
│   ├── TensorMetadata.cs          Lightweight readonly record struct description
│   ├── DType.cs                   readonly record struct with Name field (dotLLM pattern)
│   └── TensorPool.cs              Thread-safe unmanaged buffer pool for temp allocations
├── Backends/
│   ├── IBackend.cs                Full backend interface (matmul, conv, norm, attn, etc.)
│   ├── BackendCapabilities.cs     What ops a backend can handle
│   └── DeviceKind.cs              Cpu, Cuda enum + ordinal
├── Pipelines/
│   ├── IDiffusionPipeline.cs
│   ├── IAudioPipeline.cs
│   ├── IVisionPipeline.cs
│   └── IPipelineRequest.cs        Base request record
├── Schedulers/
│   └── IScheduler.cs              SetTimesteps, Step, AddNoise
├── Models/
│   ├── IModel.cs                  Load, Forward, GetConfig
│   ├── ModelConfig.cs             Architecture params from model metadata
│   └── ModelFormat.cs             SafeTensors, Gguf, Onnx enum
├── Memory/
│   ├── NativeBuffer.cs            Thin wrapper over NativeMemory.AlignedAlloc
│   └── MmapHandle.cs              MemoryMappedFile lifetime manager
├── Logging/
│   └── Logs.cs                    Static Logs class (Info/Debug/Warning/Error)
└── Exceptions/
    ├── SharpInferenceException.cs
    ├── OutOfVramException.cs
    └── UnsupportedModelException.cs
```

---

## src/SharpInference.ModelHandler/

```
SharpInference.ModelHandler/
├── SharpInference.ModelHandler.csproj
├── SafeTensors/
│   ├── SafeTensorsLoader.cs       mmap + JSON header parse → TensorView dict
│   ├── SafeTensorsWriter.cs       Save tensors back to safetensors format
│   └── SafeTensorsShardLoader.cs  Multi-file sharded models
├── Gguf/
│   ├── GgufLoader.cs              GGUF header + metadata + tensor mmap
│   ├── GgufDequantizer.cs         Dequantize Q4_0/Q8_0/Q4_K_M blocks to F16/F32
│   └── GgufMetadata.cs            Typed metadata key-value access
├── Registry/
│   ├── ModelRegistry.cs           In-memory loaded model cache
│   ├── ModelCacheStore.cs         Disk cache at ~/.sharpinference/models/
│   └── ModelInfo.cs               Name, format, architecture, size, local path
├── HuggingFace/
│   ├── HuggingFaceClient.cs       Search, pull, resolve GGUF variant
│   └── HuggingFaceModelIndex.cs   Parse model card metadata
├── CheckpointConverters/
│   ├── Utils/
│   │   └── CheckpointConvertUtils.cs  Shared key remapping (ResNet, VAE, time_embed, tensor splitting)
│   ├── Sd15CheckpointConverter.cs     Single-file SD1.5 → diffusers format (UNet + CLIP-L + VAE)
│   ├── SdxlCheckpointConverter.cs     Single-file SDXL → diffusers format (UNet + CLIP-L + CLIP-G + VAE)
│   ├── FluxCheckpointConverter.cs     (planned) Single-file Flux → diffusers format
│   └── Sd3CheckpointConverter.cs      (planned) Single-file SD3 → diffusers format
└── Convert/
    ├── CheckpointConverter.cs     .ckpt → .safetensors (planned)
    └── QuantizeConverter.cs       FP32 → FP16, FP16 → Q8_0 (planned)
```

---

## src/SharpInference.Tokenizers/

```
SharpInference.Tokenizers/
├── SharpInference.Tokenizers.csproj
├── ClipTokenizer.cs               BPE tokenizer matching CLIP vocab, 77-token limit
├── T5Tokenizer.cs                 SentencePiece T5, 4096-token context
├── WhisperTokenizer.cs            Whisper multilingual BPE + special tokens
└── TokenizerCache.cs              Reuse tokenizers across pipeline instances
```

---

## src/SharpInference.Cpu/

```
SharpInference.Cpu/
├── SharpInference.Cpu.csproj
├── CpuBackend.cs                  IBackend implementation — routes to SIMD kernels
├── Kernels/
│   ├── MatMulKernels.cs           GEMM (row-major), GEMV, batched matmul
│   ├── Conv2DKernels.cs           im2col + GEMM, 1×1, 3×3, depthwise
│   ├── NormKernels.cs             LayerNorm, RMSNorm, GroupNorm, InstanceNorm
│   ├── AttentionKernels.cs        SDPA, tiled flash-attention-style O(N) memory
│   ├── ActivationKernels.cs       GELU, SiLU, GELU-approx, Mish, Swish
│   ├── UpDownSampleKernels.cs     Nearest, bilinear upsample; strided downsample
│   ├── AudioKernels.cs            FFT (Cooley-Tukey), STFT, mel filterbank
│   └── ElementWiseKernels.cs      Add, Mul, Scale, Clamp, Concat, Split
├── Threading/
│   ├── ComputeThreadPool.cs       Zero-alloc work-stealing pool
│   └── NumaAffinity.cs            NUMA + P-core pinning (Windows + Linux)
└── SimdDispatch.cs                Runtime AVX2/AVX-512/NEON detection and dispatch
```

---

## src/SharpInference.Cuda/

```
SharpInference.Cuda/
├── SharpInference.Cuda.csproj
├── CudaBackend.cs                 IBackend implementation — routes to PTX + cuBLAS
├── CudaDriverApi.cs               P/Invoke surface: "cuda" lib name, int returns (dotLLM pattern)
├── CuBlasWrapper.cs               cuBLAS HGEMM, SGEMM
├── CuDnnWrapper.cs                cuDNN Conv2D (optional, fallback path)
├── CudaMemoryPool.cs              cuMemPool-based async memory pool
├── CudaStream.cs                  Stream lifecycle management
├── CudaKernels.cs                 All kernel function handles as nint fields, loaded in constructor
├── CudaModule.cs                  LoadFromFile() + GetFunction() wrapper (dotLLM pattern)
├── CudaLibraryResolver.cs         Maps "cuda" -> nvcuda.dll / libcuda.so at runtime
└── Ptx/                           PTX content files loaded from this directory at runtime (NOT embedded)
    ├── conv2d_f16_3x3.ptx
    ├── conv2d_f16_1x1.ptx
    ├── group_norm_f16.ptx
    ├── group_norm_silu_fused.ptx    Fused GroupNorm+SiLU (bandwidth optimization)
    ├── layer_norm_f16.ptx
    ├── sdpa_f16.ptx
    ├── upsample2d_nearest.ptx
    ├── upsample2d_bilinear.ptx
    ├── fft_radix2.ptx
    ├── mel_filterbank.ptx
    ├── elementwise_f16.ptx
    ├── silu_f16.ptx
    ├── gelu_f16.ptx
    ├── rope_2d.ptx                  2D RoPE for Flux/SD3 DiT
    ├── timestep_embed.ptx
    ├── conv2d_bias_silu_fused.ptx   Fused Conv2D+bias+SiLU
    ├── dequant_q8.ptx
    └── dequant_q4k.ptx
```

---

## src/SharpInference.Vulkan/

```
SharpInference.Vulkan/
├── SharpInference.Vulkan.csproj
├── VulkanBackend.cs               IBackend implementation — routes to SPIR-V compute shaders
├── VulkanApi.cs                   P/Invoke surface for Vulkan API (~40 functions)
├── VulkanDevice.cs                Physical/logical device selection, queue families
├── VulkanMemoryAllocator.cs       Sub-allocation from large device memory blocks
├── VulkanCommandPool.cs           Command buffer lifecycle management
├── VulkanDescriptorManager.cs     Descriptor set layout and pool management
├── SpirVShaderLoader.cs           Load SPIR-V from disk, create compute pipelines (mirrors CudaModule)
├── VulkanKernels.cs               Compute pipeline handles, dispatch wrappers (mirrors CudaKernels)
├── VulkanLibraryResolver.cs       Cross-platform vulkan-1.dll / libvulkan.so.1 resolution
└── Spirv/
    ├── conv2d_f16_3x3.spv
    ├── conv2d_f16_1x1.spv
    ├── group_norm_f16.spv
    ├── group_norm_silu_fused.spv
    ├── layer_norm_f16.spv
    ├── sdpa_f16.spv
    ├── upsample2d_nearest.spv
    ├── upsample2d_bilinear.spv
    ├── fft_radix2.spv
    ├── mel_filterbank.spv
    ├── elementwise_f16.spv
    ├── silu_f16.spv
    ├── gelu_f16.spv
    ├── rope_2d.spv
    ├── timestep_embed.spv
    ├── conv2d_bias_silu_fused.spv
    ├── dequant_q8.spv
    ├── dequant_q4k.spv
    └── matmul_tiled.spv            Tiled GEMM via subgroup ops (no cuBLAS equivalent)
```

---

## src/SharpInference.Diffusion/

```
SharpInference.Diffusion/
├── SharpInference.Diffusion.csproj
├── Pipelines/
│   ├── StableDiffusion15Pipeline.cs
│   ├── SdxlPipeline.cs
│   ├── SdxlRefinerPipeline.cs
│   ├── Sd3Pipeline.cs
│   ├── FluxPipeline.cs
│   └── PipelineFactory.cs         Auto-detect model arch → return correct pipeline
├── Models/
│   ├── TextEncoders/
│   │   ├── ClipTextEncoder.cs
│   │   ├── ClipTextEncoderG.cs
│   │   └── T5TextEncoder.cs
│   ├── Denoisers/
│   │   ├── UNet.cs
│   │   ├── DiT.cs
│   │   ├── UNetBlocks/
│   │   │   ├── ResNetBlock.cs
│   │   │   ├── CrossAttentionBlock.cs
│   │   │   ├── DownSampleBlock.cs
│   │   │   └── UpSampleBlock.cs
│   │   └── DiTBlocks/
│   │       ├── MmDiTBlock.cs
│   │       ├── SingleStreamBlock.cs
│   │       └── DoubleStreamBlock.cs
│   └── Vae/
│       ├── VaeEncoder.cs
│       ├── VaeDecoder.cs
│       └── VaeTiledDecoder.cs
├── Schedulers/
│   ├── EulerDiscreteScheduler.cs
│   ├── DpmPlusPlus2MScheduler.cs
│   ├── DpmPlusPlus2MSdeScheduler.cs
│   ├── DdimScheduler.cs
│   ├── LcmScheduler.cs
│   └── FlowMatchEulerScheduler.cs
├── Adapters/
│   ├── LoraLoader.cs
│   ├── LoraManager.cs
│   ├── ControlNetLoader.cs
│   └── IpAdapterLoader.cs
├── Requests/
│   ├── TextToImageRequest.cs
│   ├── ImageToImageRequest.cs
│   ├── InpaintRequest.cs
│   └── GenerationProgress.cs
└── Utilities/
    ├── LatentPreviewDecoder.cs
    ├── ImagePostProcessor.cs
    └── SeedGenerator.cs
```

---

## src/SharpInference.Audio/

```
SharpInference.Audio/
├── SharpInference.Audio.csproj
├── Stt/
│   ├── WhisperPipeline.cs
│   ├── WhisperEncoder.cs
│   ├── WhisperDecoder.cs
│   ├── WhisperStreamingPipeline.cs
│   └── WhisperOptions.cs
├── Tts/
│   ├── KokoroPipeline.cs
│   ├── KokoroPhonemeEncoder.cs
│   ├── HiFiGanVocoder.cs
│   ├── VocosVocoder.cs
│   └── TtsOptions.cs
├── VoiceConversion/
│   ├── RvcPipeline.cs
│   └── F0Extractor.cs
└── Preprocessing/
    ├── AudioPreprocessor.cs
    ├── StftProcessor.cs
    └── MelSpectrogramProcessor.cs
```

---

## src/SharpInference.Vision/

```
SharpInference.Vision/
├── SharpInference.Vision.csproj
├── Clip/
│   ├── ClipImageEncoder.cs
│   └── ClipScorer.cs
├── Embeddings/
│   ├── ImageEmbeddingPipeline.cs
│   └── TextEmbeddingPipeline.cs
├── Detection/
│   ├── YoloPipeline.cs
│   ├── YoloPostProcessor.cs
│   └── DetectionResult.cs
├── Segmentation/
│   ├── SamPipeline.cs
│   └── SamMaskDecoder.cs
└── FaceDetection/
    ├── FaceDetector.cs
    └── LandmarkExtractor.cs
```

---

## src/SharpInference.Server/

```
SharpInference.Server/
├── SharpInference.Server.csproj
├── Setup/
│   ├── SharpInferenceServiceExtensions.cs
│   └── SharpInferenceServerOptions.cs
├── Endpoints/
│   ├── ImageGenerationEndpoints.cs
│   ├── AudioTranscriptionEndpoints.cs
│   ├── VisionEndpoints.cs
│   └── ModelManagementEndpoints.cs
├── Streaming/
│   ├── SseProgressStream.cs
│   └── AudioChunkStream.cs
├── Queue/
│   ├── InferenceQueue.cs
│   └── InferenceQueueEntry.cs
└── Auth/
    └── ApiKeyMiddleware.cs
```

---

## tests/

```
tests/
├── SharpInference.Core.Tests/
│   ├── TensorTests.cs
│   ├── TensorShapeTests.cs
│   └── NativeBufferTests.cs
├── SharpInference.ModelHandler.Tests/
│   ├── SafeTensorsLoaderTests.cs
│   └── GgufLoaderTests.cs
├── SharpInference.Cpu.Tests/
│   ├── MatMulKernelTests.cs
│   ├── Conv2DKernelTests.cs
│   ├── NormKernelTests.cs
│   └── AttentionKernelTests.cs
├── SharpInference.Diffusion.Tests/
│   ├── SchedulerTests.cs
│   ├── ClipTokenizerTests.cs
│   └── PipelineIntegrationTests.cs
├── SharpInference.Audio.Tests/
│   ├── StftTests.cs
│   ├── MelSpectrogramTests.cs
│   └── WhisperIntegrationTests.cs
└── SharpInference.Server.Tests/
    └── ImageApiTests.cs
```

---

## samples/

```
samples/
├── BasicImageGeneration/          Console app: text → image
├── StreamingServer/               ASP.NET with SSE image progress
├── VoiceAssistant/                Whisper STT + dotLLM LLM + Kokoro TTS loop
└── SwarmUIExtension/              Full SwarmUI backend extension example
```

---

## benchmarks/

```
benchmarks/
├── SharpInference.Benchmarks/
│   ├── KernelBenchmarks.cs
│   ├── PipelineBenchmarks.cs
│   └── AudioBenchmarks.cs
└── scripts/
    ├── bench_compare.py
    └── bench_trend.py
```

---

## native/

```
native/
├── cuda/
│   ├── kernels/                   CUDA C++ source for PTX generation
│   │   ├── conv2d.cu
│   │   ├── group_norm.cu
│   │   ├── group_norm_silu_fused.cu
│   │   ├── sdpa.cu
│   │   ├── fft.cu
│   │   ├── silu.cu
│   │   ├── gelu.cu
│   │   ├── rope_2d.cu
│   │   ├── timestep_embed.cu
│   │   ├── conv2d_bias_silu_fused.cu
│   │   ├── dequant_q8.cu
│   │   └── dequant_q4k.cu
│   └── build.sh                   nvcc -ptx -arch=compute_80 → PTX
└── vulkan/
    ├── shaders/                   GLSL compute shaders for SPIR-V generation
    │   ├── conv2d.comp.glsl
    │   ├── group_norm.comp.glsl
    │   ├── group_norm_silu_fused.comp.glsl
    │   ├── sdpa.comp.glsl
    │   ├── fft.comp.glsl
    │   ├── silu.comp.glsl
    │   ├── gelu.comp.glsl
    │   ├── rope_2d.comp.glsl
    │   ├── timestep_embed.comp.glsl
    │   ├── conv2d_bias_silu_fused.comp.glsl
    │   ├── matmul_tiled.comp.glsl
    │   ├── dequant_q8.comp.glsl
    │   └── dequant_q4k.comp.glsl
    └── build.sh                   glslangValidator --target-env vulkan1.2 → SPIR-V
```
