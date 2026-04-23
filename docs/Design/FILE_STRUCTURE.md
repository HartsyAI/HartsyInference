# File Structure

> Back to [Core Design](CORE_DESIGN.md)

## Full Project Layout

```
SharpInference/
├── SharpInference.slnx / Directory.Build.props / Directory.Packages.props
├── CLAUDE.md / GEMINI.md / README.md / LICENSE / CONTRIBUTING.md
├── src/
│   ├── SharpInference.Core / ModelHandler / Tokenizers / Cpu / Cuda / Vulkan
│   ├── SharpInference.Diffusion / Audio / Vision / Video / Server
├── tests/ / samples/ / benchmarks/ / docs/ / native/
```

---

## src/SharpInference.Core/

| File | Description |
|---|---|
| `Tensors/Tensor.cs` | Core tensor type (unmanaged memory, N-D shape) |
| `Tensors/TensorShape.cs` | Shape + stride metadata |
| `Tensors/TensorView.cs` | Non-owning view, `Dispose()` no-op |
| `Tensors/TensorMetadata.cs` | Lightweight readonly record struct description |
| `Tensors/DType.cs` | Data type metadata |
| `Tensors/TensorPool.cs` | Thread-safe unmanaged buffer pool |
| `Backends/IBackend.cs` | Full backend interface |
| `Backends/BackendCapabilities.cs` | Backend capability flags |
| `Backends/DeviceKind.cs` | Cpu / Cuda / Vulkan enum |
| `Pipelines/` | `IDiffusionPipeline`, `IAudioPipeline`, `IVisionPipeline`, `IPipelineRequest` |
| `Schedulers/IScheduler.cs` | `SetTimesteps`, `Step`, `AddNoise` |
| `Models/IModel.cs` / `ModelConfig.cs` / `ModelFormat.cs` | Load/Forward/GetConfig, architecture params, format enum |
| `Memory/NativeBuffer.cs` / `MmapHandle.cs` | `NativeMemory.AlignedAlloc` wrapper, `MemoryMappedFile` lifetime manager |
| `Logging/Logs.cs` | Static Logs class |
| `Exceptions/` | `SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException` |

---

## src/SharpInference.ModelHandler/

| File | Description |
|---|---|
| `SafeTensors/SafeTensorsLoader.cs` | mmap + JSON header → TensorView dict |
| `SafeTensors/SafeTensorsWriter.cs` | Save tensors to safetensors |
| `SafeTensors/SafeTensorsShardLoader.cs` | Multi-file sharded models |
| `Gguf/GgufLoader.cs` | GGUF header + metadata + tensor mmap |
| `Gguf/GgufDequantizer.cs` | Dequantize Q4_0/Q8_0/Q4_K_M to F16/F32 |
| `Gguf/GgufMetadata.cs` | Typed metadata key-value access |
| `Registry/ModelRegistry.cs` | In-memory model cache |
| `Registry/ModelCacheStore.cs` | Disk cache at `~/.sharpinference/models/` |
| `Registry/ModelInfo.cs` | Model metadata |
| `HuggingFace/HuggingFaceClient.cs` | Search, pull, resolve GGUF variant |
| `CheckpointConverters/` | SD1.5 / SDXL / Flux / SD3 `.ckpt` → diffusers format |
| `Convert/CheckpointConverter.cs` | `.ckpt` → `.safetensors` (planned) |
| `Convert/QuantizeConverter.cs` | FP32 → FP16, FP16 → Q8_0 (planned) |

---

## src/SharpInference.Tokenizers/

| File | Description |
|---|---|
| `ClipTokenizer.cs` | BPE tokenizer, 77-token limit |
| `T5Tokenizer.cs` | SentencePiece, 4096-token context |
| `WhisperTokenizer.cs` | Multilingual BPE + special tokens |
| `TokenizerCache.cs` | Reuse tokenizers across pipeline instances |

---

## src/SharpInference.Cpu/

| File | Description |
|---|---|
| `CpuBackend.cs` | IBackend implementation — routes to SIMD kernels |
| `Kernels/MatMulKernels.cs` | GEMM, GEMV, batched matmul |
| `Kernels/Conv2DKernels.cs` | im2col + GEMM, 1×1, 3×3, depthwise |
| `Kernels/NormKernels.cs` | LayerNorm, RMSNorm, GroupNorm, InstanceNorm |
| `Kernels/AttentionKernels.cs` | SDPA, tiled O(N) memory |
| `Kernels/ActivationKernels.cs` | GELU, SiLU, GELU-approx, Mish, Swish |
| `Kernels/UpDownSampleKernels.cs` | Nearest, bilinear upsample; strided downsample |
| `Kernels/AudioKernels.cs` | FFT, STFT, mel filterbank |
| `Kernels/ElementWiseKernels.cs` | Add, Mul, Scale, Clamp, Concat, Split |
| `Threading/ComputeThreadPool.cs` | Zero-alloc work-stealing pool |
| `Threading/NumaAffinity.cs` | NUMA + P-core pinning (Windows + Linux) |
| `SimdDispatch.cs` | Runtime AVX2/AVX-512/NEON detection and dispatch |

---

## src/SharpInference.Cuda/

| File | Description |
|---|---|
| `CudaBackend.cs` | IBackend implementation — routes to PTX + cuBLAS |
| `CudaDriverApi.cs` | P/Invoke surface (~40 functions, `int` returns) |
| `CuBlasWrapper.cs` | cuBLAS HGEMM, SGEMM |
| `CuDnnWrapper.cs` | cuDNN Conv2D (optional fallback) |
| `CudaMemoryPool.cs` | cuMemPool-based async memory pool |
| `CudaStream.cs` | Stream lifecycle management |
| `CudaKernels.cs` | Kernel function handles as `nint` fields, loaded in constructor |
| `CudaModule.cs` | `LoadFromFile()` + `GetFunction()` wrapper (dotLLM pattern) |
| `CudaLibraryResolver.cs` | Maps `"cuda"` → `nvcuda.dll` / `libcuda.so` at runtime |
| `Ptx/*.ptx` | 18 kernel families (not embedded — loaded from disk at runtime) |

---

## src/SharpInference.Vulkan/

| File | Description |
|---|---|
| `VulkanBackend.cs` | IBackend implementation — routes to SPIR-V compute shaders |
| `VulkanApi.cs` | P/Invoke surface (~40 functions) |
| `VulkanDevice.cs` | Physical/logical device selection, queue families |
| `VulkanMemoryAllocator.cs` | Sub-allocation from large device memory blocks |
| `VulkanCommandPool.cs` | Command buffer lifecycle |
| `VulkanDescriptorManager.cs` | Descriptor set layout and pool management |
| `SpirVShaderLoader.cs` | Load SPIR-V from disk, create compute pipelines |
| `VulkanKernels.cs` | Compute pipeline handles, dispatch wrappers |
| `VulkanLibraryResolver.cs` | Cross-platform `vulkan-1.dll` / `libvulkan.so.1` resolution |
| `Spirv/*.spv` | 19 kernel families (loaded from disk at runtime) |

---

## src/SharpInference.Diffusion/

| File | Description |
|---|---|
| `Pipelines/StableDiffusion15Pipeline.cs` / `SdxlPipeline.cs` / `SdxlRefinerPipeline.cs` / `Sd3Pipeline.cs` / `FluxPipeline.cs` | Pipeline implementations |
| `Pipelines/PipelineFactory.cs` | Auto-detect model arch → correct pipeline |
| `Models/TextEncoders/ClipTextEncoder.cs` / `ClipTextEncoderG.cs` / `T5TextEncoder.cs` | CLIP-L, CLIP-G, T5 text encoders |
| `Models/Denoisers/UNet.cs` / `DiT.cs` | UNet and DiT denoisers |
| `Models/Denoisers/UNetBlocks/` | ResNetBlock, CrossAttentionBlock, DownSampleBlock, UpSampleBlock |
| `Models/Denoisers/DiTBlocks/` | MmDiTBlock, SingleStreamBlock, DoubleStreamBlock |
| `Models/Vae/VaeEncoder.cs` / `VaeDecoder.cs` / `VaeTiledDecoder.cs` | VAE encode/decode/tiled decode |
| `Schedulers/` | EulerDiscrete, DPM++ 2M, DPM++ 2M SDE, DDIM, LCM, FlowMatchEuler |
| `Adapters/LoraLoader.cs` / `LoraManager.cs` / `ControlNetLoader.cs` / `IpAdapterLoader.cs` | LoRA, ControlNet, IP-Adapter |
| `Requests/` | TextToImage, ImageToImage, Inpaint, GenerationProgress |
| `Utilities/LatentPreviewDecoder.cs` / `ImagePostProcessor.cs` / `SeedGenerator.cs` | Preview decode, post-process, seed |

---

## src/SharpInference.Audio/

| File | Description |
|---|---|
| `Stt/WhisperPipeline.cs` / `WhisperEncoder.cs` / `WhisperDecoder.cs` / `WhisperStreamingPipeline.cs` / `WhisperOptions.cs` | Whisper STT pipeline, streaming, options |
| `Tts/KokoroPipeline.cs` / `KokoroPhonemeEncoder.cs` / `HiFiGanVocoder.cs` / `VocosVocoder.cs` / `TtsOptions.cs` | Kokoro TTS pipeline, phoneme encoder, vocoders |
| `VoiceConversion/RvcPipeline.cs` / `F0Extractor.cs` | Voice conversion |
| `Preprocessing/AudioPreprocessor.cs` / `StftProcessor.cs` / `MelSpectrogramProcessor.cs` | Audio preprocessing |

---

## src/SharpInference.Vision/

| File | Description |
|---|---|
| `Clip/ClipImageEncoder.cs` / `ClipScorer.cs` | CLIP image encoder, text-image similarity |
| `Embeddings/ImageEmbeddingPipeline.cs` / `TextEmbeddingPipeline.cs` | Image and text embedding pipelines |
| `Detection/YoloPipeline.cs` / `YoloPostProcessor.cs` / `DetectionResult.cs` | YOLO detection |
| `Segmentation/SamPipeline.cs` / `SamMaskDecoder.cs` | SAM segmentation |
| `FaceDetection/FaceDetector.cs` / `LandmarkExtractor.cs` | Face detection and landmarks |

---

## src/SharpInference.Server/

| File | Description |
|---|---|
| `Setup/SharpInferenceServiceExtensions.cs` / `SharpInferenceServerOptions.cs` | DI registration, server options |
| `Endpoints/` | ImageGeneration, AudioTranscription, Vision, ModelManagement |
| `Streaming/SseProgressStream.cs` / `AudioChunkStream.cs` | SSE progress, audio chunk streaming |
| `Queue/InferenceQueue.cs` / `InferenceQueueEntry.cs` | FIFO inference queue |
| `Auth/ApiKeyMiddleware.cs` | Optional API key validation |

---

## tests/

| Project | Tests |
|---|---|
| `SharpInference.Core.Tests` | TensorTests, TensorShapeTests, NativeBufferTests |
| `SharpInference.ModelHandler.Tests` | SafeTensorsLoaderTests, GgufLoaderTests |
| `SharpInference.Cpu.Tests` | MatMulKernelTests, Conv2DKernelTests, NormKernelTests, AttentionKernelTests |
| `SharpInference.Diffusion.Tests` | SchedulerTests, ClipTokenizerTests, PipelineIntegrationTests |
| `SharpInference.Audio.Tests` | StftTests, MelSpectrogramTests, WhisperIntegrationTests |
| `SharpInference.Server.Tests` | ImageApiTests |

---

## samples/

| Sample | Description |
|---|---|
| `BasicImageGeneration` | Console app: text → image |
| `StreamingServer` | ASP.NET with SSE image progress |
| `VoiceAssistant` | Whisper STT + dotLLM LLM + Kokoro TTS loop |
| `SwarmUIExtension` | Full SwarmUI backend extension example |

---

## benchmarks/

| File | Description |
|---|---|
| `SharpInference.Benchmarks/KernelBenchmarks.cs` / `PipelineBenchmarks.cs` / `AudioBenchmarks.cs` | BenchmarkDotNet suites |
| `scripts/bench_compare.py` / `bench_trend.py` | Comparison and trend analysis scripts |

---

## native/

| Directory | Contents |
|---|---|
| `cuda/kernels/` | CUDA C++ source for PTX generation: conv2d, group_norm, sdpa, fft, silu, gelu, rope_2d, timestep_embed, conv2d_bias_silu_fused, dequant_q8/q4k |
| `cuda/build.sh` | `nvcc -ptx -arch=compute_80` |
| `vulkan/shaders/` | GLSL compute shaders for SPIR-V: same kernels + `matmul_tiled.comp.glsl` |
| `vulkan/build.sh` | `glslangValidator --target-env vulkan1.2` |
