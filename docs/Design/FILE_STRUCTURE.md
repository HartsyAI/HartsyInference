# File Structure

> Back to [Core Design](CORE_DESIGN.md)

## Full Project Layout

```
HartsyInference/
├── HartsyInference.slnx / Directory.Build.props / Directory.Packages.props
├── CLAUDE.md / GEMINI.md / README.md / LICENSE / CONTRIBUTING.md
├── src/
│   ├── HartsyInference.Core / ModelHandler / Tokenizers / Phonemizer / Cpu / Cuda / Vulkan
│   ├── HartsyInference.LLM / Diffusion / Audio / Vision / Video / Interactive / ThreeD
│   ├── HartsyInference.Engine       (service layer — the single source of truth; see CORE_DESIGN.md)
│   ├── HartsyInference.Meta        (dependencies-only meta-package)
│   ├── HartsyInference.Cli         (developer / verification CLI)
│   └── HartsyInference.API         (thin HTTP adapter over Engine — secondary to SwarmUI, see below)
├── tests/ / samples/ / benchmarks/ / docs/ / native/
```

> The recommended way to run the engine is the external [SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend) extension; it lives in its own repo, not here.

---

## src/HartsyInference.Core/

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
| `Pipelines/` | `IAudioPipeline`, `IVisionPipeline`, `IPipelineRequest` (aspirational scaffolds for not-yet-built modalities). Diffusion has its own base class in `HartsyInference.Diffusion` — see `Pipelines/DiffusionPipelineBase.cs` there. |
| `Schedulers/IScheduler.cs` | `SetTimesteps`, `Step`, `AddNoise` |
| `Models/IModel.cs` / `ModelConfig.cs` / `ModelFormat.cs` | Load/Forward/GetConfig, architecture params, format enum |
| `Memory/NativeBuffer.cs` / `MmapHandle.cs` | `NativeMemory.AlignedAlloc` wrapper, `MemoryMappedFile` lifetime manager |
| `Logging/Logs.cs` | Static Logs class |
| `Exceptions/` | `HartsyInferenceException`, `OutOfVramException`, `UnsupportedModelException` |

---

## src/HartsyInference.ModelAssets/

| File | Description |
|---|---|
| `SafeTensors/SafeTensorsLoader.cs` | mmap + JSON header → TensorView dict |
| `SafeTensors/SafeTensorsWriter.cs` | Save tensors to safetensors |
| `SafeTensors/SafeTensorsShardLoader.cs` | Multi-file sharded models |
| `PyTorch/PytorchPickleLoader.cs` / `PickleMachine.cs` | Torch `.pt` reader (ZIP + safe-subset pickle VM, no code exec) → Tensor dict — for `.pt`-only models (GameCraft, Cosmos) |
| `Gguf/GgufLoader.cs` | GGUF header + metadata + tensor mmap |
| `Gguf/GgufDequantizer.cs` | Dequantize Q4_0/Q8_0/Q4_K_M to F16/F32 |
| `Gguf/GgufMetadata.cs` | Typed metadata key-value access |
| `Registry/ModelRegistry.cs` | In-memory model cache |
| `Registry/ModelCacheStore.cs` | Disk cache at `~/.hartsyinference/models/` |
| `Registry/ModelInfo.cs` | Model metadata |
| `HuggingFace/HuggingFaceClient.cs` | Search, pull, resolve GGUF variant |
| `CheckpointConverters/` | SD1.5 / SDXL / Flux / SD3 `.ckpt` → diffusers format |
| `Convert/CheckpointConverter.cs` | `.ckpt` → `.safetensors` (planned) |
| `Convert/QuantizeConverter.cs` | FP32 → FP16, FP16 → Q8_0 (planned) |

---

## src/HartsyInference.ModelAssets.Tokenizers/

| File | Description |
|---|---|
| `ClipTokenizer.cs` | BPE tokenizer, 77-token limit |
| `T5Tokenizer.cs` | SentencePiece, 4096-token context |
| `WhisperTokenizer.cs` | Multilingual BPE + special tokens |
| `TokenizerCache.cs` | Reuse tokenizers across pipeline instances |

---

## src/HartsyInference.Audio.Phonemizer/

Pure-C# G2P / IPA phonemization (an espeak-ng-style port) for phoneme-input TTS models. Depends on Core only.

| File | Description |
|---|---|
| `IPhonemizer.cs` / `PhonemeIdMap.cs` | Phonemizer interface + phoneme-to-id mapping |
| `Espeak/EspeakTranslator.cs` / `EspeakPhonemizer.cs` | Text → phoneme translation, top-level phonemizer |
| `Espeak/EspeakPhonemeInterpreter.cs` / `EspeakProgram.cs` / `EspeakPhonemeRenderer.cs` | Phoneme-program bytecode VM (allophones + IPA), rule interpreter, renderer |
| `Espeak/EspeakDictFile.cs` / `EspeakWordLookup.cs` / `EspeakRuleCodes.cs` / `EspeakStress.cs` | Compiled-dictionary parser, per-word lookup, letter/rule codes, stress rules |
| `Resources/ipa_phoneme_map*.tsv` | IPA phoneme maps (base + en / en-us) embedded as resources |

---

## src/HartsyInference.LLM/

Native LLM text generation. One config-driven generic transformer serves decoder LLMs, text encoders, and
seq2seq. Depends on Core + ModelHandler + Tokenizers. See [LLM_LANGUAGE_PACKAGE.md](LLM_LANGUAGE_PACKAGE.md).

| File | Description |
|---|---|
| `Transformer/GenericTransformer.cs` / `TransformerConfig.cs` | Config-driven block stack (Qwen2/Qwen3/Llama/Mistral) + config record with presets |
| `Transformer/KvCache.cs` / `FixedKvCache.cs` / `IKvCache.cs` / `KvCaches.cs` | Device-resident KV cache variants |
| `Transformer/MoeFeedForward.cs` / `GgufConfigFactory.cs` | MoE FFN routing; GGUF metadata → config |
| `Generation/TextGenerationPipeline.cs` / `GgufLanguageModel.cs` / `GenerationRequest.cs` / `GenerationResult.cs` / `ContinuousBatchScheduler.cs` | Prefill→decode→sample loop, GGUF model wrapper, request/result types, batching |
| `Sampling/SamplerChain.cs` / `ISamplerStep.cs` / `TemperatureStep.cs` / `TopKStep.cs` / `TopPStep.cs` / `MinPStep.cs` / `RepetitionPenaltyStep.cs` | Composable sampler chain |
| `ChatTemplates/ChatTemplateRegistry.cs` / `ChatMlTemplate.cs` / `JinjaChatTemplate.cs` / `JinjaEngine.cs` | Chat templating incl. a Jinja engine for GGUF-embedded templates |
| `Embeddings/BertEmbeddingModel.cs` / `DecoderEmbeddingModel.cs` | Text-embedding heads on the generic core |
| `Seq2Seq/T5Model.cs` | Encoder-decoder (T5) on the shared core |
| `Ssm/MambaModel.cs` / `Mamba2Model.cs` / `RwkvModel.cs` / `Rwkv7Model.cs` | State-space / RWKV sequence models |
| `Multimodal/MllamaGenerator.cs` / `Qwen25VlEncoder.cs` / `SiglipVlmEncoder.cs` / `MultimodalGenerator.cs` | VLM generators + vision encoders (Llama-3.2-Vision, Qwen2.5-VL) |

---

## src/HartsyInference.Cpu/

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

## src/HartsyInference.Cuda/

| File | Description |
|---|---|
| `CudaBackend.cs` | IBackend implementation — routes to PTX + cuBLAS. Auto-transfer pattern with GPU weight cache. Provides `PreloadWeights()`, `FreePreloadedWeights()`, `GetGpuCacheStats()`. |
| `CudaDriverApi.cs` | P/Invoke surface (~40 functions): cuInit, cuDeviceGet, cuCtxCreate, cuModuleLoadData, cuLaunchKernel, cuMemAlloc/Free, cuMemcpyHtoD/DtoH, cuStreamCreate/Synchronize |
| `CublasApi.cs` | cuBLAS P/Invoke — SGEMM for Linear/Conv2D, handle bound to stream |
| `CudaMemory.cs` | GPU memory allocation (Allocate/Free/CopyHtoD/CopyDtoH) wrapping Driver API |
| `CudaStream.cs` | Stream lifecycle (blocking mode — non-blocking causes race conditions with synchronous transfers) |
| `CudaKernels.cs` | Kernel function handles as `nint` fields, launch wrappers (LaunchIm2Col, LaunchBiasAdd, LaunchGroupNorm, etc.) |
| `CudaModule.cs` | PTX module loading + function handle extraction |
| `GpuTransferHelper.cs` | Device memory management + GPU weight cache (`Dictionary<Tensor, ulong>` with reference equality). Cache-aware `CopyToDevice`/`FreeDevice`. `PreloadWeight`/`FreeAllCached`. |
| `Ptx/elementwise_f32.ptx` | Add, Scale, SiLU, GELU, Sigmoid, Clamp kernels |
| `Ptx/spatial_f32.ptx` | Im2Col (64-bit indexing for 1024+), UpsampleNearest2D, Col2BiasAdd |
| `Ptx/norm_f32.ptx` | GroupNorm, LayerNorm (3-pass: mean → variance → normalize+affine) |
| `Ptx/sdpa_f32.ptx` | Scaled dot-product attention with per-row softmax (shared memory reduction) |

---

## src/HartsyInference.Vulkan/

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

## src/HartsyInference.Diffusion/

| File | Description |
|---|---|
| `Pipelines/DiffusionPipelineBase.cs` | Abstract base for every pipeline — holds the `IBackend`, idempotent disposal flag, `ThrowIfDisposed()`, `DisposeCore()` hook. Class-level docs explain why no `DenoiseLoopRunner` (per-model loop variation resists abstraction). |
| `Pipelines/StableDiffusion15Pipeline.cs` / `SdxlPipeline.cs` / `SdxlInpaintPipeline.cs` / `SdxlRefinerPipeline.cs` / `Sd3Pipeline.cs` / `FluxPipeline.cs` / `Flux2Pipeline.cs` / `ChromaPipeline.cs` / `AuraFlowPipeline.cs` / `FLitePipeline.cs` / `HiDreamPipeline.cs` / `HunyuanImagePipeline.cs` / `Kandinsky5Pipeline.cs` / `Lumina2Pipeline.cs` / `QwenImagePipeline.cs` / `ErnieImagePipeline.cs` / `ZImagePipeline.cs` / `AnimaPipeline.cs` / `OmniGen2Pipeline.cs` | 19 pipeline implementations, all inherit `DiffusionPipelineBase`. Take pre-loaded components via constructor; expose `GenerateFromTokens` / `GenerateFromEmbeddings` / `InpaintFromTokens` / `RefineFromTokens` (signature varies per pipeline). Callbacks via `Action<GenerationProgress>?` — NOT `IAsyncEnumerable`. |
| `Pipelines/PipelineFactory.cs` | Scaffolding only — `LoadAuto` throws `NotImplementedException` with a list of 5 unresolved design questions (model-type detection, layout discovery, tokenizer ownership, quality profile, caching). A real factory needs a design conversation; callers currently construct pipelines directly. |
| `Models/TextEncoders/ClipTextEncoder.cs` / `T5TextEncoder.cs` / `LlamaStyleEncoder.cs` | CLIP-L/G (penultimate-layer or pooled), T5 (Pile-T5, T5-XXL), Llama-style (Qwen-3 / Qwen2.5-VL / Mistral) text encoders |
| `Models/Denoisers/` | One transformer/UNet config + class per model family (UNet for SD1.5/SDXL; per-model DiT/MMDiT for newer arches) |
| `Models/Denoisers/UNetBlocks/` | ResNetBlock, CrossAttentionBlock, DownSampleBlock, UpSampleBlock |
| `Models/Denoisers/DiTBlocks/` | MmDiTBlock, FluxSingleStreamBlock, FluxDoubleStreamBlock, OmniGen2Block, AnimaLlmAdapter, etc. |
| `Models/Vae/VaeEncoder.cs` / `VaeDecoder.cs` / `QwenImage/QwenImageVaeDecoder.cs` | VAE encode/decode (tiled decode is on `VaeDecoder.DecodeTiled`). Qwen-Image VAE is separate (3D causal autoencoder collapsed to 2D for image mode). |
| `Schedulers/` | EulerDiscrete, DPM++ 2M, DDIM, LCM, FlowMatchEulerDiscrete (static + dynamic shift). `SchedulerFactory.Create(name)` centralizes the user-selectable-scheduler switch. |
| `Adapters/ControlNet.cs` / `LoraLoader.cs` / `IpAdapter*.cs` | ControlNet residual injection, LoRA loading, IP-Adapter conditioning |
| `Requests/TextToImageRequest.cs` / `ImageToImageRequest.cs` / `SdxlRefinerRequest.cs` | Request types. Inpaint is enabled by setting `ImageToImageRequest.Mask` (no separate request type). |
| `Utilities/ImagePostProcessor.cs` / `SeedGenerator.cs` / `LatentPreview.cs` / `MaskBlendUtilities.cs` / `TaesdDecoder.cs` / `LatentArchitecture.cs` | Image I/O, RNG/noise, preview decode, mask blending, TAESD preview |
| `Utilities/CfgHelper.cs` | Shared CFG helpers used by every pipeline that runs uncond+cond passes: `SliceBatchElement`, `SliceBatchElement1D`, `ApplyCfg`, `ConcatLastDim`. Z-Image's non-standard formula stays local in `ZImagePipeline`. |
| `Utilities/DtypeCastHelper.cs` | `EnsureDtype` / `EnsureF32` — single source of truth for F16↔F32↔BF16 activation casts. Replaces ~20 inline `new Tensor(shape, dt); backend.CastTo*(...)` sites in pipelines + UNet/ControlNet/VaeDecoder/FluxSingleStreamBlock. |
| `Utilities/Img2ImgSetup.cs` | `Img2ImgSetup.Prepare(request, h, w, steps)` returns a `Plan` with `StartStep`, `MaskPixel`, `PassThrough`. Centralizes source-shape / mask validation + strength-clamp + start-step computation. Flux.2 keeps its own validation (16-rounded dimensions). |

---

## src/HartsyInference.Audio/

| File | Description |
|---|---|
| `Stt/WhisperPipeline.cs` / `WhisperEncoder.cs` / `WhisperDecoder.cs` / `WhisperStreamingPipeline.cs` / `WhisperOptions.cs` | Whisper STT pipeline, streaming, options |
| `Tts/KokoroPipeline.cs` / `KokoroPhonemeEncoder.cs` / `HiFiGanVocoder.cs` / `VocosVocoder.cs` / `TtsOptions.cs` | Kokoro TTS pipeline, phoneme encoder, vocoders |
| `VoiceConversion/RvcPipeline.cs` / `F0Extractor.cs` | Voice conversion |
| `Preprocessing/AudioPreprocessor.cs` / `StftProcessor.cs` / `MelSpectrogramProcessor.cs` | Audio preprocessing |

---

## src/HartsyInference.Vision/

| File | Description |
|---|---|
| `Clip/ClipImageEncoder.cs` / `ClipScorer.cs` | CLIP image encoder, text-image similarity |
| `Embeddings/ImageEmbeddingPipeline.cs` / `TextEmbeddingPipeline.cs` | Image and text embedding pipelines |
| `Detection/YoloPipeline.cs` / `YoloPostProcessor.cs` / `DetectionResult.cs` | YOLO detection |
| `Segmentation/SamPipeline.cs` / `SamMaskDecoder.cs` | SAM segmentation |
| `FaceDetection/FaceDetector.cs` / `LandmarkExtractor.cs` | Face detection and landmarks |
| `Siglip/SiglipVisionEncoder.cs` / `Dinov2/Dinov2VisionEncoder.cs` | SigLIP + DINOv2 ViT towers (DINOv2 LayerScale optional → also serves DINOv1; conditioning for 3D models) |

---

## src/HartsyInference.World/  (Phase 10 — world models)

| File | Description |
|---|---|
| `Sessions/IInteractiveSession.cs` | Bidirectional streaming session — `SubmitActionAsync` / `ReadFramesAsync` |
| `Sessions/BackgroundComputeSession.cs` | Default impl — dedicated compute thread + CUDA stream, bounded action/frame channels with backpressure |
| `Sessions/InteractiveSessionStats.cs` | p50/p99 step latency, dropped frames, queue depths |
| `ActionEncoders/IActionEncoder.cs` | Multi-stream encoder interface (each model emits typed streams: PerBlockSelfAttn / PerBlockCrossAttn / PluckerMap / TimestepAddon) |
| `ActionEncoders/KeyboardOneHotEncoder.cs` / `MouseDeltaEncoder.cs` | Reusable building blocks |
| `ActionEncoders/MatrixGameUniversalActionEncoder.cs` / `Gta` / `TempleRun` / `MatrixGame3ActionEncoder.cs` | Per-model action encoders |
| `ActionEncoders/MinecraftVptActionEncoder.cs` | Oasis 25-dim VPT action vector |
| `ActionEncoders/GameCraftActionEncoder.cs` | GameCraft `(w/a/s/d, speed)` → 33-pose camera trajectory → 6-ch Plücker |
| `Camera/SE3Math.cs` / `PluckerEmbedding.cs` | Shared SE(3) inverse, SLERP, integrate-actions-to-poses; Plücker ray-coord computation |
| `Memory/FrameHistoryBuffer.cs` | Rolling buffer of `(latent, camera_pose, frame_index)` |
| `Memory/FrustumOverlapSelector.cs` / `MatrixGame3MemoryRetrieval.cs` | Camera-FOV memory selection (Matrix-Game 3.0 specific) |
| `Models/Denoisers/DiTBlocks/MatrixGame2ActionModule.cs` / `MatrixGame3ActionModule.cs` | Per-block dual-stream (mouse=self-attn, keyboard=cross-attn) modules |
| `Models/GameCraftCameraNet.cs` | GameCraft action-to-token CameraNet (PixelUnshuffle + Convs + PatchEmbed + scale) |
| `Pipelines/MatrixGame2Pipeline.cs` / `MatrixGame3StandardPipeline.cs` / `MatrixGame3InteractivePipeline.cs` | Skywork Matrix-Game pipelines |
| `Pipelines/OasisPipeline.cs` | Decart/Etched Oasis-500m AR frame-by-frame Minecraft world model |
| `Pipelines/GameCraftLatentBuilder.cs` | 33-ch composite `[noisy16 + history16 + mask1]` |
| `Pipelines/HunyuanGameCraftPipeline.cs` / `Sessions/GameCraftFrameStepper.cs` | Tencent GameCraft pipeline + live `IFrameStepper`. No license gate — uses the HunyuanVideo MM-DiT (in Diffusion) + reused HunyuanVideo VAE / encoders / scheduler |

---

## src/HartsyInference.ThreeD/  (Phase 11 — 3D asset generation)

| File | Description |
|---|---|
| `Geometry/Mesh.cs` / `ScalarField3D.cs` / `PointCloud.cs` / `GaussianSplatCloud.cs` / `Triplane.cs` | Representation-agnostic geometry types |
| `Geometry/Ops/MarchingCubes.cs` / `MeshOps.cs` / `GridSampler.cs` / `SurfaceSampler.cs` | Mesh extraction (Bourke tables), normals, trilinear/triplane sampling, FPS |
| `Io/GlbWriter.cs` / `ObjWriter.cs` / `PlyWriter.cs` | glTF 2.0 (primary), OBJ, PLY (mesh + 3DGS) export |
| `Pipelines/ThreeDPipelineBase.cs` / `Requests/ImageTo3DRequest.cs` / `ThreeDResult.cs` | Pipeline scaffolding |
| `Models/Hunyuan3D/*` | Hunyuan3D-2 image→mesh: VecSet flow-match DiT + ShapeVAE occupancy decoder + converter |
| `Models/TripoSr/*` / `Pipelines/TripoSrPipeline.cs` | TripoSR image→mesh: triplane transformer + NeRF MLP decoder (feed-forward) |

## src/HartsyInference.Cli/

Developer / verification CLI (`Program.cs`) that references most modality packages (Diffusion, LLM, Audio,
Vision, Video, Interactive) plus the backends. Used for local runs and parity checks, not a shipped product.

---

## src/HartsyInference.API/  (thin HTTP adapter over Engine — secondary to SwarmUI)

ASP.NET Core Minimal API process. Live and supported, not a sample: constructs one `InferenceEngine`, gates every
call through `HartsyInference.Engine.InferenceQueue`, and maps health/admin/settings plus native + OpenAI-compat
generation endpoints. `IsPackable=false` (it's a runnable app, not a NuGet library) — see `docs/Agents/API.md` for
the full endpoint catalog and design rationale.

| File | Description |
|---|---|
| `Program.cs` | Entry point — `AddHartsyInference()` + `MapHartsyInferenceEndpoints()` |
| `HartsyInferenceServiceExtensions.cs` | DI registration (`IInferenceEngine`, two `InferenceQueue` gates — fast + keyed long-running, `WorldSessionRegistry`) and top-level endpoint wiring |
| `HartsyInferenceServerOptions.cs` | Backend selector, concurrency/queue-depth limits (fast + long-running), API key, model cache dir |
| `WorldSessionRegistry.cs` | Tracks open interactive-world sessions by id; idle-timeout eviction |
| `OpenAiDtos.cs` | OpenAI-shaped request/response DTOs for the compat routes |
| `Endpoints/HealthEndpoints.cs` | `/health`, `/ready`, `/version` |
| `Endpoints/SettingsEndpoints.cs` | `/settings` (read-only, API key redacted) |
| `Endpoints/AdminEndpoints.cs` | `/admin/{catalog,models,cache,models/pull,memory/free,backend,queue}` |
| `Endpoints/ImageEndpoints.cs` / `TextEndpoints.cs` / `AudioEndpoints.cs` / `VisionEndpoints.cs` / `MeshEndpoints.cs` / `VideoEndpoints.cs` / `WorldEndpoints.cs` | Native `/v1/native/*` routes, one file per modality |
| `Endpoints/CompatEndpoints.cs` | `/v1/chat/completions`, `/v1/images/generations`(+`/stream`) — thin wrappers calling the same handlers as the native routes |
| `Endpoints/SseHelpers.cs` | Shared SSE plumbing: queue-gated streaming + JSON-options-consistent event formatting |
| `Endpoints/GenerationErrors.cs` | Maps Engine exceptions (`FileNotFoundException`, `NotSupportedException`, `HartsyInferenceException`, `QueueFullException`) to HTTP status codes |
| `Endpoints/*Dtos.cs` | Envelope request DTOs (`{model, modelPath, request}`) per modality |

---

## tests/

One test project per package plus shared fixtures and Python references.

| Project | Scope |
|---|---|
| `HartsyInference.Core.Tests` | Tensor, TensorShape, NativeBuffer, finalizer cleanup |
| `HartsyInference.ModelAssets.Tests` | Safetensors, GGUF, PyTorch-pickle loaders |
| `HartsyInference.Cpu.Tests` | MatMul, Conv2D, Norm, Attention kernels |
| `HartsyInference.Cuda.Tests` | CUDA kernels, multi-backend isolation |
| `HartsyInference.Vulkan.Tests` | SPIR-V kernels vs CPU/CUDA reference |
| `HartsyInference.ModelAssets.Tokenizers.Tests` / `HartsyInference.Audio.Phonemizer.Tests` | Tokenizer + G2P parity |
| `HartsyInference.LLM.Tests` | Generic transformer, sampler chain, chat templates, decode parity |
| `HartsyInference.Diffusion.Tests` | Schedulers, tokenizer, pipeline integration |
| `HartsyInference.Audio.Tests` / `HartsyInference.Vision.Tests` | STT/TTS + CLIP/YOLO/SAM |
| `HartsyInference.Video.Tests` / `HartsyInference.World.Tests` / `HartsyInference.ThreeD.Tests` | Video, world-model, and 3D pipelines |
| `HartsyInference.API.Tests` | `WebApplicationFactory`-based routing/validation/error-mapping tests for every `HartsyInference.API` route, plus direct unit tests for `WorldSessionRegistry` |
| `HartsyInference.Tests.Common` / `python-reference/` | Shared test helpers + Python golden-reference scripts |

---

## samples/

Per-modality sample CLI apps plus offline conversion tools (developer / verification tools).

| Sample | Description |
|---|---|
| `HartsyInference.TextGen.Cli` | Console app: prompt → LLM text generation |
| `HartsyInference.Speech.Cli` / `HartsyInference.Whisper.Cli` / `HartsyInference.Music.Cli` / `HartsyInference.Audio.Codec.Cli` | TTS, Whisper STT, music generation, audio codec |
| `HartsyInference.Vision.Cli` / `HartsyInference.Video.Cli` / `HartsyInference.ThreeD.Cli` | Vision, video, and image→3D runs |
| `ConvertSafetensorsToGguf` / `FuseWeightNorm` | Offline checkpoint conversion utilities |

---

## benchmarks/

| File | Description |
|---|---|
| `HartsyInference.Benchmarks/KernelBenchmarks.cs` / `PipelineBenchmarks.cs` / `AudioBenchmarks.cs` | BenchmarkDotNet suites |
| `scripts/bench_compare.py` / `bench_trend.py` | Comparison and trend analysis scripts |

---

## native/

| Directory | Contents |
|---|---|
| `cuda/kernels/` | CUDA C++ source for PTX generation: conv2d, group_norm, sdpa, fft, silu, gelu, rope_2d, timestep_embed, conv2d_bias_silu_fused, dequant_q8/q4k |
| `cuda/build.sh` | `nvcc -ptx -arch=compute_80` |
| `vulkan/shaders/` | GLSL compute shaders for SPIR-V: same kernels + `matmul_tiled.comp.glsl` |
| `vulkan/build.sh` | `glslangValidator --target-env vulkan1.2` |
