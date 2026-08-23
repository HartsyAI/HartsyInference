# Dead/duplicate/redundant code candidates — for review, not yet acted on

Collected during the 2026-08-22 comment/summary cleanup pass (see `docs/CODE_STYLE.md` §Comments). Each
entry is a **candidate**, not a confirmed defect — verify call sites before removing anything, since a
"duplicate-looking" helper can differ in a load-bearing edge case (dtype, device, batch=1 assumption, etc.).
Nothing in this list has been changed. Organized by package in the order they were reviewed.

## HartsyInference.Core

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Rope/RopeScaling.cs` | `YarnLogMultiplier` | Parsed from GGUF and stored, but never read — including by `RopeFrequencyBuilder.InferMscale`, which its own doc says should consume it for DeepSeek-V2/V3 MLA mscale. Looks like a real wiring gap, not just cruft. | High |
| `Backends/DeviceKind.cs` | `Cuda0`, `Vulkan0` statics, `IsVulkan` | Zero call sites repo-wide. Public API on a foundational type — an out-of-repo consumer (SwarmUI extension) isn't ruled out. | Medium |
| `Backends/MixContract.cs` (`ValidateGeometry`) vs `Backends/SplitContract.cs` (`ValidateTensorGeometry`) | Near-identical private geometry validators | Duplicate — could hoist to one shared helper. | Medium |
| `Backends/IBackend.cs` — `CfgEulerStep`, `CfgRenormEulerStep`, `CfgNormalizedEulerStep` | Each repeats an identical ~10-line dtype/shape/alias/finite validation preamble | Repo already has precedent for this exact extraction (`MixContract.ValidateAffineMix`/`ValidateMaskedAffineMix`). | High |
| `Backends/IBackend.cs` — `OasisRopeInterleaved`, `WanRopeInterleaved`, `WanRopeInterleavedPerHead`, `Mg3RopeBatched`, `ApplyRopeInterleaved` | Each reimplements the same interleaved rotate-pair math | A local precedent already extracts this once (`RotateInterleaved` in `Ltx25NaRope3dReference`) — layouts differ somewhat so verify before hoisting. | Medium |
| `Tensors/TensorPool.cs` (whole class) | Zero instantiations/call sites in src/ or tests/ | Referenced only in docs as the intended hot-path pattern, never actually used. Public — external consumers not ruled out. | Medium |
| `Pipelines/ISttPipeline.cs`, `ITtsPipeline.cs`, `IPipelineRequest.cs` | No implementers, no callers anywhere | | Medium |
| `Tensors/Int8ConvRotCodec.cs` (`ToBf16`) + `Tensors/Nvfp4ResidentCodec.cs` (`ToBf16`) + inline copy in `Tensor.cs`'s `ConvertRange` F32→BF16 case | Byte-for-byte identical bit-truncation helper duplicated 3x | | High |
| `Codecs/Fsq.cs` — `Quantize`, `CodesToIndices`, `Dequantize` | Each reimplements the mixed-radix place-value loop inline instead of calling the existing `Fsq.Basis(levels, basis)` helper (which does have its own test call site, so it's not itself dead) | | High |
| `MemoryManagement/BlockStreamingScope.cs` + `VramPlanner.cs` (`Mb` formatting helper) | Verbatim-identical `private static string Mb(long bytes)` | Also duplicated outside Core in `HartsyInference.Vulkan/VulkanGpuTransferHelper.cs` and `HartsyInference.Engine/Recipes/Video/MiniMaxH3RecipePipeline.cs` — a shared formatting helper is the right home. | High |
| `MemoryManagement/BlockStreamingScope.cs` — `EnumerateStreamedWeights()`, `BlockBytes` | No call sites/readers found (sibling `EnumerateResidentWeights()` IS used) | Public API, external-consumer caveat applies. | Medium |

## HartsyInference.Cpu / API / World

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Cpu/Threading/NumaAffinity.cs` | `TryPinToCore` | `try` block unconditionally `return false;` before any work — the `catch` and the `IsWindows()` branch are unreachable; an explicitly labeled stub but the branch is dead now. | High |
| `Cpu/Threading/ComputeThreadPool.cs` + `NumaAffinity.cs` | whole classes | Zero call sites anywhere in repo. Public, so unwired future API isn't ruled out. | Medium |
| `Cpu/Kernels/AudioKernels.cs` | `Fft` vs `Stft` | Bit-reversal + radix-2 butterfly loop (~30 lines) duplicated verbatim (whole-buffer vs per-frame); both already share `ReverseBits` — hoistable into one `FftInPlaceComplex(float*, int)`. | High |
| `World/Pipelines/MatrixGame2Pipeline.cs` | `AssembleInput` local `src` | A ternary is computed then discarded via `_ = src;` — the actual copy loop recomputes its own pointer independently. | High |
| `MatrixGame2Pipeline.SliceFrame` vs `MatrixGame3Pipeline.SliceFrame` | Duplicate `Buffer.MemoryCopy` loop over channels for one frame index; differs only in how channel count is obtained. | High |
| `MatrixGame2Pipeline.ConcatFrames` vs `MatrixGame3Pipeline.ConcatFrames` | MG3's version is a strict generalization of MG2's T=1 special case. | Medium-high |
| `MatrixGame2Pipeline.BuildActionRows` vs `MatrixGame3Pipeline.ActionRows` | Same clamp-and-copy-row-into-tensor core loop; MG2 adds null-handling + VAE-temporal-rate scaling on top. | Medium |
| `DiamondWorldPipeline.DecodeFrame` / `OasisPipeline.RgbToBytes` vs `HartsyInference.Video/VideoRgbFrames.ExtractFrame` | Both pipelines reimplement the CHW→interleaved-RGB24 denormalization already centralized in `VideoRgbFrames.ExtractFrame`, which both pipelines already call elsewhere for multi-frame output. | High (decode); Medium (encode direction — the `/127.5f - 1f` pattern recurs repo-wide, may be an established convention not a bug) |
| `API/Endpoints/*.cs` | `Model`/`ModelPath` property pair repeated verbatim across ~14 native request DTOs | Mirrors the existing shared `Save`/`OutputDir` extraction pattern in `NativeArtifactRequest`. Flat DTOs sometimes flatten deliberately for serialization — a deliberate call, not an obvious fix. | Medium |

**Also found (not dead code, a small misplaced-comment bug):** `API/HartsyInferenceServiceExtensions.cs` — a "Production-hardening surface..." comment sits above the `WyomingEnabled` block, but its content actually describes the `ApiKeyStore`/`UsageTracker`/`ApiMetrics` registrations further down. Looks displaced by an earlier edit.

## HartsyInference.Engine (Features/Requests/Vision/Placement/HuggingFace/Dispatch)

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Requests/VideoRequest.cs:37` | `VideoModel` | Zero references repo-wide; video model selection actually goes through `ModelSpec`/`InferenceEngine.GetOrConstructVideoRecipe`, not the request object. External consumers (SwarmUI extension) unchecked. | Medium-high |
| `Requests/VideoRequest.cs:52` | `VideoFormat` | Same pattern — zero references, `VideoService.cs` never reads a format field. | Medium-high |
| `Requests/Composition/Regional.cs:10` | `SortOrder` | Zero references, but already tracked — `ROADMAP.md:829` documents this as `SegmentSortOrder`, deliberately deferred. Not a new finding. | N/A (tracked) |
| `Vision/YoloObjectDetector.cs`, `RtDetrObjectDetector.cs`, `GroundingDinoObjectDetector.cs` | `_cache`/`_lock`/`_boundBackend`/`GetOrLoad`/`DisposeAll`/`Dispose` skeleton | Near-byte-identical backend-bound pipeline-cache pattern in all three; a generic `BackendBoundCache<TPipeline>` helper would remove ~40 duplicated lines per file. | High |
| `Requests/ImageRequest.cs` vs `Requests/VideoRequest.cs` | `Prompt`, `NegativePrompt`, `Width`, `Height`, `Steps`, `CfgScale`, `Sampler`, `Seed`, `Components`, `Loras`, `Extra` | 11 fields duplicated verbatim across the two records. `ImageRequest`'s own class doc states the flat-contract (no inheritance) design is deliberate, so this may be an intentional tradeoff — flag, don't assume. | Medium |

(Also checked: no dead private methods/fields found in `Features/`, `Vision/`, `Placement/`, `HuggingFace/`, `Dispatch/`. A `ComponentOverrides.T5Xxl` lead turned out to be part of the already-tracked `E-IMG-4/5` deferred-components gap.)

## HartsyInference.ThreeD

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Models/Trellis/SlatGaussianDecoder.cs:9-10` | `Heads`, `OutCh`, `Scale`, `HeadDim` | Unused — real values live in sibling class `GsBlock`. | High |
| `Models/Trellis/SparseStructureFlow.cs:9` | `Heads = 16` | Unused — real count is in `SsFlowBlock.H`. | High |
| `SlatFlowModel.AbsolutePositionEmbed` vs `SlatGaussianDecoder.SlatFlowModelApe` | Identical sin/cos position-embedding formula, differ only by hardcoded channel count (1024 vs 768); the latter's name betrays copy-paste origin. | High |
| `Hunyuan3DDit.F32`, `Hunyuan3DShapeVae.F32`, `SparseStructureDecoder.F`, `SparseStructureFlow.F`, + inline in `SparseOps.cs` | Same "cast to F32 if not already" ternary reimplemented 5x; a repo `DtypeCastHelper.EnsureF32` exists but isn't a drop-in (different disposal/backend contract). | Medium |
| `Models/Trellis/SparseOps.cs` — `PermuteConvWeight`/`SubmanifoldConv3d` | Zero production call sites, superseded by `SubmanifoldConv3dSparse`; only used by a parity test fixture (likely intentional). | Medium |
| `Models/Hunyuan3D/Hunyuan3DDebugDump.cs` | Whole class has zero call sites, unlike sibling models' `*DebugDump` classes which ARE wired in. | Low (may be intentionally unwired scaffolding) |

## HartsyInference.Video

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Models/Cosmos/CosmosArBlock.cs:203` | `Project(...)`'s `lowVram` param | Never read; all 9 call sites pass `false`. May be deliberate signature-mirroring. | High (inert) |
| `Tokenizers/CosmosDvTokenizer.cs:29-30,165,211-217` | `_encoderConvs`/`_decoderConvs` | Never assigned non-null; unreachable branches — documented as integration-pending scaffolding. | High on facts |
| `WanAnimatePipeline.cs:317 HostCopy`, `WanAnimate2Pipeline.cs:367 HostCopy` (byte-identical), `WanVideoPipeline.cs:876 CloneLatents` (F32-hardcoded), `CosmosDvModule.cs:204 Clone` | 4-way duplicate tensor-copy primitive; `Tensor.To(tensor.Device)` already exists and also preserves `Fp8ScaleFactor`/`QuantInfo`, which these don't. | High |
| `LtxVideoPipeline.cs:329 UnpackLatents` vs `LtxVideo2Pipeline.cs:966 UnpackVideoLatents` | Byte-identical. | High |
| `LtxVideoPipeline.cs:291 ExtractMiddleFrame` vs `LtxVideo2Pipeline.cs:1029 ExtractMiddleFrame` | Byte-identical (HunyuanVideoPipeline's same-named method is genuinely different, not a duplicate). | High |
| `LtxVideoPipeline.cs:311 PackLatents` vs `LtxVideo2Pipeline.cs:1010 PackVideoLatents` | Same algorithm, minor signature difference. | Medium-high |
| `ConcatChannels` in `WanVideoPipeline.cs:864`, `WanAnimatePipeline.cs:427`, `WanAnimate2Conditioning.cs` | Triplicated (the last is public, adds validation). | High |
| `WanAnimate2Conditioning.TrimReferenceFrame` vs `WanAnimatePipeline.cs:440 DropLeadingFrames` | Former is a fixed-arg special case of the latter. | High |
| `FlowMatchSampling.IsAnySelection(...)` refusal-guard block | Repeated at 6 call sites across `WanVideoPipeline.cs`/`WanVacePipeline.cs`/`WanS2VPipeline.cs`/`WanAnimatePipeline.cs` — hoistable to one `ThrowIfSamplerSelected(scheduler, familyName)` helper. | Medium-high |

## HartsyInference.Vulkan

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `VulkanUtilities.cs:14-16` | `VkApiVersion10`, `VkApiVersion11`, `VkApiVersion12` | Zero references (only `VkApiVersion13` is used). | High |
| `VulkanBackend.cs:3056-3062` | `CastToF16`'s second branch | Unreachable — first branch's condition is a strict superset and always returns first. | High |
| `VulkanBackend.cs:958` | `DispatchMatmulBatched`'s `batchIndex`/`bIs2D` params | Unused; combined with an early throw on `a.Shape[0] != 1`, `BatchedMatMul`'s loop can't get past its first iteration for `batch > 1`. | Medium |
| `VulkanDescriptorManager.cs:194 PushSet` vs `:232 WriteSet` | Identical descriptor-info-building logic, differ only in the final Vulkan call. | Medium |
| `VulkanBackend.cs:1122` (tiled matmul), `:1316 Conv2D`, `:2111 DispatchMatmulWithOffsets` | All three independently build the same 13-entry spec-constant array and 11-uint push-constant layout. | Medium |
| `VulkanBackend.cs:1796 DispatchFlashAttention` vs `:2049 ScaledDotProductAttentionNaive` | Identical output-dtype-cast tail logic. | Medium |

## HartsyInference.Vision

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `DinoVit/DinoViTEncoder.cs` | `DinoViTLayer.ToHeads`/`.FromHeads` (private) | Zero call sites; superseded by `backend.Permute0213` used in the same class's `Attention`. | High |
| `Siglip/SiglipVisionEncoder.cs` | `ReshapeToHeads`/`ReshapeFromHeads`/`ReshapeToMh`/`ReshapeFromMh` | Four near-identical manual host-loop reshapes duplicating each other AND the existing `IBackend.Permute0213` op that `Dinov2VisionEncoder`/`DinoViTEncoder` already use. | High |
| `Detection/GroundingDino/GroundingDinoMultiheadAttention.cs` | `Forward` | Duplicates `RtDetrAttention.MultiHead` line-for-line; `GroundingDinoEncoder.BiAttentionFusion` in the same file family already calls `RtDetrAttention.MultiHead` directly for the identical op. | High |
| `Detection/YoloModel.cs`, `YoloSegModel.cs`, `FaceDetection/YoloV8FaceModel.cs`, `YoloV11Model.cs`, `YoloV11PoseModel.cs` | `ConcatChannel`/`Upsample` (5 byte-identical private copies) | `YoloV11PoseModel` already correctly calls `YoloV11Model.ConcatChannel`/`.Upsample` (marked `internal`) instead of duplicating — the other 4 don't follow that precedent. | High |
| `Segmentation/Sam2/*`, `Detection/RtDetrAttention.cs`, `Detection/GroundingDino/*` | The `[1,S,H·D]→[1,H,S,D]` "ToHeads" reshape-via-Permute0213 pattern | Reimplemented near-identically in `DecoderAttention`, `HieraBlock`, `RtDetrAttention`, `SwinBackbone`, `Dinov2Layer`, `DinoViTLayer` (~6 sites). | Medium |
| `Detection/YoloSegModel.cs`, `FaceDetection/YoloV8FaceModel.cs` | Whole backbone+neck body | Real duplication but already acknowledged/tracked by the authors (own doc comments explain it, `YoloV11PoseModel.cs` carries a `TODO(refactor)` toward a shared `YoloV11Trunk`) — not a new finding. | High (known) |

**Real bug found opportunistically (NOT dead code — flagging separately, zero code changed):** `YoloModel.Forward`, `YoloSegModel.ForwardWithProtos`, `YoloV8FaceModel.Forward`, `YoloV11Model.Forward`, `YoloV11PoseModel.Forward` all leak the `x6` intermediate tensor (P4 backbone output) every forward pass — passed into `ConcatChannel` but never `.Dispose()`d, unlike every sibling tensor in the same graph. `Tensor` has a finalizer so this isn't a permanent leak, just non-deterministic cleanup/extra GC-and-native pressure per inference call, in 5 places. `YoloModel.cs` has a comment right at the leak site claiming "the graph is correct," which is now wrong — left in place as the breadcrumb for whoever fixes this.

## HartsyInference.Cuda

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `CudaGraph.cs` | `TryUpdate` | Zero call sites repo-wide. | High |
| `CudaDriverApi.cs` | `cuModuleLoadData` (non-Ex) | `CudaModule` only ever calls `cuModuleLoadDataEx`. | High |
| `CudaDriverApi.cs` | `cuEventQuery` | Zero call sites. | High |
| `CudaDriverApi.cs` | `CU_EVENT_DEFAULT` | Every event uses `CU_EVENT_DISABLE_TIMING`. | High |
| `CudaDriverApi.cs` | `cuDeviceGetGraphMemAttribute`, `CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT`, `CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT` | Diagnostic binding + constants, never referenced. | High |
| `CudaDriverApi.cs` | `cuMemcpyPeer` (sync variant) | Only `cuMemcpyPeerAsync` is used. | High |
| `Profiling/CudaProfilerControl.cs` | whole class | No call sites, but an nsys-profiling helper meant for ad-hoc manual use — may be intentionally kept. | Medium |
| `CudnnConv.cs` + `CudnnSdpa.cs` | `BuildExecutionPlan`/`TryPlan` heuristic-search loop | Near-identical cuDNN backend-graph engine-config search; the code ITSELF cross-references the other as "identical pattern" — strong hoist candidate. | High |
| `Fp8GemmExecutor.cs`, `Fp4GemmExecutor.cs`, `Int8GemmExecutor.cs` | constructor/Dispose/finalizer lifecycle | Identical cuBLASLt handle+workspace create/free/finalize boilerplate in all three, differing only in the SM-version gate and `Run()` body. | Medium |
| `LtGemmExecutor.cs` vs the three above | overall design | `LtGemmExecutor.TryRun` is already generic over dtypes, could in principle subsume Int8/Fp8 — but Fp4 needs block-scale pointer attributes LtGemm doesn't expose, so not a clean duplicate. | Low/uncertain |

(Unused `CU_DEVICE_ATTRIBUTE_*` constants also noted but not actionable — typical to keep as native-enum-surface mirrors.)

## HartsyInference.Engine (Audio/Services/Registry)

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Audio/Tts/TtsCatalog.cs:29` | `RegisteredIds` | Zero call sites outside its own declaration; sibling catalogs (`MusicCatalog`/`VcCatalog`/`SttCatalog`) don't expose an equivalent member at all. | High |
| `Audio/Stt/SpeakerDiarizer.cs` (~145-171) vs `Audio/Wake/Speakers/CamPlusEmbedder.cs` (~100-160) | CAM++ checkpoint loading (prefix detection + encoder construction) and the cepstral-mean-normalization embedding loop | Near byte-identical (down to identical inline comments), duplicated instead of hoisted into one shared CAM++ helper. | High |
| `TtsCatalog.cs`/`VcCatalog.cs`/`MusicCatalog.cs`/`SttCatalog.cs` | `private static Dictionary<string,T>? _registry` lazy-init idiom | Repeated 4x with identical comment; only a 2-line idiom over 4 different value-types — low value to hoist, flagged for awareness only. | Low |

## HartsyInference.ModelAssets

No dead private methods/fields/classes found anywhere in the package (grep-confirmed zero call sites repo-wide).

| File(s) | Symbol | Reason | Confidence |
|---|---|---|---|
| `CheckpointConverters/{BooguImage,Krea2,Ideogram4}CheckpointConverter.cs` + `Engine/Recipes/Image/{Ideogram4Recipe,MageFlowRecipe,BooguImageRecipe,Krea2Recipe}.cs` | `RemapQwenLanguageKey`/`RemapQwenKey` | Byte-identical Qwen3-VL language-tower key-remap duplicated 7 times total. | High |
| `CheckpointConverters/{BooguImage,Ideogram4,Krea2}CheckpointConverter.cs` | `StripTransformerPrefix` | Identical 3-branch prefix-strip duplicated verbatim. | High |
| `Lora/Mappers/{AiToolkitFlux,KohyaFlux,DiffusersFlux,KohyaSd,WanLora}Mapper.cs` | `GroupBuffer` (private nested class) + `GetOrCreate` + finalization loop | Byte-for-byte identical `GroupBuffer` class and near-identical group-collect/finalize logic across all 5 LoRA mappers, differs only in a warning string and 1-2 per-format details. | High |
| `CheckpointConverters/{Flux,Flux2,Lens,QwenImage,Sd3}CheckpointConverter.cs` | `SplitQkvWeight`/`SplitQkvBias` | Same memcpy-based 3-way QKV split reimplemented in ≥5 files; `Sdxl`/`SdxlRefiner` already share the equivalent via `CheckpointConvertUtils.SplitInProjWeight/Bias`, showing centralization is the established pattern. Flux's variant uses quant-aware `DType.ComputeByteCount`; Lens's uses plain `SizeInBytes` — a hoist must preserve that distinction. | High |
| `CheckpointConverters/{BooguImage,Krea2}CheckpointConverter.cs` | `DiscoverShards`/`LoadShards`/`LoadShardsRemap` | Near-identical shard-discovery-and-merge helpers, differ mainly in error text and default key-map. | Medium |
| `CheckpointConverters/{AuraFlow,Flux,Flux2,HunyuanImage,HunyuanVideo}CheckpointConverter.cs` | `SwapScaleShiftHalves` family | Same `[shift,scale]`↔`[scale,shift]` half-swap reimplemented 5x; Flux2 adds GGUF block-alignment validation and HunyuanVideo force-casts to F32 — a merge must preserve those differences. | Medium |

**Checked, not flagged:** `Gguf/Codecs/Codec_{Q4_0,Q4_1,Q5_0,Q5_1}.cs` — superficially similar structure but genuinely different bit-layout math per format; correctly left as bespoke per-codec code.

## HartsyInference.LLM

No dead-code candidates met the confidence bar (every private/internal/public member checked had live call sites).

| File(s) | Symbol | Reason | Confidence |
|---|---|---|---|
| `Embeddings/BertEmbeddingModel.cs`, `Seq2Seq/T5Model.cs`, `Multimodal/SiglipVlmEncoder.cs`, `Multimodal/MllamaVisionEncoder.cs`, `Multimodal/Qwen25VlEncoder.cs`, `Multimodal/Qwen3VlEncoder.cs`, `Ssm/MambaModel.cs`, `Ssm/Mamba2Model.cs` | `private static Tensor Relabel(Tensor t)` | Near-identical [in,out]→[out,in] copy-relabel helper reimplemented in 8 separate model files. | High |
| `Ssm/RwkvModel.cs` vs `Ssm/Rwkv7Model.cs` | `Lin`, `LayerNorm`, `ShiftDiff` | Byte-identical (or near-identical, doc-only diff) between the two RWKV variants. | High |
| `Sampling/JsonGrammarStep.cs` vs `Sampling/SentinelJsonGrammarStep.cs` | `TokenText(int id)` | Byte-identical private helper in both files; natural home is `JsonVocabText`, which both already depend on. | High |
| `Generation/PromptBuilder.cs` vs `Generation/DynamicBatchScheduler.cs` | `BuildPromptIds` | `TextGenerationPipeline` correctly delegates to the shared `PromptBuilder.BuildPromptIds`; `DynamicBatchScheduler.BuildPromptIds` reimplements the same logic inline instead of calling it. | High |
| `Multimodal/Qwen25VlEncoder.cs` vs `Multimodal/Qwen3VlEncoder.cs` | `BuildRope` + `SetCosSin`; merge-block patchify loop in `Encode` | `SetCosSin` byte-identical; `BuildRope`/patchify loop differ only trivially. | High |
| `Generation/TextGenerationPipeline.cs` vs `Generation/DynamicBatchScheduler.cs` | `GraphDecodeWarmupHeadDims`/`HeadDimPerLayer` | Same 4-line loop duplicated verbatim. | High |
| `Multimodal/MllamaGenerator.cs` vs `Multimodal/MultimodalGenerator.cs` | `SampleLast` | Near-identical (project last hidden row → logits → sample) duplicated across both generators. | High |
| `Multimodal/SiglipVlmEncoder.cs`, `LlavaNextEncoder.cs`, `Qwen25VlEncoder.cs`, `MllamaVisionEncoder.cs` | `Dbg`/`Dump` debug helpers | All four implement "mean/maxabs + optional raw-f32 dump" with minor per-file variation. | Medium |
| `Multimodal/MultimodalGenerator.cs` vs `Multimodal/Qwen35VlGenerator.cs` | pre/image/post embedding-splice assembly in `Generate` | Both build `[pre][image][post]` host embedding sequences the same way. | Medium |
| `Generation/TextGenerationPipeline.GenerateGraphDecode` vs `Generation/DynamicBatchScheduler.CaptureGraphSession` | graph-capture warmup/setup sequence | Code itself documents this as intentionally mirroring — acknowledged duplication, not an oversight. | Informational |

## HartsyInference.Diffusion (Adapters/Conditioning/Quality/Requests/Schedulers/Utilities/Sampling/Pipelines)

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Adapters/ControlNetUnionFusion.cs:45` | `NumControlTypes` | Dead public property, zero call sites. | High |
| `Adapters/IpAdapterFaceIdPlusProjection.cs:30` | `UseShortcut` | Dead public property, zero call sites. | High |
| `Adapters/ReduxImageEncoder.cs`, `IpAdapter.cs`, `IpAdapterFaceIdProjection.cs`, `IpAdapterFaceIdPlusProjection.cs`, `IpAdapterStandardProjection.cs`, `IpAdapterPlusResampler.cs` (×2) | private `EnsureF32` | Byte-identical one-liner defined 7 times; candidate for `Utilities/DtypeCastHelper`. | High |
| `Adapters/FluxControlNet.cs:276-286` vs `Adapters/QwenImageControlNet.cs:175-185` | `ProjectResidual` | Byte-identical private static method, same signature. | High |
| `Pipelines/HunyuanImagePipeline.cs`, `ChromaRadiancePipeline.cs`, `Krea2Pipeline.cs`, `Ideogram4Pipeline.cs`, `QwenImagePipeline.cs` | private static `SumBytes` | Byte-identical across all 5 files. | High |
| `Pipelines/FluxPipeline.cs:1494` (`internal static`) vs `ChromaPipeline.cs:712` | `PackLatent`/`UnpackLatent` | Algorithmically identical 2×2-patchify pack/unpack (16ch); Flux's is already `internal static`, so Chroma could call it directly instead of duplicating. | High |
| `Schedulers/DdimScheduler.cs`, `DpmPlusPlus2MScheduler.cs`, `LcmScheduler.cs`, `TcdScheduler.cs` | `AddNoise` | Byte-identical body across all 4 VP-diffusion schedulers. | High |
| `Schedulers/LcmScheduler.cs` vs `TcdScheduler.cs` | `ComputeAlphasCumprod` + `SetTimesteps` | Near-identical (diff shows only cosmetic naming/comment differences). | High |
| `Pipelines/ChromaPipeline.cs` vs `HunyuanImagePipeline.cs` | `FreeTransformerWeights` | Identical modulo one extra line in Chroma (`InvalidateStepGraph`); `QwenImagePipeline`'s N-way variant is genuinely different (own doc comment says so) — excluded. | Medium |
| `Utilities/FeatureCache.cs` vs `Utilities/DeviceFeatureCache.cs` | state-machine bookkeeping (threshold/counters/Reset/MarkCompute) | Structurally parallel, but `DeviceFeatureCache`'s own doc explains why it's separate (host `DataPointer` reads evict CUDA activations mid-forward); only counter bookkeeping, not the tensor math, is realistically shareable. | Medium |

**Checked and explicitly excluded** (real duplication risk on the surface, not duplicates on inspection): `EstimateActivationReserveBytes` (5 pipelines, different formulas per architecture — documented as intentional); `ClassifierFreeGuidanceStep` (4 pipelines, different signatures per family — legitimate variation, not copy-paste).

**Known-broken doc render found opportunistically (not fixed, out of scope):** a `&lt;list&gt;`-collapse regression pattern also present in `Models/Denoisers/Ideogram4Config.cs` and `Models/Denoisers/HiDreamTransformer.cs` (outside this batch's scope, part of a concurrent session's edits to `Models/`).

## HartsyInference.Diffusion.Models.Denoisers (A-K, incl. DiTBlocks/UNetBlocks/Diamond)

**Dead code (high confidence, zero call sites repo-wide):**

| File | Symbol | Reason |
|---|---|---|
| `AceStepDit.cs` | nested `Block.Pick`/`PickOpt`/`PickF32`/`PickOptF32` | byte-identical reimplementations of the outer class's own private statics (nested class already has access). |
| `AceStep15Dit.cs` | nested `Layer.EnsureF32` | identical duplicate of outer class's own `EnsureF32`. |
| `AceStepDebugDump.cs`, `AnimaDebugDump.cs`, `ChromaDebugDump.cs`, `ErnieImageDebugDump.cs` | `Enabled` property | zero call sites (contrast: `AuraFlowDebugDump.Enabled` is actually used). |
| `FluxToolsConfig.cs` | `FluxToolsConfig` record + `FluxToolVariant` enum | entirely unused; tool variants are actually routed via `FluxConfig` flags. |
| `FluxConfig.cs` | `FromWeights(...)` | no call sites. |
| `ErnieImageConfig.cs` | `V1Turbo` preset | no call sites. |
| `FLiteConfig.cs` | `Texture` preset | no call sites. |
| `DiTBlocks/DiTUtils.cs` | `ConcatAlongLastDim`, `LinearProjectBatched`, `LinearProject1D`, `PadLastDim(Tensor,int,int)`, `ConcatPooled` | dead — `Sd3Pipeline`/`HiDreamPipeline` each carry their own private reimplementations instead of calling these. |
| `DiTBlocks/ErnieImageRope.cs` | `ApplyRotaryEmb(...)` + private `ApplyNonInterleavedRotation` | whole unused host/CPU rotation path; model uses `ApplyRotaryEmbGpu`. |
| `DiTBlocks/AuraFlowJointBlock.cs`/`AuraFlowSingleBlock.cs` | fields `_innerDim`, `_mlpDim`, (`_qkNormEps` in Joint) | write-only, never read. |
| `DiTBlocks/HiDreamBlock.cs:584` | local `inShape` in `MoeForward` | explicitly discarded (`_ = inShape;`), never used. |
| `DiTBlocks/FLiteRope.cs` | `ApplyRotation(...)` + `RotateOnePosition` | unused host rotation path; `FLiteAttention` always uses GPU `backend.ApplyRope`. |
| `DiTBlocks/Kandinsky5Rope.cs:48` | public `CachedSeqLen` | no readers anywhere. |
| `Kandinsky5Transformer.cs:742` | `ToBchw` | zero call sites; its own doc comment falsely claimed internal use. |
| `UNetBlocks/CrossAttentionBlock.cs` | outer-class fields `_channels`, `_headDim`, `_crossAttentionDim` | write-only (nested sub-blocks have their own same-named, actually-used fields). |
| `UNetBlocks/DownBlock.cs` | field `_inChannels` | write-only. |

**Dead code (medium confidence):** `HunyuanVideoDit.cs:59` `OnBlockOutput` — invoked internally but never assigned anywhere in the repo; may be an intentional parity-harness extension point (analogous hooks in `SeedVr2Dit`/`LtxVideo2Transformer` ARE used).

**Duplicate logic:**

| Files | Symbols | Confidence |
|---|---|---|
| `AnimaDebugDump.cs`, `AuraFlowDebugDump.cs` (+ ~19 sibling `*DebugDump.cs` repo-wide) | entire class structure (`WriteRawF32`, `EnsureInit`/lock, env-var dir resolve) | High — same I/O plumbing, not architecture-specific (`AceStepDebugDump` is a deliberate exception, resolves env var per-call for test isolation). |
| `DiTBlocks/AuraFlowJointBlock.cs`/`AuraFlowSingleBlock.cs` | `GetOrFakeOnes` | High — byte-identical; sibling's own doc comment points at the other. |
| `DiTBlocks/AnimaLlmAdapter.cs`, `AnimaLlmAdapterBlock`, `ErnieImageBlock.cs` | `LoadAsF32` | High — all three byte-identical to existing `DiTUtils.LoadF32`. |
| `DiTBlocks/ChromaDoubleStreamBlock.cs`, `ChromaSingleStreamBlock.cs` | private `NormModulate` | High — reimplements `DiTUtils.NormModulate` inline instead of calling it. |
| `DiTBlocks/ChromaDoubleStreamBlock.cs` vs `ChromaSingleStreamBlock.cs` | `SliceModRows`/`SliceModRowsDevice` | Medium — single-stream version is the `rowStart=0` special case of double-stream's parameterized version. |
| `DiTBlocks/HiDreamBlock.cs` | private `GpuModulate` | Medium-high — reimplements `DiTUtils.Modulate`/`NormModulate`. |
| `Kandinsky5Block.cs` (`Kandinsky5BlockOps.NormModulate`) vs `DiTUtils.NormModulate` | | Medium-high — near-identical, forces F32 instead of following input dtype. |
| `ChromaTransformer.cs`, `FluxTransformer.cs` | `MoveAcross`/`CopyAcross` | High — identical peer-copy helpers. |
| `FluxTransformer.cs` vs `Flux2Transformer.cs` vs `ChromaTransformer.cs` | private `LayerNormNoAffine` | Medium-high — Flux/Flux2 reimplement it locally while Chroma already calls the shared `DiTUtils` version. |
| `ChromaTransformer.cs`, `FluxTransformer.cs`, `Flux2Transformer.cs` | `ExtractImageTokens`; `ConcatTextImage`/`ConcatAlongSeqDim3D` | Medium |
| `DiTBlocks/FLiteAttention.cs` (`OnesHeadDim`) vs `FLiteBlock.cs` (`OnesHidden`) | lazy all-ones tensor cache | Medium — small, may not be worth sharing. |
| `Krea2Transformer.cs` (`PatchifyChannelOuter`/`UnpatchifyChannelOuter`) vs `QwenImagePipeline` (`PackLatent`/`UnpackLatent`) | same patchify algorithm structure | Medium — not byte-identical. |

**Also flagged (comment/attribute placement, not a logic issue, left alone):** `DiTBlocks/ErnieImageRope.cs` — `[MethodImpl(AggressiveInlining)]` sits on `ApplyRotaryEmbGpu` but reads like it was meant for the small private `ApplyNonInterleavedRotation` right below it — likely a merge artifact.

## HartsyInference.Diffusion.Models.Denoisers (L-Z, incl. DiTBlocks/UNetBlocks)

| File(s) | Symbol | Reason | Confidence |
|---|---|---|---|
| `LtxVideoTransformer.cs`, `MatrixGame2Transformer.cs`, `MatrixGame3Transformer.cs`, `WanS2VTransformer.cs`, `WanVaceTransformer.cs`, `WanVideoTransformer.cs`, `WanAnimate2Transformer.cs`, `WanAnimateTransformer.cs`, `LanceTransformer.cs` | private `LoadF32(weights, key)` | Byte-identical helper duplicated across 9 files; `DiTUtils.LoadF32` already exists and is already used by `LtxVideo2Transformer.cs`. | High |
| `MatrixGame2Transformer.cs` + `MatrixGame3Transformer.cs` | `SliceFrames`, `SliceRows`, `CloneCpu` | Byte-identical (diff = zero output). | High |
| `Lumina2Transformer.cs` (and `ZImageTransformer.cs`, out of this batch) | `ConcatAlongSeqDim` | Byte-identical. | High |
| `Lumina2DebugDump.cs` | public `Enabled` property | Zero call sites repo-wide; `Dump`/`DumpOutput` self-guard directly. | High |
| `MatrixGame3Config.cs:111` | `Ti2V5BShape` static alias | No in-repo call sites, but public — may serve external (SwarmUI) consumers. | Medium |
| `OasisDit.cs:209` | `private static double ElapsedMs(long)` | Zero call sites; half-wired dead profiling path (`tsEntry` at line 97 also never read). | High |
| `OmniGen2Transformer.cs:601` | `PatchifyLatent` | Byte-identical to `DiTUtils.PatchifyNCHW`, which the same class already calls elsewhere. | High |
| `OmniGen2Transformer.cs:646` | `UnpatchifyTokens` | Same loop as `DiTUtils.UnpatchifyToNCHW` plus a negation; could unify behind a `negate:` param. | Medium |
| `DiTBlocks/AdaLNModulation.cs` | `ApplyModulation` (static) | Zero real call sites (only referenced in doc-comment prose). | High |
| `DiTBlocks/DiTUtils.cs` | `LinearProject1D`, `LinearProjectBatched`, `ConcatAlongLastDim`, `ConcatPooled`, `PadLastDim` | Zero call sites repo-wide (corroborates the same finding from the A-K batch). | High |
| `DiTBlocks/AuraFlowJointBlock.cs` | fields `_innerDim`, `_mlpDim`, `_qkNormEps` | Write-only. | High |
| `DiTBlocks/AuraFlowSingleBlock.cs` | fields `_innerDim`, `_mlpDim` | Write-only. | High |
| `DiTBlocks/AuraFlowJointBlock.cs` + `AuraFlowSingleBlock.cs` | `GetOrFakeOnes` | Near-identical private helper (AuraFlow non-affine QK-norm shim). | Medium-high |
| `DiTBlocks/StableAudioDit.cs` (2 places, same file) + `StableAudioNumberEmbedder.cs` | `Pick`/`PickOpt`/`PickF32` trio | Identical 3-method trio duplicated 3x, two within the same file. | High |
| `DiTBlocks/LtxVideo2TextConnectors.cs:267` | `Connector.AddInPlace` | Zero call sites; `Forward` uses `backend.Add` instead. | High |
| `DiTBlocks/Lumina2Block.cs` + `Lumina2ContextRefinerBlock.cs` | `ReshapeToMultiHead`, `ReshapeFromMultiHead`, `RepeatKvHeads`, `LoadAsF32` | Near-byte-identical host-fallback helpers (same idiom also appears in TextEncoders, outside scope). | High |
| `DiTBlocks/Krea2Block.cs` + `Krea2TextFusion.cs` | `SwiGlu` | Near-identical to each other and to `DiTBlocks/SwiGluFfn.cs`, already in the same directory. | Medium-high |
| `DiTBlocks/WanAnimateFaceEncoder.cs` + `WanS2VAudioEncoder.cs` | `LayerNormSilu`, `ChannelsToTokens`, `Transpose`/`TokensToChannels` | Byte-identical/near-identical across two Wan-variant encoders. | High |
| `DiTBlocks/ZImageBlock.cs` + `ZImageContextRefinerBlock.cs` | `LoadSplitQkv` | Byte-identical; duplication already acknowledged in the second file's own doc comment. | High |
| `DiTBlocks/ZImageBlock.cs` + `ZImageContextRefinerBlock.cs` | `ForwardSwiGlu` | Near-identical (one adds a trace hook). | Medium |

**3 real pre-existing doc bugs fixed in this pass (comment relocation, no logic change):** `MiniMaxH3Transformer.cs` (`Attention`), `MiniMaxMusic3Dit.cs` (`Block.Attend`), `WanAnimateTransformer.cs` (`Forward`/`EncodeMotion`) each had two `<summary>` blocks stacked over one member, orphaning a summary meant for an undocumented method further down.

## HartsyInference.Diffusion.Models (Music/TextEncoders/Vae)

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| ~20 sites across Music/TextEncoders/Vae | `EnsureF32`/`CastToF32IfNeeded`/`AsF32`/`F32`/`LoadF32` one-liners | Byte-identical `t.DType==DType.F32 ? t : t.CastTo(DType.F32)` redefined dozens of times; `TextEncoders/TextEncoderTensorHelpers.CastToF32IfNeeded` already exists as a shared home but most sites don't call it; Vae/Music have no shared equivalent at all. | High |
| `Models/Vae/*` (8 files) | `Bias()` helper | Duplicated across 8 Vae files; the LTX family already hoisted an equivalent — `LtxVideoVaeEncoder` just doesn't use it. | High |
| `Models/Vae/*` (~7 files) | `Clone`/`CloneRef`/`CloneTensor` | Same tensor-clone-via-`Buffer.MemoryCopy` logic duplicated across ~7 files (e.g. `Wan22VaePatch.CloneRef`). | High |
| `Models/Vae/*` (3 files) | `SliceChannels` | Duplicated across 3 Vae files. | Medium-High |
| `Models/Vae/*Encoder.cs` (5 files) | `EncodeRgbFrame` | Duplicated across 5 Vae encoder files. | Medium-High |
| `Models/Vae/LtxVideoVaeEncoder.cs` | `ChannelRms` (private copy) | Duplicates the already-`internal`, doc-flagged-as-shared `LtxVaeResnetBlock3d.ChannelRms`; encoder should call the shared one. | High |
| `Models/Vae/LtxVideoVaeDecoder.cs` vs `LtxVideo2VaeDecoder.cs` | `Reverse(bool[])` | Genuinely duplicated (both used). | Medium |
| `Models/Vae/LtxVideo2VaeDecoder.cs:357` | `Reverse(int[])` | Dead — private, zero call sites in this file (the twin in `LtxVideoVaeDecoder.cs` IS used; only this copy is dead). | High |
| `Models/Vae/MiniMaxH3VideoVaeDecoder.cs:663` | `private static int Sum(int[] values)` | Dead — private, zero call sites anywhere in the file. | High |
| `Models/Vae/QwenImage/QwenImageVaeOps.cs` `FlattenGamma` (~85-89) | redundant branch | `if (ReferenceEquals(f32, gamma)) return f32; return f32;` — both branches return the same value, the inner `if` is a no-op. | High |
| `Models/Vae/QwenImage/*` Downsample class summary vs `Forward` | stale doc | Summary claims padding is "approximated with symmetric pad=1" but `Forward` actually uses the correct asymmetric `PadRightBottom` — content is wrong, not just verbose (flagged, not fixed — outside comment-style scope). | Medium |
| `Models/Vae/Mage/*` | `GlobalAvgPool`/`ScaleByChannel`/`ChannelLayerNormAffine`/`TimestepEmbedZero` | Each duplicated across the Mage encoder/decoder pair. | Medium |
| `Models/TextEncoders/GptOssEncoder.cs`/`GptOssMoeFfn.cs` | `EnsureMasks`/`EnsureRopeTable`-style helpers | Overlap with logic already centralized in `TextEncoderTensorHelpers`. | Medium |
| `Models/Music/AceStep15*` (4 files) | per-file `EnsureF32` redefinitions | Same cross-cutting duplicate as the top row, called out separately (4 copies in one model family). | High |
| `Models/Vae/MiniMaxH3VideoVaeWeights.cs` vs `Models/Vae/LtxVideo25WeightScope.cs` | weight-scope/registration abstraction | Same conceptual abstraction (scoped weight enumeration) reimplemented per-model; non-trivial to merge, needs design review. | Medium |
| `Models/TextEncoders/Qwen25VlImageProcessor.cs` vs `Qwen3VlImageProcessor.cs` | `ResizeAndNormalize`/`SmartResize` | Near-identical bilinear-resize-and-normalize host loops and rounding algorithm; Qwen3's `SmartResize` has one extra `maxSideLength` clamp step. | Medium |

## HartsyInference.Engine.Recipes

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Image/HunyuanImageRecipePipeline.cs:21` | `DefaultCfg = 3.5f` | Never referenced; pipeline uses `_config`/family defaults instead. | High |
| `Image/QwenImageRecipePipeline.cs:30` | `DefaultCfg = 2.5f` | Same dead pattern (only these two exist repo-wide). | High |
| `Image/Flux1Recipe.cs` (`ConvertClipLFromStandalone`/`StripT5Prefix`), `Sd3Recipe.cs` (`StripT5Prefix`), `ChromaRecipe.cs` (`LoadT5FromStandalone`), `ChromaRadianceRecipe.cs` (`NormalizeT5`), `HiDreamRecipe.cs:209-229` (`StripStandalonePrefix`), `Kandinsky5TextEncoding.cs` (`ConvertClipLFromStandalone`) | comfy `text_encoders.*.transformer.` prefix + `position_ids` strip | 6 near-identical reimplementations; `HiDreamRecipe`'s is the general (parameterized-prefix) form and the right hoist target; `Flux1Recipe.StripT5Prefix` doesn't drop `position_ids` while the others do — a real behavioral inconsistency, not just duplication. | High |
| `Krea2Recipe.cs`/`QwenImageRecipe.cs` (`CastToF32`), `Lumina2Recipe.cs` (`CastWeightsToF32`), `HiDreamRecipe.cs:199` (`CastToF32`), `OmniGen2Recipe.cs` (`CastWeightsToBf16`), `Video/HunyuanVideoRecipe.cs` (`CastBf16ToF16`+inline loop), `Video/Kandinsky5VideoRecipe.cs` (2 inline loops) | weight-dict dtype cast | All reinvent `HartsyInference.Engine.Features.VaePrecisionHelper.CastVaeWeights(dict, targetDtype)`, which already does this more robustly (disposes partial results on failed cast) and is already used this way by `LtxVideo2Recipe`/`LtxVideoRecipe`/`LanceVideoRecipe`/`WanVideoRecipe`. | High |
| `Image/Flux1Recipe.cs`, `ChromaRecipe.cs`, `HiDreamRecipe.cs`, `Sd3Recipe.cs`, `Lumina2Recipe.cs`, `Krea2Recipe.cs`, `HunyuanImageRecipe.cs`, `Video/MiniMaxH3Recipe.cs` | DiT 2-way shard-split-block (per-block/shared weight bytes → query both backends' free VRAM → `PlacementPlanner.DitSplitPlan` → log) | Near-verbatim ~15-line block repeated 8 times. `QwenImageRecipe.cs` correctly does NOT share this (genuine N-way sharding, different path). | High |
| `Image/Sd15RecipePipeline.cs`, `Image/SdxlRecipePipeline.cs` | `IpAdapterCache` field + `LookupIpAdapter`/`CacheIpAdapter` + dispose loop | Byte-for-byte identical. | High |
| `Image/BooguImageRecipe.cs:117-144`, `ErnieImageRecipe.cs:154-169`, `Krea2Recipe.cs:167`, `MageFlowRecipe.cs:113`, `Ideogram4Recipe.cs:156` | side-model loader (load safetensors → skip `.scaled_fp8` marker keys → optional key transform → `ApplyFp8ScaledDequant`) | Copy-pasted shape across 5 recipes. | High |
| `Video/WanAnimateRecipe.cs`, `WanAnimate2Recipe.cs`, `WanS2VRecipe.cs`, `WanVaceRecipe.cs`, `WanVideoRecipe.cs` | umT5 side-model loading block | Near-verbatim across all 5. | High |
| Same 5 files | Wan2.1 VAE loading block | 4 near-verbatim, `WanVideoRecipe` has a Wan22Vae-branching variant. | High |
| `WanAnimateRecipePipeline.cs`, `WanS2VRecipePipeline.cs`, `WanVaceRecipePipeline.cs`, `WanVideoRecipePipeline.cs` (×2 sites), `WanAnimate2RecipePipeline.cs` (3-prompt variant) | prompt-encode sequence in `Generate` (tokenize → umT5 encode → `CfgHelper.SliceBatchElement` → `VideoRecipeUtils.ZeroPaddedRows` → sync → free) | 6 near-verbatim sites. | High |
| `WanAnimateRecipePipeline.cs`, `WanAnimate2RecipePipeline.cs`, `WanS2VRecipePipeline.cs`, `WanVaceRecipePipeline.cs` (byte-identical), `WanVideoRecipePipeline.cs` (structurally same + `_transformer2`) | `Dispose()` body | Field types differ per class, so not a free hoist. | Medium |
| `WanAnimateRecipePipeline.cs`, `WanVideoRecipePipeline.cs` (inline), `WanAnimate2RecipePipeline.cs` (already named `EncodeClipVision`) | CLIP-vision encode sequence | | Medium |

**Real bugs found opportunistically (not fixed):**
- `Video/WanS2VRecipe.cs` and `Video/WanVaceRecipe.cs`: `Construct`'s `catch` block disposes `loaders` but not `loraStack`, leaking the merged LoRA stack's weight tensors on the construction-failure path. `WanAnimateRecipe.cs`/`WanAnimate2Recipe.cs`/`WanVideoRecipe.ConstructBase` correctly call `loraStack?.Dispose()` in `catch` — 2 of 5 near-identical `Construct` methods dropped the line.

## HartsyInference.Audio.Models (Bark/Chatterbox/Csm/Demucs/Dia/FishSpeech/GptSoVits/Hubert/MeloTts/NeuTts/OpenVoice/Orpheus/ResembleEnhance/Rvc/SparkTts/StyleTts2/Vocoders/Wake/ZipVoice)

| File(s) | Symbol | Reason | Confidence |
|---|---|---|---|
| `Bark/BarkCausalStage.cs` | `ForwardLogits` | Public, self-documented "parity-debug only"; zero call sites repo-wide including tests. | High |
| `Bark/BarkFineModel.cs` | `DebugLogits` | Zero call sites repo-wide, no test exercises it. | High |
| `Demucs/HtDemucs.cs` + `DemucsConvBlock.cs`/`DemucsCrossTransformer.cs` | `DebugHook` and its 6 wired probe delegates | Never assigned anywhere in the repo (unlike the analogous `RvcRmvpe.DebugHook`, which IS wired by `RmvpeParityTests.cs`) — likely scaffolding for a Demucs parity test that was never written. | Medium-high |
| `Rvc/RvcRmvpeConfig.cs` | `GruHidden` property | `RvcRmvpe.cs` hardcodes its own `const int GruHidden = 256` and never reads `_cfg.GruHidden`. | High |
| `ResembleEnhance/ResembleEnhanceConfig.cs` | `DenoiserHiddenDim`, `DenoiserNumBlocks`, `DenoiserNumMiddleBlocks`, `DenoiserKernel` | `ResembleDenoiser`'s constructor takes no config and hardcodes its own consts; zero readers repo-wide. | High |
| `ResembleEnhance/ResembleEnhanceConfig.cs` | `UnivNetNc` property | `ResembleUnivNet` defaults `nc=96`; the only construction site never passes this through. | High |
| `StyleTts2/StyleHifiGanGenerator.cs` | `private static readonly int[] UpsampleKernels` | Declared, never read — kernel size is derived from the loaded weight tensor's shape instead. | High |
| `StyleTts2/StyleEncoder.cs` | `private static readonly float _invSqrt2` (outer class) | Declared, never referenced within `StyleEncoder`'s own methods. | High |
| `Codecs/NeuCodec/NeuCodecDecoder.cs`, `Kyutai/MoshiDepthTransformer.cs` (private `FlatToHeads`/`HeadsToFlat`) vs `Dia/DiaHeads.cs` | reshape helpers | Near-identical flat⇄head-major reshape math reimplemented privately instead of reusing the already-public, cross-namespace-reused `DiaHeads` methods (already flagged as future-unification debt in `DiaHeads.cs`'s own comment). | Medium |
| `Hubert/Hubert.cs` (`ExactGelu`) vs `Layers/Activations.cs` (`ErfGelu`/`Erf`) | duplicate erf-GELU polynomial | Identical Abramowitz-Stegun 7.1.26 formula; `Activations` is `internal` and documented as existing specifically to be shared. | High |
| `SparkTts/SparkTtsLm.cs` (`SliceLastFrame`) vs `CosyVoiceQwenLm.cs`, `OrpheusPipeline.cs`, `NeuTtsPipeline.cs` | duplicate helper | Byte-identical body duplicated across ≥4 files; a shared public version already exists in `VibeVoiceOps.cs` and could absorb these. | Medium |
| `Bark/BarkCausalStage.cs`, `Chatterbox/ChatterboxT3.cs`, `Csm/CsmModel.cs` (`SliceLast`) | duplicate helper | Byte-identical 5-line slice-last-row helper, also present in 7 more files repo-wide — an established repo convention, low-priority to consolidate. | Medium |
| `Demucs/DemucsConvBlock.cs`, `DemucsDConv.cs`, `DemucsCrossTransformer.cs`, `HtDemucs.cs` (`Bias` helper) | duplicate helper | Identical `TryGetValue → EnsureF32` 1-liner duplicated 4x in scope, plus ~14 more repo-wide — accepted repo convention. | Medium |

**Public-API items surfaced but not flagged as dead code** (outside the strict private/internal bar, listed for awareness): `FishSpeechConfig.V1_4`, `FishSpeechDualAr.DebugSlowLogits`/`DebugFastLogits`, `HubertConfig.Downsample`/`HubertBaseLs960`/`ContentVec`, `NeuTtsConfig.Nano`, `StyleDenoiser.SetNetwork` (own doc inaccurately claims a test calls it — no test does), `VocosConfig.Encodec24k` (unused preset masking a real gap: `Vocos.Forward` never reads `AdaNormNumEmbeddings`/`Padding`, so this preset would silently misbehave if ever constructed), `StyleTts2Config.LibriTts`/`LjSpeech`, `BiCodecDecoder.DecodeIntermediates`.

## HartsyInference.Audio.Models (Kokoro/Kyutai/LanguageModels/Vits/Whisper/Zonos/PocketTts/QwenTts/Moonshine/HeartMula/F5Tts)

**Dead code**

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `F5Tts/F5Ops.cs` | `GroupedConv1D` | Unused, superseded by `backend.Conv1d` per an explicit comment in `F5InputEmbed.cs:114`. | High |
| `F5Tts/F5Dit.cs` | `F5TimestepEmbedding.LastTimeEmb` | Written/disposed every `Forward()` call, never read anywhere. | High |
| `Moonshine/MoonshineTokenizer.cs` | `DecodeOne` | Zero call sites; streaming pipeline decodes via the batch `Decode(...)` instead. | High |
| `Moonshine/RotaryEmbedding.cs` | `RopeScaling` record, its `Llama3` preset, 4-arg `GetTables` overload | Zero callers repo-wide (a different `Core.Rope.RopeScaling` type is what CSM/Orpheus/Chatterbox actually use). | High |
| `Kyutai/MoshiDepthTransformer.cs`, `MoshiDepthConfig.cs`, `KyutaiTtsModel`/`KyutaiTtsPipeline` cluster | whole cluster | Superseded by `MoshiDepformer`+`MoshiTtsGenerator`; the real production path is a differently-located `KyutaiTtsModel` in `HartsyInference.Engine`. | High |
| `Whisper/WhisperOps.cs` | `TransposeMatrix`, `AddBiasBroadcast`, `Expand2DMaskTo4D` | Zero call sites repo-wide. | High |
| `LanguageModels/Gpt/GptConfig.cs` | `Bias` | Never read anywhere (self-acknowledged by an adjacent `// PARITY-TODO` comment — known gap, not oversight). | Medium |
| `PocketTts/PocketTtsFlowLm.cs` | `GenerateLatents`, `DebugPrimedHidden` | Zero call sites including tests. | High |
| `QwenTts/Qwen3TtsTalker.cs` | `AddCodecEmbed` | Superseded by `EmbedStep` + `Qwen3MtpCodePredictor.AddAcousticFeedback` in the actual pipeline. | High |

**Duplicate logic**

| File(s) | Symbol | Reason | Confidence |
|---|---|---|---|
| `Kyutai/MoshiDepformer.cs` + `MoshiTtsGenerator.cs` | `ArgMax`/`SampleTopK` | Byte-identical private helpers. | High |
| `Whisper/WhisperDecoder.BuildIncrementalCausalMask` vs `Zonos/ZonosBackbone.BuildCausalMask` | | Identical offset-aware causal-mask algorithm, differ only in NegInf sentinel. | High |
| `LanguageModels/Gpt/GptBackbone.BuildCausalMask` vs `Music/MusicGenDecoder.BuildCausalMask` (outside this batch's scope) | | Byte-for-byte identical. | High |
| `Whisper/WhisperOps.TransposeMatrix`/`AddBiasBroadcast` (dead here) vs `Diffusion/Models/TextEncoders/{ClipTextEncoder,ClipVisionEncoder}.cs`/`IpAdapterPlusResampler.cs` | | Near-identical reimplementation; `WhisperOps`'s own doc admits it "mirrors" `ClipTextEncoder.cs`. | Medium |
| `Zonos/ZonosSpeakerEncoder.SoftmaxOverTime` vs `QwenTts/EcapaSpeakerEncoder.SoftmaxOverTime` | | Byte-for-byte identical private static helper. | High |
| `PocketTts/PocketTtsStreamingTransformer` (`SplitQkv`/`Merge`/`AddUpdate`/`Rope`/`BuildMask`) vs `PocketTtsMimiDecoder` (`SplitQkvToHeads`/`MergeHeads`/`AddScaled`/`ApplyRopeInterleaved`/`BuildSlidingCausalMask`) | | Near-identical attention-block plumbing. | Medium |
| `PocketTts/PocketTtsFlowLm.GenerateVoiced` vs `GenerateLatents` | | Copy-pasted AR-step body; resolved automatically if the dead `GenerateLatents` is removed. | Medium |

**Redundant/dead branch:** `F5Tts/F5Ops.cs` `Grn` (~70-91) — a first loop computes `sq` and discards it (`_ = sq;`), immediately followed by a "two-pass" block that recomputes the same value and uses it. High confidence.

## HartsyInference.Audio.Models (Codecs/CosyVoice/Music/VibeVoice)

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `VibeVoice/VibeVoiceOps.cs` | `LayerScaleApplyCF`, `AdaLnModulate`, `AdaLnGatedAdd`, `RmsNormChannelsFirst` (host), `CausalConv1d`, `CausalConvTranspose1d` | Zero callers repo-wide; each has a GPU-resident twin actually used (`ChannelScaleGpu`, inline `ModulateGpu`/`GatedAddGpu` in `VibeVoiceDiffusionHead`, `RmsNormChannelsFirstGpu`); the two causal-conv methods were additionally referenced only from now-corrected misleading doc comments in `SConv1d.cs`/`SConvTranspose1d.cs`. | High |
| `Music/MusicGenDecoder.cs` | `Forward`, `EmbedFrames`, `ForwardStep` | Zero call sites; `MusicGenPipeline` exclusively uses the graph-batched path. | High |
| `Music/MusicGenBlock.cs` | `Forward` | Only caller is `MusicGenDecoder.Forward` above, itself dead — transitively dead. | High |
| `Music/MusicGenConfig.cs` | `Melody`, `StereoSmall`, `StereoMedium`, `StereoLarge` presets | No call sites in src/tests (only `Small`/`Medium`/`Large`/`AudioGen` used); could be intentional public API for NuGet consumers. | Medium |
| `CosyVoice/CosyVoiceFlow.cs` | `InferenceChunk`, `InferenceGrowing` | Zero call sites anywhere incl. tests; `InferenceChunk`'s doc records a real measured mel-drift finding — flag, don't blind-delete. | High |
| `CosyVoice/CosyVoiceFlow.cs` | `TrimPromptFrames`, `TrimTrailingFrames`, `SliceFrames` | Each reimplements the identical per-channel contiguous-range memcpy already provided by `ConditionalCfm.SliceNoise` (public, used elsewhere in the same file). | High |
| `Codecs/EnCodec/SeaNetBlock.cs`, `SeaNetDecoder.cs` | private `GetExtraRightPadding` (x2) | Dead — only `SeaNetEncoder.cs`'s copy is ever called; identical body copy-pasted 3x. | High |
| `Codecs/EnCodec/SeaNetDecoder.cs` | `EnCodecConvPad.PaddedConv`'s `eluUnused` parameter | Never read in the body; no call site passes it explicitly. | High |
| `Codecs/Dac/DacResidualVectorQuantizer.cs` | codebook index→vector reconstruction loop | Copy-pasted verbatim between `Encode` and `Decode` in the same file. | High |
| `Codecs/Dac/DacResidualUnit.cs` ↔ `Codecs/Snac/SnacResidualUnit.cs` | `Forward` | Near-verbatim copy-paste (Snake→Conv→Snake→Conv→center-crop-and-add), differing only in `groups` support (Oobleck's version checked and is genuinely distinct — not flagged). | High |
| `Codecs/Dac/DacResidualVectorQuantizer.cs` ↔ `Codecs/Snac/SnacResidualVectorQuantizer.cs`, `Codecs/BiCodec.cs` (`L2NormalizeRows`), `ResidualVectorQuantizer.cs` (`Normalize`) | `LoadWeights`, `L2NormalizeRows`, cosine-nearest-neighbor lookup loop | Byte-identical `LoadWeights`/`L2NormalizeRows`; near-identical NN loop; the same L2-normalize-with-1e-12-epsilon pattern appears independently in 4 files — worth a shared `VqOps` helper. | High |
| `Codecs/Snac/{Encoder,Decoder,ResidualUnit,ResidualVectorQuantizer}.cs`, `Codecs/WavTokenizer/WavTokenizer.cs` | private `LoadFusedWeight` | Identical body (`EnsureF32` weight_g/weight_v → `WeightNormFusion.Fuse`) copy-pasted in 5 places; `Oobleck/OobleckOps.cs` already solves this with one shared helper — same fix applies. | High |
| `Codecs/Oobleck/OobleckOps.cs` | `LoadFusedWeight` vs `LoadFusedTransposeWeight` | Byte-identical bodies today; kept `<remarks>` documents this as intentional (tracks a convention that could diverge) — awareness only. | Medium |

**Checked, not flagged (legitimately distinct designs):** `EnCodec/SeaNet{Block,Encoder,Decoder}` vs `Mimi/MimiSeanet{Encoder,Decoder}` (generic/config-driven vs fixed 4-stage unsafe-pointer); `VibeVoice/SConv1d`/`SConvTranspose1d` vs Mimi's streaming convs (different cache data structures for different constraints — multi-speaker dictionary cache vs single-sequence positional array).

**Real doc/content bugs fixed opportunistically (not logic, already committed as comment fixes):** stale `<see cref>` to nonexistent `DecodeFromCodes` removed; `NeuCodecResidualUnit` cref corrected to `ResidualUnit`; unbalanced `<para>` tags fixed in 3 files; `MimiConfig.cs` stale cross-reference corrected; `SConv1d.cs`/`SConvTranspose1d.cs` docs corrected to state the true call path (they call `IBackend.Conv1d`/`ConvTranspose1d` directly, not the dead `VibeVoiceOps` twins above).

## HartsyInference.Audio (Pipelines/Frontends/Io/Layers/Phonemizer/Preprocessing/Sampling/Streaming/Dsp/Cache)

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `Pipelines/F5TtsPipeline.cs` | `SmokeForward` | Public, zero call sites anywhere including tests. | High |
| `Pipelines/KyutaiSttPipeline.cs` + `Models/Kyutai/MoshiTtsGenerator.cs` + `Models/Kyutai/MoshiDepformer.cs` | `ArgMax(ReadOnlySpan<float>)` | Byte-identical private helper reimplemented 3x. | High |
| `Pipelines/MoonshinePipeline.cs` vs `MoonshineStreamingPipeline.cs` | `ArgMax`, `GreedyDecode`, `TranscribeWav`, dispose boilerplate | Near-identical decode loop duplicated. | High |
| `Pipelines/DiaPipeline.cs` (`LoadAny`) vs `Pipelines/MeloTts.cs` (`LoadCheckpoint`) | checkpoint loader | Identical safetensors-vs-pickle selection logic, differs only by one bool. | High |
| `Pipelines/CosyVoicePipeline.cs` | `Synthesize`/`SynthesizeStream` | Byte-identical 13-line `HighPrecisionGemm` save/restore block duplicated in-file. | High |
| `Pipelines/CsmPipeline.cs` | `BuildContext(frames)` loop | Unreachable in practice — only call site always passes empty list. | Medium |
| `Pipelines/HeartMulaPipeline.cs` | `DebugFrameLogits` + helper | Zero call sites; self-documented as debug-only, may be intentional. | Medium |
| `Layers/BiLstm.cs`, `Layers/Gru.cs`, `Layers/UnidirectionalLstm.cs` | `ZeroAllocate`, `LoadTimestep`, `StoreOutputHalf`/`StoreTimestep` | Byte-identical/near-identical RNN step-loop scaffolding duplicated 3 ways. | High |
| `Layers/GruCell.cs` (`GruOps`) vs `Layers/LstmCell.cs` (`LstmOps`) | `SigmoidS` | Byte-identical stable-sigmoid helper; natural home is `Layers/Activations.cs`. | High |
| `Models/Hubert/Hubert.cs`, `Models/Vits/VitsStochasticDurationPredictor.cs`, `Models/Moonshine/MoonshineOps.cs` | `ExactGelu`/`Gelu`+`Erf` | Reimplement the exact erf-GELU math already shared via `Layers/Activations.cs` (used by 13+ other call sites). | High |
| `Dsp/NsfVocoderDsp.cs` | `ForwardStftMagPhase` vs `ForwardStftRealImag` | Identical pad/frame/window logic duplicated in-file, differ only in the final step. | Medium-High |
| `Preprocessing/KaldiFbankExtractor.cs`, `Preprocessing/MelSpectrogramExtractor.cs` | `NextPow2` | Identical private helper, no shared home exists yet. | High |
| `Preprocessing/MelSpectrogramExtractor.cs` vs `Models/CosyVoice/S3GenReference.cs` | `ReflectPad`/`ReflectIndex` | Same reflect-101 padding convention reimplemented; S3Gen's copy has an edge-case guard the other lacks — reconcile before merging. | High |
| `Phonemizer/Espeak/EspeakTranslator.cs` | `_expectVerb`, `_toneNumbers`, `_loptSuffix` | Set once to fixed constants, never mutated — makes 3 conditional branches permanently dead (likely unfinished non-English hooks). | High |
| `Phonemizer/Espeak/EspeakPhonemeList.cs:64` | `if (wordStartIndex < list.Count) _ = wordStartIndex;` | True no-op statement, vestigial. | High |
| `Cache/AudioModelCache.cs` | `Get(string,string,string,string)` sync wrapper | Zero internal callers (vs 35 for `GetAsync`); may be used by an out-of-repo SwarmUI extension. | Medium |
| `Frontends/GptSoVitsSymbols.cs` | `ExpandBertToPhonemes` | Only a unit-test caller — not yet wired into a production pipeline (unfinished feature, not dead). | Medium |
| `Pipelines/YuePipeline.cs`, `Pipelines/ZipVoicePipeline.cs` | `Peak`/`Rms` helpers | Low-value 3-5 line duplication; `LoudnessNormalizer` exists but doesn't expose raw RMS in a compatible shape. | Medium |

**Real bug found opportunistically (not dead code, zero changes made):** `Pipelines/DemucsPipeline.cs`'s `Dispose()` does not call `_model.Dispose()` — a potential resource leak.

## HartsyInference.Cli

| File | Symbol | Reason | Confidence |
|---|---|---|---|
| `src/HartsyInference.Cli/Dispatch/GenerationDispatch.cs:615` | `ToEncoderFrames(...)` | Zero call sites repo-wide; `VideoOutputWriter.Write` takes raw byte arrays directly instead — leftover from an earlier encoder path. | High |
| `src/HartsyInference.Cli/Commands/SpeechCommand.cs`, `MusicCommand.cs`, `RestoreCommand.cs`, `TextCommand.cs`, `FxEnhanceCommand.cs` | Manual `if (settings.X is { } v) parameters.Put(...)` blocks | `CommandRunner.PutIfSet(this ParamState, string, IFormattable?)` already exists and is used by `ImageCommand`/`VideoCommand` for the same purpose; these five reimplement it inline. | High |
| `PreviewCommand.cs:51-56` (`Decode`), `Repl/ReplSession.cs:212-214` (`RenderPreview`), `Dispatch/GenerationDispatch.cs:891-898` (`LoadImage`) | PNG/BMP decode dispatch | Triplicated identical `path.EndsWith(".bmp") ? BmpEncoder.Decode : PngDecoder.DecodeFromFile` dispatch — one shared `ImageIo.DecodeFile(path)` helper would collapse all three. | Medium-High |
| `Commands/PullCommand.cs` (`PullCatalogAsync`) and `Infra/ModelAcquisition.cs` (`EnsurePresent`) | Missing-asset listing + progress-bar download block | Copy-pasted near-verbatim (`PullCommand`'s own comment acknowledges the mirroring); ~40 duplicated lines could hoist into one helper parameterized on {confirm, onMissing}. | Medium |
| 13 `Commands/*.cs` `Settings` classes | `-b\|--backend` / `-o\|--output` / `-q\|--quiet` options | Redeclared identically in 13 separate Settings classes; a shared base `GenerationCommandSettings` would remove ~9 lines × 13 files. Bigger structural change than the others — lower priority. | Medium |
| `Repl/ReplSession.cs:309-341` (`Generate`) vs `Commands/CommandRunner.cs:13-65` (`Run`) | Both implement resolve→dispatch→present→save→catch for one generation | Partial overlap only (REPL intentionally skips engine-construction since it keeps one long-lived engine) — candidate for a smaller shared "dispatch+present+save" helper, not a full merge. | Medium (partial) |
