# Dead-code audit — resolution log

Running resolution of every `DEAD_CODE_CANDIDATES.md` entry (branch `engine-cleanup`). Verdicts:
`DEAD-REMOVE` (deleted) · `DEAD-KEEP` (kept, justified) · `CONSOLIDATE` (hoisted to a shared helper) ·
`BLOAT-STRIP` (inert member/branch removed) · `BUG-FIX` · `DEFER` (tracked in ROADMAP) ·
`NOT-A-FINDING` (verified deliberate or the audit claim was wrong — recorded so it isn't re-flagged).

## Core

| Entry | Verdict | Notes |
|---|---|---|
| `RopeScaling.YarnLogMultiplier` | DEAD-REMOVE | Audit's "wire into InferMscale" framing was WRONG — would double-apply the DeepSeek mscale (already folded into `MlaConfig.AttnScale` via the direct metadata read, `GgufConfigFactory.cs` deepseek2 branch). Property + parse init deleted. Non-deepseek yarn archs never carry the key in practice (llama.cpp reads it only for deepseek2). |
| `DeviceKind.Cuda0`/`Vulkan0` | DEAD-REMOVE | Redundant with the `Cuda(int)`/`Vulkan(int)` factories (4 live uses). |
| `DeviceKind.IsVulkan` | DEAD-KEEP | API symmetry with the live IsCpu/IsCuda/IsGpu trio. |
| `ISttPipeline`/`ITtsPipeline`/`IPipelineRequest` | DEAD-REMOVE | Plus orphaned `TranscriptionResult`/`TranscriptionSegment` (5 files). Same precedent as the deleted `IDiffusionPipeline`. Sibling interfaces (IDetection/IEmbedding/IUpscale/IVision) checked — all live. |
| `TensorPool` | DEAD-KEEP | AGENTS.md/CLEANUP.md prescribe it as the hot-path pattern; why-comment added; ROADMAP tracks adopt-or-remove. |
| `Int8ConvRotCodec`/`Nvfp4ResidentCodec`/`Nvfp4Linear` `ToBf16` + 3 inline in `Tensor.ConvertRange` | CONSOLIDATE | → `TensorCasts.F32ToBf16Bits`. Audit undercounted (4 copies + 3 inline, not 3). Nvfp4 dequant cores left separate — the parity test treats `Nvfp4Linear` as the reference. |
| `Fsq.Quantize/CodesToIndices/Dequantize` inline basis loops | CONSOLIDATE | → existing `Fsq.Basis`. |
| `Mb(long)` ×5 | CONSOLIDATE | → `ByteFormat.Mb`/`MbF1` (audit said ×4 identical; actually ×5 with the Vulkan one format-different — preserved via MbF1). |
| `BlockStreamingScope.EnumerateStreamedWeights`/`BlockBytes` | DEAD-REMOVE | Ctor param cascade removed too. |
| `MixContract.ValidateGeometry` vs `SplitContract.ValidateTensorGeometry`; IBackend Cfg*Step preambles; interleaved-RoPE family | PENDING | Deferred to a later batch (numerics-heavy; needs parity-gated pass). |

## Cpu / World

| Entry | Verdict | Notes |
|---|---|---|
| `ComputeThreadPool` + `NumaAffinity` | DEAD-REMOVE | Self-declared stubs, zero call sites, unreachable catch confirmed. |
| `AudioKernels.Fft` vs `Stft` butterfly | CONSOLIDATE | → `FftInPlaceComplex`, identical op order (bit-identical output); also removes Stft's per-frame duplicated code. |
| MG2 `AssembleInput` `_ = src` | BLOAT-STRIP | Confirmed pure discard. |
| MG2/MG3 `SliceFrame`/`ConcatFrames`/`ActionRows` | CONSOLIDATE | → `MatrixGameOps` (MG3's general forms; MG2 keeps null-handling + VAE-rate scaling wrapper). |

## Cuda

| Entry | Verdict | Notes |
|---|---|---|
| `cuModuleLoadData`, `cuMemcpyPeer` (sync), `cuEventQuery`, `CU_EVENT_DEFAULT`, `cuDeviceGetGraphMemAttribute` + 2 consts | DEAD-REMOVE | `cuDeviceGraphMemTrim` untouched (live). |
| `CudaGraph.TryUpdate` | DEAD-REMOVE | + orphaned `cuGraphExecUpdate`; self-marked untested; production re-captures instead. Class doc updated. |
| `CudaProfilerControl` | DEAD-KEEP | Documented nsys capture-range workflow; would orphan cuProfilerStart/Stop. |
| CudnnConv/CudnnSdpa plan search | CONSOLIDATE + BUG-FIX | NEW BUG found in verification: Conv's TryPlan leaked the plan descriptor on a workspace-query throw (Sdpa had the guard). Fixed, then hoisted to `CudnnPlanSearch` (parameterized by workspace cap); byte-identical SetAttr/Check moved to `CudnnApi`. |
| Fp8/Fp4/Int8 Gemm executor lifecycle | NOT-A-FINDING | Ctors differ meaningfully (SM gates vs failure-tolerant create); classes public+sealed; hoisting ~10 lines of idiomatic Dispose needs an inheritance layer for no behavior gain (CLEANUP.md: invalid motivation). |
| `LtGemmExecutor` subsuming Int8/Fp8 | DEFER | Audit's own assessment (Fp4 block-scale attrs not exposed); ROADMAP. |
| Unused `CU_DEVICE_ATTRIBUTE_*` | DEAD-KEEP | Native-enum-surface mirrors (audit agrees). |

## Vulkan

| Entry | Verdict | Notes |
|---|---|---|
| `VkApiVersion10/11/12` | DEAD-REMOVE | |
| `CastToF16` second branch | BLOAT-STRIP | Unreachable confirmed (first condition strict superset). |
| `PushSet`/`WriteSet` | CONSOLIDATE | → private `BindStorageBuffers` (delivery call parameterized). |
| `DispatchMatmulBatched` inert params + batch>1 throw | BUG-FIX | 2D-b batches = one flattened GEMM; 3D-b = per-slice `DispatchMatmulWithOffsets` loop. Live 3D-b callers exist (SamMaskDecoder, ChromaRadianceNerfHead). New GpuIntegration test vs scalar reference passes on hardware. |
| Flash/naive SDPA output-cast tails | CONSOLIDATE | → `CacheOutputCastingFrom` (also used by the BatchedMatMul fix). |
| Tiled-matmul/Conv2D/WithOffsets spec+push-constant builders ×3 | DEFER | Real but the three arrays differ in the bias/activation/residual tail slots; a shared builder needs ~5 params for ~13 lines — flagged, low value now. |

## ModelAssets (+ Engine recipe call sites)

| Entry | Verdict | Notes |
|---|---|---|
| `RemapQwenLanguageKey`/`RemapQwenKey` ×7 | CONSOLIDATE | → `CheckpointConvertUtils.RemapQwenLanguageKey` (two textual dialects, zero behavioral diff — `Contains(string)` is ordinal). |
| `StripTransformerPrefix` ×3 (+4 identical Engine recipe copies found during implementation) | CONSOLIDATE | → `CheckpointConvertUtils.StripTransformerPrefix` — 7 total. |
| `SplitQkvWeight`/`SplitQkvBias` ×5 | CONSOLIDATE + latent BUG-FIX | Audit's suggested target (`SplitInProjWeight`) was the naive non-quant-aware form and would have corrupted quantized Flux checkpoints. Consolidated on Flux2's `SliceByteCount` form instead (quant-aware + block-alignment throw); Lens/QwenImage/Sd3 were silently wrong for block-quantized dtypes, and every bias path (incl. Flux2's own) was — both fixed. `SplitInProjWeight` (Sdxl) untouched. |
| `SwapScaleShiftHalves` ×5 | CONSOLIDATE | One helper; HunyuanVideo's F32 force-cast via `castToF32` (Fp8ScaleFactor deliberately not propagated there — cast folds it in); even-dim + quant-alignment validation now uniform. |
| LoRA `GroupBuffer` ×5 | CONSOLIDATE (×3) + NOT-A-FINDING (×2) | Audit said byte-identical ×5; actually 3 shapes. KohyaFlux/DiffusersFlux/KohyaSd → shared `LoraGroupBuffer` (warning text per-mapper via delegate). AiToolkit (deliberate alpha=rank — folding would change LoRA strength) and Wan (extra roles, no Target) stay bespoke. |
| `DiscoverShards`/`LoadShards`/`LoadShardsRemap` ×2 | CONSOLIDATE | Error text parameterized; nullable key-map subsumes the remap variant. |
| GGUF `Codec_Q4_0/Q4_1/Q5_0/Q5_1` similarity | NOT-A-FINDING | Genuinely different bit-layout math per format (audit agrees). |

## Diffusion — batch A (Adapters/Pipelines/Schedulers + Denoisers dead code)

| Entry | Verdict | Notes |
|---|---|---|
| `NumControlTypes`, `UseShortcut` | DEAD-REMOVE | Backing fields live, kept. |
| Adapter `EnsureF32` ×7 | CONSOLIDATE | → `TensorCasts.EnsureF32`. Audit's suggested home (`DtypeCastHelper`) was WRONG for these — it disposes the source and needs a backend; a verbatim swap would free loader-owned weights. |
| `ProjectResidual` ×2 | CONSOLIDATE | QwenImageControlNet delegates to FluxControlNet's (now internal). |
| `SumBytes` ×5 | CONSOLIDATE | → `Utilities/WeightBytes`. Two more byte-identical copies found out of scope (FluxTransformer.cs:241, Video/MiniMaxH3Pipeline.cs:73) — queued for later batches. |
| Flux↔Chroma `PackLatent`/`UnpackLatent` | CONSOLIDATE | Chroma's dispose-on-pack ownership moved to explicit call-site disposals (audit missed this load-bearing asymmetry; a naive swap would leak or double-dispose). |
| Scheduler `AddNoise` ×4; Lcm/Tcd `ComputeAlphasCumprod` | CONSOLIDATE | → `NoiseSchedule` (the shared home Ddim/Dpm already used). Beta-loop float-op order now matches the shared path — last-ulp drift vs the old private loops, inside every tolerance. `SetTimesteps` left per-scheduler. |
| `FluxToolsConfig`/`FluxToolVariant`/`FromWeights` + `Flux1Tools`/`Flux1Fill` cascade | DEAD-REMOVE | |
| AceStepDit/AceStep15Dit nested Pick/EnsureF32 dupes; StableAudio Pick trio ×3 | CONSOLIDATE | One set each. |
| DebugDump `Enabled` ×5 | DEAD-REMOVE | AuraFlow/Kandinsky5/WanVideo's are live, kept. |
| `V1Turbo`/`Texture` presets; DiTUtils dead trio; ErnieImageRope + FLiteRope host rotation paths; `CachedSeqLen`; `ToBchw`; OasisDit `ElapsedMs`+`tsEntry`; `ApplyModulation`; `AddInPlace`; write-only fields (AuraFlow blocks, CrossAttentionBlock incl. `_numHeads`, DownBlock); HiDream `_ = inShape` | DEAD-REMOVE / BLOAT-STRIP | `PadLastDim`/`ConcatPooled` were dead in DiTUtils but had live private duplicates in Sd3/HiDream pipelines — pipelines migrated onto the DiTUtils forms. `ApplyModulation` doc references reworded (class itself heavily live). |
| `HunyuanVideoDit.OnBlockOutput` | DEAD-REMOVE | Never assigned repo-wide incl. tests (checked); SeedVr2/LtxVideo2's analogous hooks ARE used and stay. |
| Known pre-existing failure | — | `MiniMaxH3AssetsTests.FlatLayout_PrefersLoadableFormatsOverSmallerUnsupportedOnes` fails on pre-edit DLLs too (selection logic in Engine's MiniMaxH3Recipe.cs, untouched). |

## Diffusion — batch B (Denoisers consolidations)

| Entry | Verdict | Notes |
|---|---|---|
| `LoadF32` ×26 + `LoadAsF32` ×5 + DiTUtils.LoadF32's 11 callers | CONSOLIDATE | All onto `TensorCasts.LoadF32`; `DiTUtils.LoadF32` retired. Audit said ×9 — real count 3.5× higher. `LoadF32Opt` siblings preserved; 5 more unlisted `LoadAsF32` + 2 `LoadF32In` handled in batch C. |
| DebugDump plumbing ×21 | CONSOLIDATE (×19) + NOT-A-FINDING (×1) | → `DebugDumpSink` (cached-dir + per-call modes; AceStep's per-call parity rationale preserved verbatim; WanVideo/Anima/QwenVae quirks kept). OmniGen2DebugDump genuinely divergent — left. |
| Chroma `NormModulate` ×2; HiDream `GpuModulate`; Flux/Flux2 `LayerNormNoAffine`; OmniGen2 patchify pair; `MoveAcross`/`CopyAcross` | CONSOLIDATE | Onto DiTUtils (each verified dtype-identical first; `UnpatchifyToNCHW` gained `negate:`, non-negate path bit-identical). |
| `SliceModRows`, `GetOrFakeOnes`, MatrixGame `SliceFrames`/`SliceRows`/`CloneCpu`, `ConcatAlongSeqDim` (Lumina2/ZImage), ZImage `SplitQkv`, Wan encoder ops, Krea2TextFusion SwiGlu | CONSOLIDATE | Family-level hoists (incl. new `WanEncoderOps`). |
| `Kandinsky5Block.NormModulate` (forces F32), `Krea2Block.SwiGlu` (DitDtype.Act + F16-overflow rationale), HiDream norm+modulate pairs (F32-pinned), ZImage `ForwardSwiGlu` (10-param de-instancing) | NOT-A-FINDING | Verified real behavioral deltas — deliberately not merged. |
| FLite ones-caches; Krea2-vs-QwenImage patchify | NOT-A-FINDING | Audit's own medium-confidence flags; confirmed not worth merging. |

## Diffusion — batch C (Vae/Music/TextEncoders)

| Entry | Verdict | Notes |
|---|---|---|
| `EnsureF32`/`AsF32`/`F32` one-liner wave (~20 claimed, ~44 found) | CONSOLIDATE | → `TensorCasts.EnsureF32`; `TextEncoderTensorHelpers.CastToF32IfNeeded` retired and its LlamaStyle/Gemma4 callers migrated. Ownership-signalling variants deliberately NOT folded: `AsF32(Tensor, out bool owned)` (AudioWeightNorm, MiniMaxH3AudioVaeDecoder), the ledger form (MiniMaxH3VideoVaeWeights), nullable in/out (OmniGen2Transformer) — verified untouched. |
| Vae `Bias()` ×9, `Clone`/`CloneRef`/`CloneTensor` ×6, `SliceChannels` ×3 | CONSOLIDATE + BUG-FIX | → new `Models/Vae/VaeOps`. **New bug fixed:** all 6 clone copies hardcoded `n * 4` bytes while allocating at the source dtype — silently wrong for any non-F32 input; now uses `DType.ComputeByteCount`. |
| `LtxVideoVaeEncoder.ChannelRms` | CONSOLIDATE | → the already-shared `LtxVaeResnetBlock3d.ChannelRms`. |
| `LtxVideo2VaeDecoder.Reverse(int[])`, `MiniMaxH3VideoVaeDecoder.Sum(int[])` | DEAD-REMOVE | The `Reverse(bool[])` twins are live in both decoders — untouched. |
| `QwenImageVaeOps.FlattenGamma` no-op `if` | BLOAT-STRIP | Only the dead branch removed; the ownership asymmetry underneath is DEFER (ROADMAP). |
| QwenImage Vae `Downsample` stale summary | BUG-FIX (doc) | Claimed symmetric pad=1; Forward actually uses asymmetric `PadRightBottom`. |
| Mage `GlobalAvgPool`/`ScaleByChannel`/`ChannelLayerNormAffine`/`TimestepEmbedZero` | CONSOLIDATE | → new `Models/Vae/Mage/MageVaeOps`. |
| Batch-B stragglers (`LoadAsF32` ×5, `LoadF32In` ×2, FluxTransformer `SumBytes`) | CONSOLIDATE | Onto TensorCasts / WeightBytes. |
| `LoadF32Opt` ×5 (nullable TryGetValue variant) | CONSOLIDATE | Folded into `TensorCasts.LoadF32Opt` as a follow-up within this phase. |
| Remaining inline `t.DType == F32 ? t : CastTo` expressions inside larger methods | NOT-A-FINDING | Not duplicate helper definitions; migrating them is cosmetic. |

## LLM (pre-execution correction)

**The audit's `Relabel` → `RelabelRank2ToPyTorchOrder` migration is WRONG and must not be done as described.** Verified by reading both:

- private `Relabel(Tensor)` (9 copies): allocates a new **F32** tensor, `Buffer.MemoryCopy`s the data, returns an **owned copy** the caller disposes; byte math is F32-hardcoded.
- `GgufModelLoader.RelabelRank2ToPyTorchOrder(dict)`: **dict→dict**, returns `Reshape` **views still borrowing the GGUF mmap** — pure metadata swap, no copy, no ownership transfer, dtype preserved.

Redirecting the 9 call sites at it would hand callers borrowed mmap views they then dispose (use-after-free / double-free class), change dtype behavior, and doesn't even match the signature. Correct action for Phase 5: delete Qwen35Model's dead copy, then hoist ONE shared helper preserving the owned-F32-copy contract (Core/Tensors is reachable from LLM). The quant-safety comment in Qwen35Model refers to GGUF dict loading, a different code path.

## Audio — batch A (dead-code deletion)

Verified against `dotnet build src/HartsyInference.Cli` (0 warnings / 0 errors) and both audio suites at their
baselines: `HartsyInference.Audio.Tests` 240 passed / 5 pre-existing failures / 2 skipped, and
`HartsyInference.Audio.Phonemizer.Tests` 22/22.

| Entry | Verdict | Notes |
|---|---|---|
| Kyutai `MoshiDepthTransformer`/`MoshiDepthConfig`/`MoshiDelay` + `KyutaiTtsModel`/`KyutaiTtsConfig`/`KyutaiTtsPipeline` | DEAD-REMOVE | 6 files. Self-closed cluster superseded by `MoshiDepformer`+`MoshiTtsGenerator`; production `KyutaiTtsModel` is the unrelated `Engine/Audio/Tts/Models` type. Two live doc crefs (`MoshiDepformer`, `Qwen3MtpCodePredictor`) reworded to prose, keeping the why (the old type assumed a Qwen2-style per-(set,layer) layout the checkpoint doesn't use). |
| `MusicGenDecoder.Forward`/`EmbedFrames`/`ForwardStep` + transitive privates `HeadLogits`/`EmbedFrame`/`BuildCausalMask`/`SliceLast`; `MusicGenBlock.Forward` | DEAD-REMOVE | Pipeline uses only the graph-batched path. `MusicGenBlock.ForwardStep`/`PrimeCross` are LIVE (called from `RunBatchedIntoFixed`) — untouched. Class docs rewritten to describe the surviving path. |
| `VibeVoiceOps` `CausalConv1d`/`CausalConvTranspose1d`/`RmsNormChannelsFirst`/`LayerScaleApplyCF`/`AdaLnModulate`/`AdaLnGatedAdd` | DEAD-REMOVE | Host twins of live GPU ops. Class-doc bullet list dropped with them; `SConv1d`/`SConvTranspose1d` docs reworded (they carried the only crefs). `GetExtraRightPadding` kept (live via `SConv1d`). |
| `F5Ops.GroupedConv1D`; `F5TimestepEmbedding.LastTimeEmb`; `MoonshineTokenizer.DecodeOne`; `WhisperOps.TransposeMatrix`/`AddBiasBroadcast`/`Expand2DMaskTo4D`; `PocketTtsFlowLm.GenerateLatents`/`DebugPrimedHidden`; `Qwen3TtsTalker.AddCodecEmbed`; `F5TtsPipeline.SmokeForward`; `AudioModelCache.Get` (4-arg sync) | DEAD-REMOVE | `LastTimeEmb` was written+never disposed each `Forward`, so removing it also drops a small per-call leak. `SmokeForward`'s doc claimed an integration-test caller that does not exist. |
| `Moonshine/RotaryEmbedding` `RopeScaling` record + `Llama3` preset + 4-arg `GetTables` | NOT-A-FINDING | **The audit's "zero callers repo-wide" claim is wrong.** `tests/HartsyInference.Audio.Tests/RotaryEmbeddingTests.cs` is a dedicated 3-test class exercising all three (lines 18, 28, 48, 52). Skipped; deleting would have silently dropped 3 tests. |
| `RvcRmvpeConfig.GruHidden`; `ResembleEnhanceConfig` `DenoiserHiddenDim`/`DenoiserNumBlocks`/`DenoiserNumMiddleBlocks`/`DenoiserKernel`/`UnivNetNc`; `StyleHifiGanGenerator.UpsampleKernels`; `StyleEncoder._invSqrt2` (outer) | DEAD-REMOVE | All write-only. `OpenVoiceSpeakerConfig.GruHidden` is a different type and IS read — untouched; `ResBlk2D`'s own `_invSqrt2` (same file) is live. |
| `EspeakTranslator` `_toneNumbers`/`_loptSuffix`/`_expectVerb` + their 3 permanently-dead branches | BLOAT-STRIP | Set once to constants, never mutated. Checked the escape hatch: `EspeakVoiceVariant.Parse` (the real per-language extension point) parses `phonemes`/`dictrules`/`reduce_t`/`replace` and has **no** parsing for any of these three, so there was no partially-wired path to drop. Collapsed to the reachable side with a one-line note at each site recording that espeak's `tone_numbers`, `lopt[LOPT_SUFFIX]` and `expect_verb` are not ported. `_wordVowelCount`/`_wordStressedCount` survive (other reads at ~636/642). Phonemizer suite 22/22 unchanged. |
| `EspeakPhonemeList:64` `if (wordStartIndex < list.Count) _ = wordStartIndex;` | BLOAT-STRIP | True no-op; the now-orphaned local removed with it. |
| `SeaNetBlock`/`SeaNetDecoder` private `GetExtraRightPadding` (×2); `EnCodecConvPad.PaddedConv`'s `eluUnused` param | DEAD-REMOVE | Only `SeaNetEncoder`'s copy is ever called. No call site passed `eluUnused` explicitly. |
| `F5Ops.Grn` dead first accumulation loop | BLOAT-STRIP | A full O(dim·t) pass accumulated `sq` then discarded it via `_ = sq;` under a comment block narrating an abandoned design; the next block recomputes the identical accumulation into `gx[d]` and uses it. Verified side-effect-free (reads `xp` only). Roughly halves the method's cost. |
| `CosyVoiceFlow.InferenceChunk`, `InferenceGrowing` (+ transitively-dead `TrimTrailingFrames`) | DEAD-REMOVE | Zero call sites, but ~12 live doc crefs (in `ConditionalCfm`, `CosyVoicePipeline`, and `InferenceGrowingWindowed`'s own remarks) — all reworded to prose. **Both methods' measurements preserved first** in `MODEL_STATUS_AUDIO.md` under a new `### CosyVoice 2` section: InferenceChunk's exposure-bias mel-level drift series and InferenceGrowing's O(n²) disqualification + its role as the discriminator for the "quick"→"quit" artifact. The settling-margin measurement (~0.1–0.6 non-decaying floor, 1.6–4.5 near the live edge) was moved into `InferenceGrowingWindowed`'s remarks because it is the empirical basis for that live method's `marginFrames` parameter. `SliceFrames` kept (live). |
| `MusicGenConfig` `Melody`/`StereoSmall`/`StereoMedium`/`StereoLarge` | DEAD-REMOVE (trap) | Not merely unused — **unreachable and wrong.** The only loader, `MusicCatalog.InferMusicGenSize`, dispatches solely on decoder width 1024/1536/2048 → Small/Medium/Large, so no checkpoint can ever select them; only `facebook/musicgen-{small,medium,large}` + `audiogen-medium` are registered. `MusicGenConfig.AudioChannels` has zero readers repo-wide, so a "stereo" preset would not produce stereo even if constructed, and `Melody` is field-identical to `Medium` while naming chroma conditioning whose `NumChroma`/`ChromaLength` also have zero readers. |
| Bark `BarkCausalStage.ForwardLogits`, `BarkFineModel.DebugLogits`; Demucs `HtDemucs.DebugHook` + its 6 probe delegates; `HeartMulaPipeline.DebugFrameLogits` | DEAD-KEEP | **The audit's "scaffolding for a test that was never written" reading is backwards.** Each is the C# half of a *live* Python oracle whose C# xunit driver was removed in the test-suite cleanup (same `~~BarkParityTests~~` strikethrough convention in PARITY_VERIFICATION.md): `bark_reference/dump_bark_reference.py` names `BarkCausalStage.ForwardLogits` by name; `DebugLogits` is cited in PARITY_VERIFICATION.md as carrying the inclusive-sum fix; `demucs_reference/dump_demucs_stages.py` dumps `spec_cac`/`enc0_conv`/`enc0_postdconv`/`ct_in_x`/`ct_l0_attn`/`ct_l0_afterattn`/`enc{i}`/`dec{i}` — the DebugHook's key set, one for one; `dump_heartmula_lm_reference.py` dumps exactly `DebugFrameLogits`' `c0_logits`/`dec_logits` pair. One-line oracle-pointer remarks added to each so they aren't re-flagged. (`RvcRmvpe.DebugHook`, wired by `RmvpeParityTests`, was the audit's own contrast case — the difference is a deleted driver, not missing intent.) |

(Later phases appended as they land.)
