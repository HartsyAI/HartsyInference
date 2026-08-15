using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Loads PTX modules from disk; exposes typed kernel launches via nint function handles for zero-alloc dispatch.</summary>
public sealed class CudaKernels : IDisposable
{
    // ── F32 Modules ──────────────────────────────────────────────────────
    private readonly CudaModule _elementwiseModule;
    private readonly CudaModule _groupnormModule;
    private readonly CudaModule _layernormModule;
    private readonly CudaModule _spatialModule;
    private readonly CudaModule _im2colBandedModule;
    private readonly CudaModule _maxpool2dModule;
    private readonly CudaModule _depthwiseConv2dModule;
    private readonly CudaModule _msdaModule;
    private readonly CudaModule _softmaxModule;
    private readonly CudaModule _transposeModule;
    private readonly CudaModule _wanRopeModule;
    private readonly CudaModule _wanVaeFramesModule;
    private readonly CudaModule _wanVaeConv3dModule;
    private readonly CudaModule _wanVaeNormModule;
    private readonly CudaModule _gegluModule;
    private readonly CudaModule _broadcastAddModule;

    // Optional: across-step feature-cache gate reductions (stepcache.ptx). Null when the PTX has not been
    // compiled on this install yet — DeviceFeatureCache gates on HasStepCacheKernels and stays off without it.
    private readonly CudaModule? _stepCacheModule;
    private readonly nint _stepCacheRelL1F32;
    private readonly nint _stepCacheRelL1F16;

    // Optional: LTX-2.5 NA diffusion-decoder kernels (ltx25_na_decoder.ptx, src/HartsyInference.Cuda/Kernels/ltx25vae) —
    // 3D neighborhood attention + its 3-axis interleaved rope. Null when the PTX has not been deployed; the
    // backend then falls through to the managed reference rather than failing.
    private readonly CudaModule? _ltx25NaModule;
    private readonly nint _ltx25Na3dF32;
    private readonly nint _ltx25Na3dTiled8x8F32;
    private readonly nint _ltx25Na3dTiled4x8F32;
    private readonly nint _ltx25NaRope3dF32;
    private readonly nint _ltx25PixelShuffleF32;

    // Optional: W8A8 IMMA chain kernels (w8a8.ptx, src/HartsyInference.Cuda/Kernels/dequant) — per-row INT8 activation quant +
    // int32→F16/F32 dequant epilogue around Int8GemmExecutor. Null when not compiled.
    private readonly CudaModule? _w8a8Module;
    private readonly nint _w8a8QuantRowwiseF16;
    private readonly nint _w8a8QuantRowwiseF32;
    private readonly nint _w8a8DequantBiasF16;
    private readonly nint _w8a8DequantBiasF32;
    private readonly nint _w8a8DequantBiasStridedF16;
    private readonly nint _w8a8DequantBiasStridedF32;

    // Optional: ConvRot activation rotation (convrot.ptx, src/HartsyInference.Cuda/Kernels/dequant) — the x @ H a
    // ComfyUI int8_tensorwise+convrot weight owes its GEMM. Null when not compiled.
    private readonly CudaModule? _convRotModule;
    private readonly nint _convRotRotateF16;
    private readonly nint _convRotRotateF32;
    private readonly nint _convRotQuantF16;
    private readonly nint _convRotQuantGatedF16;
    private readonly nint _convRotQuantF32;
    private readonly CudaModule? _int8MmaModule;
    private readonly nint _int8MmaGemmF16;
    private readonly nint _int8MmaGemmF16Pad;

    // Optional: NVFP4 packed-weight dequant (dequant_nvfp4_to_f16.ptx, src/HartsyInference.Cuda/Kernels/dequant) — the
    // per-GEMM unpack that lets a ComfyUI nvfp4 checkpoint stay resident at 0.5 byte/param. Null when not compiled.
    private readonly CudaModule? _nvfp4Module;
    private readonly nint _nvfp4DequantF16;
    private readonly nint _nvfp4DequantBf16;

    // Optional: SageAttention-v1 INT8 flash attention (sage_attn_int8.ptx, src/HartsyInference.Cuda/Kernels/attention). Null when
    // not compiled — CudaBackend.ScaledDotProductAttention gates on HasSageAttentionKernels and falls through.
    private readonly CudaModule? _sageAttnModule;
    private readonly nint _sageKMeanF32;
    private readonly nint _sageQuantQInt8F32;
    private readonly nint _sageQuantKInt8F32;
    private readonly nint _sageAttnInt8F32;
    private readonly nint _sageKMeanF16H;
    private readonly nint _sageQuantQInt8F16H;
    private readonly nint _sageQuantKInt8F16H;

    // Optional: the register-resident mma.sync rewrite of the flash stage (sage_attn_int8_v1.ptx) — no
    // Sq%32 restriction (fully row-guarded) and the perf-viable implementation. Preferred over the wmma v0
    // when present; HARTSY_SAGE_V0=1 forces the old kernel for debugging.
    private readonly CudaModule? _sageAttnV1Module;
    private readonly nint _sageAttnV1D128;
    private readonly nint _sageAttnV1D64;
    private readonly nint _sageAttnV1D128F16Acc;
    private readonly nint _sageAttnV1D64F16Acc;
    private readonly nint _sageAttnV1D128F16Io;
    private readonly nint _sageAttnV1D64F16Io;
    private readonly nint _sageVF16T;
    private readonly nint _sageVF16TH;

    // ── F16 Modules ──────────────────────────────────────────────────────
    private readonly CudaModule _elementwiseF16Module;
    private readonly CudaModule _groupnormF16Module;
    private readonly CudaModule _layernormF16Module;
    private readonly CudaModule _spatialF16Module;
    private readonly CudaModule _softmaxF16Module;
    private readonly CudaModule _transposeF16Module;
    private readonly CudaModule _gegluF16Module;
    private readonly CudaModule _broadcastAddF16Module;

    // ── BF16 Modules (only the subset VAE needs; SDXL VAE F16 overflows) ─
    private readonly CudaModule _elementwiseBf16Module;
    private readonly CudaModule _groupnormBf16Module;
    private readonly CudaModule _layernormBf16Module;
    private readonly CudaModule _spatialBf16Module;
    private readonly CudaModule _broadcastAddBf16Module;
    private readonly CudaModule _groupnormSiluBf16Module;

    // ── Fused + Cast Modules ─────────────────────────────────────────────
    private readonly CudaModule _groupnormSiluModule;
    private readonly CudaModule _groupnormSiluF16Module;
    private readonly CudaModule _castModule;

    // ── DiT glue Modules ─────────────────────────────────────────────────
    private readonly CudaModule _ditF32Module;
    private readonly CudaModule? _ditRopeModule;
    private readonly CudaModule? _ditFp8EmitModule;
    private readonly CudaModule _mg3ActionModule;
    private readonly nint _mg3SplitQkvTemporalF32, _mg3MergeTemporalF32, _mg3RopeBatchedF32, _mg3KvExpandF32, _mg3MouseMlpConcatF32;

    // ── Audio conv Module + handles (codec/TTS Conv1d + ConvTranspose1d, F32) ─
    private readonly CudaModule _audioConvF32Module;
    private readonly nint _conv1dF32;
    private readonly nint _convTranspose1dF32;

    // ── Audio activation Module + handles (Sigmoid / Elu / Snake, F32) ───
    private readonly CudaModule _audioActF32Module;
    private readonly nint _audioSigmoidF32;
    private readonly nint _audioMishF32;
    private readonly nint _audioEluF32;
    private readonly nint _audioLeakyReluF32;
    private readonly nint _audioSnakeF32;
    private readonly nint _audioSnakeBetaF32;
    private readonly nint _audioPreluF32;
    private readonly nint _audioRepeatTimeF32;

    // ── Adaptive InstanceNorm 1D Module + handle (Kokoro / StyleTTS 2, F32) ──
    private readonly CudaModule _audioAdain1dF32Module;
    private readonly nint _audioAdain1dF32;

    // ── Language-model (decoder LLM) glue Module + handles ───────────────
    private readonly CudaModule _lmF32Module;
    private readonly nint _lmRepeatKvF32;
    private readonly nint _lmKvAppendF32;
    private readonly nint _lmKvAppendF16;
    private readonly nint _lmGatherRowsF32;
    private readonly nint _lmScatterAddWeightedRowsF32;
    private readonly nint _lmArgMaxLastDimF32;
    private readonly nint _lmArgMaxStage1F32;
    private readonly nint _lmArgMaxStage2F32;
    private readonly nint _lmGluActF32;
    private readonly nint _lmQkvRopeScatterF32;
    private readonly nint _lmQkNormRopeScatterF32;
    private readonly nint _lmRmsNormQ8_1F32;
    private readonly nint _lmAddRmsNormQ8_1F32;
    private readonly nint _lmGluActQ8_1F32;
    private readonly nint _lmNormAddRmsNormQ8_1F32;
    private readonly nint _lmRmsNormAddF32;
    private readonly nint _lmPleGatherQ5KF32;
    private readonly nint _lmSsmConvSplitStepF32;
    private readonly nint _lmSsmL2NormHeadsF32;
    private readonly nint _lmSsmDeltaStepF32;
    private readonly nint _lmSsmDeltaScalarsF32;
    private readonly nint _lmSsmDeltaStepRowsF32;
    private readonly nint _lmSsmDeltaStepWarpRowF32;
    private readonly nint _lmSsmDeltaGatedNormF32;
    private readonly nint _lmAddRmsNormF32;
    private readonly nint _lmRopeInterleavedF32;
    private readonly nint _lmRopeDecodeSplitHalf;
    private readonly nint _lmRopeDecodeInterleaved;
    private readonly nint _lmEmbedGatherDecodeF32;
    private readonly nint _lmHistoryAppend;
    private readonly nint _lmRepetitionPenaltyF32;
    private readonly nint _lmKvSliceTimeF32;
    private readonly CudaModule _flashAttnF32Module;
    private readonly nint _flashAttnF32;
    private readonly nint _flashAttnF16Kv;
    private readonly CudaModule _flashAttnF32SplitModule;
    private readonly nint _flashAttnF32Split;
    private readonly nint _flashAttnF32SplitF16Kv;
    private readonly CudaModule _flashV2Module;
    private readonly nint _flashV2Tf32;
    private readonly nint _flashAttnF32Combine;

    // ── Elementwise F32 function handles ─────────────────────────────────
    private readonly nint _addF32;
    private readonly nint _mulF32;
    private readonly nint _scaleF32;
    private readonly nint _siluF32;
    private readonly nint _geluF32;
    private readonly nint _clampF32;

    // ── Elementwise F16 function handles ─────────────────────────────────
    private readonly nint _addF16;
    private readonly nint _mulF16;
    private readonly nint _scaleF16;
    private readonly nint _siluF16;
    private readonly nint _geluF16;
    private readonly nint _clampF16;

    // ── Elementwise BF16 function handles ────────────────────────────────
    private readonly nint _addBf16;
    private readonly nint _mulBf16;
    private readonly nint _scaleBf16;
    private readonly nint _siluBf16;
    private readonly nint _geluBf16;
    private readonly nint _clampBf16;

    // ── Normalization function handles ───────────────────────────────────
    private readonly nint _groupnormF32;
    private readonly nint _groupnormF16;
    private readonly nint _groupnormBf16;
    private readonly nint _layernormF32;
    private readonly nint _layernormF16;
    private readonly nint _layernormBf16;

    // ── Spatial function handles ─────────────────────────────────────────
    private readonly nint _upsampleNearest2dF32;
    private readonly nint _im2colF32;
    private readonly nint _col2biasAddF32;
    private readonly nint _upsampleNearest2dF16;
    private readonly nint _im2colF16;
    private readonly nint _col2biasAddF16;
    private readonly nint _upsampleNearest2dBf16;
    private readonly nint _im2colBf16;
    private readonly nint _col2biasAddBf16;
    private readonly nint _im2colBandedF32;
    private readonly nint _im2colBandedF16;
    private readonly nint _im2colBandedBf16;
    private readonly nint _maxpool2dF32;
    private readonly nint _maxpool2dF16;
    private readonly nint _depthwiseConv2dF32;
    private readonly nint _depthwiseConv2dF16;
    private readonly nint _msdaForwardF32;

    // ── Softmax function handles ─────────────────────────────────────────
    private readonly nint _softmaxF32;
    private readonly nint _softmaxF16;

    // ── Transpose/Permute function handles ───────────────────────────────
    private readonly nint _transpose2dF32;
    private readonly nint _permute0213F32;
    private readonly nint _transpose2dF16;
    private readonly nint _permute0213F16;
    private readonly nint _wanRopeInterleaved;
    private readonly nint _wanRopeInterleavedPerHead;
    private readonly nint _wanVaeExtractFrame;
    private readonly nint _wanVaeWriteFrame;
    private readonly nint _wanVaeBuildPadded;
    private readonly nint _wanVaeFillBias;
    private readonly nint _wanVaeAccumulateTap;
    private readonly nint _wanVaeExtractFrameBf16;
    private readonly nint _wanVaeWriteFrameBf16;
    private readonly nint _wanVaeBuildPaddedBf16;
    private readonly nint _wanVaeFillBiasBf16;
    private readonly nint _wanVaeAccumulateTapBf16;
    private readonly nint _seedVr2PixelShuffle;
    private readonly nint _seedVr2PixelShuffleBf16;
    private readonly nint _seedVr2PadBr;
    private readonly nint _seedVr2PadBrBf16;
    private readonly nint _wanVaeRmsNormChannel;
    private readonly nint _wanVaeRmsNormChannelBf16;
    private readonly nint _wanVaeUnpatchify;
    private readonly nint _wanVaeUnpatchifyBf16;
    private readonly nint _wanVaeSplitQkv;
    private readonly nint _wanVaeTokensToFrame;

    // ── GeGlu function handles ───────────────────────────────────────────
    private readonly nint _gegluF32;
    private readonly nint _gegluF16;

    // ── BroadcastAdd function handles ────────────────────────────────────
    private readonly nint _broadcastAddF32;
    private readonly nint _broadcastAddF16;
    private readonly nint _broadcastAddBf16;

    // ── Fused GroupNorm+SiLU function handles ────────────────────────────
    private readonly nint _groupnormSiluF32;
    private readonly nint _groupnormSiluF16;
    private readonly nint _groupnormSiluBf16;

    // ── Cast function handles ────────────────────────────────────────────
    private readonly nint _castF32ToF16;
    private readonly nint _castF16ToF32;

    // ── DiT glue function handles (F32) ──────────────────────────────────
    private readonly nint _ditRmsNormF32;
    private readonly nint _ditAffineBroadcastF32;
    private readonly nint _ditGatedResidualF32;
    private readonly nint _ditModulation4F32;
    private readonly nint _ditCfgEulerF32;
    private readonly nint _ditCfgRenormReduceF32;
    private readonly nint _ditCfgRenormFinalizeF32;
    private readonly nint _ditCfgRenormEulerF32;
    private readonly nint _ditCfgNormalizedEulerF32;
    private readonly nint _ditAffineMixF32;
    private readonly nint _ditMaskedAffineMixDenseF32;
    private readonly nint _ditMaskedAffineMixPackedOuterF32;
    private readonly nint _ditMaskedAffineMixPackedInnerF32;
    private readonly nint _ditPatchifyF32;
    private readonly nint _ditPatchifyU16;
    private readonly nint _ditUnpatchifyF32;
    private readonly nint _ditUnpatchifyU16;
    private readonly nint _ditSplitSliceU32;
    private readonly nint _ditSplitSliceU16;
    private readonly nint _ditTanhF32;
    private readonly nint _ditRopeF32;
    private readonly nint _ditRopeHeadMajorF32;
    private readonly nint _ditRopeHeadMajorV2F32;
    private readonly nint _ditAffineBroadcastRowIndexedToFp8F32;
    private readonly nint _ltx2SplitRopeF32;
    private readonly nint _ltx2SplitRopeF16;
    private readonly nint _ltx2QkNormRopeHeadMajorF16;
    private readonly nint _ltx2QkNormRopeTokenMajorF16;
    private readonly nint _ltx2QkNormRopeHeadMajorRopeF16;
    private readonly nint _ltx2QkNormRopeTokenMajorRopeF16;
    private readonly nint _ltx2HeadGateF16;
    private readonly nint _ltx2QkNormRopeHeadMajorF32;
    private readonly nint _ltx2QkNormRopeTokenMajorF32;
    private readonly nint _ltx2HeadGateF32;
    private readonly nint _ltx2RmsModulateF16;
    private readonly nint _ltx2RmsModulateF32;
    private readonly nint _ditSliceLastDimF32;
    private readonly nint _ditRowScaleF32;
    private readonly nint _ditAddScalarF32;
    private readonly nint _ditLayerNormNoAffineF32;
    private readonly nint _ditMoeTopkGateF32;
    private readonly nint _ditRowGatedAccumF32;
    private readonly nint _ditIndexAddF32;
    private readonly nint _ditScatterRowsAfterF32;
    private readonly nint _ditSliceRowsF32;
    private readonly nint _oasisSplitHeadsF32;
    private readonly nint _oasisRopeInterleavedF32;
    private readonly nint _oasisMergeHeadsF32;
    private readonly nint _oasisUnpatchifyF32;
    private readonly nint _oasisAdaLnF32;
    private readonly nint _ditPixelQuantizeF32;
    private readonly nint _triplaneGridSampleF32;
    private readonly nint _gegluErfF32;
    private readonly nint _geluErfF32;
    private readonly nint _convTranspose2dF32;
    private readonly nint _ditConcat2F32;
    private readonly nint _ditLayerNormModulateF32;
    private readonly nint _ditQkvSplitNormF32;
    private readonly nint _ditQkvSplitNormHeadMajorF32;
    private readonly nint _ditQkvSplitNormHeadMajorSubsetF32;
    private readonly nint _fourierEmbedF32;
    private readonly nint _conv3dF32;
    private readonly nint _sparseScatterToGridF32;
    private readonly nint _sparseGatherFromGridF32;
    private readonly nint _rowGatherF32;
    private readonly nint _rowScatterAddF32;
    private readonly nint _ditAffineBroadcastRowIndexedF32;
    private readonly nint _ditGatedResidualRowIndexedF32;

    // ── DiT glue function handles (F16 — DiT F16 activation path) ──────
    private readonly CudaModule _ditF16Module;
    private readonly nint _ditRmsNormF16;
    private readonly nint _ditLayerNormNoAffineF16;
    private readonly nint _ditAffineBroadcastF16;
    private readonly nint _ditGatedResidualF16;
    private readonly nint _ditAddScalarF16;
    private readonly nint _ditSigmoidF16;
    private readonly nint _ditRopeInterleavedF16;
    private readonly nint _ditRopeF16;
    private readonly nint _ditSliceLastDimF16;
    private readonly nint _ditConcat2F16;
    private readonly nint _ditLayerNormModulateF16;
    private readonly nint _ditQkvSplitNormF16;
    private readonly nint _ditQkvSplitNormHeadMajorF16;
    private readonly nint _ditQkvSplitNormHeadMajorSubsetF16;
    private readonly nint _ditRepeatKvF16;
    private readonly nint _ditSliceRowsF16;
    private readonly nint _ditChwToHwcU8;
    private readonly nint _ditAffineBroadcastRowIndexedF16;
    private readonly nint _ditGatedResidualRowIndexedF16;
    private readonly nint _ditGluActF16;

    // ── DiT glue function handles (BF16 — DiT BF16 activation path) ────
    private readonly CudaModule _ditBf16Module;
    private readonly nint _ditRmsNormBf16;
    private readonly nint _ditGluActBf16;
    private readonly nint _ditGeGluBf16;
    private readonly nint _ditAffineBroadcastRowIndexedBf16;
    private readonly nint _ditGatedResidualRowIndexedBf16;
    private readonly nint _ditAffineBroadcastBf16;
    private readonly nint _ditGatedResidualBf16;

    // ── FP8 Cast Modules + Handles ────────────────────────────────────────
    private readonly CudaModule _castF8Module;
    private readonly nint _castF8E4M3ToF16;

    // ── FP8 Activation Quantization (native fp8 GEMM path) ───────────────
    private readonly CudaModule _fp8QuantModule;
    private readonly nint _fp8AbsMax;
    private readonly nint _fp8AbsMaxFinalizeScale;
    private readonly nint _fp8QuantF32ToE4M3;
    private readonly nint _fp8AbsMaxF16;
    private readonly nint _fp8QuantF16ToE4M3;

    private readonly CudaModule _castBf16Module;
    private readonly nint _castBf16ToF32;
    private readonly nint _castF32ToBf16;
    private readonly nint _castF16ToF8E4M3;

    // ── GGUF Dequant Modules + Handles ───────────────────────────────────
    private readonly CudaModule _dequantQ8_0Module;
    private readonly nint _dequantQ8_0ToF16;
    private readonly CudaModule _dequantQ4_0Module;
    private readonly nint _dequantQ4_0ToF16;
    private readonly CudaModule _dequantQ5_0Module;
    private readonly nint _dequantQ5_0ToF16;
    private readonly CudaModule _dequantQ4_KModule;
    private readonly nint _dequantQ4_KToF16;
    private readonly CudaModule _dequantQ5_KModule;
    private readonly nint _dequantQ5_KToF16;
    private readonly CudaModule _dequantQ6_KModule;
    private readonly nint _dequantQ6_KToF16;

    // ── Fused quantized GEMV (decode M=1) ────────────────────────────────
    private readonly CudaModule _mulMatVecQ4KModule;
    private readonly nint _mulMatVecQ4KF32;
    private readonly CudaModule _mulMatVecQ6KModule;
    private readonly nint _mulMatVecQ6KF32;
    private readonly CudaModule _mulMatVecQ8_0Module;
    private readonly nint _mulMatVecQ8_0F32;
    // Dense 16-bit-float GEMV (the BF16/F16 counterpart of the quant GEMVs, for checkpoints that ship
    // float16 weights — Orpheus/most audio LMs. cuBLAS GemmEx is inefficient at M=1). Loaded best-effort: a
    // load failure disables this fused path (callers fall back to cuBLAS) rather than breaking the backend.
    private readonly CudaModule? _mulMatVecF16Bf16Module;
    private readonly nint _mulMatVecBf16F32;
    private readonly nint _mulMatVecF16F32;
    /// <summary>True when the dense BF16/F16 decode-GEMV kernels loaded successfully.</summary>
    public bool HasFloatGemv { get; private set; }
    private readonly CudaModule _mulMatVecQ5_0Module;
    private readonly nint _mulMatVecQ5_0F32;
    private readonly CudaModule _mulMatVecQ4_0Module;
    private readonly nint _mulMatVecQ4_0F32;
    private readonly CudaModule _mulMatVecQ5KModule;
    private readonly nint _mulMatVecQ5KF32;
    private readonly CudaModule _quantActQ8_1Module;
    private readonly nint _quantActQ8_1F32;
    private readonly CudaModule _mulMatVecQ4KQ8_1Module;
    private readonly nint _mulMatVecQ4KQ8_1;
    private readonly nint _mulMatVecQ4KQ8_1Ksplit;
    private readonly CudaModule _mulMatVecQ8_0Q8_1Module;
    private readonly nint _mulMatVecQ8_0Q8_1;
    private readonly nint _mulMatVecQ8_0Q8_1Ksplit;
    private readonly CudaModule _mulMatVecQ6KQ8_1Module;
    private readonly nint _mulMatVecQ6KQ8_1;
    private readonly nint _mulMatVecQ6KQ8_1Ksplit;
    private readonly CudaModule _mulMatVecQ4_0Q8_1Module;
    private readonly nint _mulMatVecQ4_0Q8_1;
    private readonly CudaModule _mulMatVecQ5_0Q8_1Module;
    private readonly nint _mulMatVecQ5_0Q8_1;
    private readonly nint _mulMatVecQ5_0Q8_1Ksplit;
    private readonly CudaModule _mulMatVecQ5KQ8_1Module;
    private readonly nint _mulMatVecQ5KQ8_1;

    private const uint BlockSize = 256;
    private readonly List<CudaModule> _ownedModules = [];
    private int _disposed;
    private static Func<string, Exception?>? _moduleLoadFailureForTests;

    /// <summary>Test-only fault injector invoked with each PTX path immediately before it is loaded.</summary>
    internal static Func<string, Exception?>? ModuleLoadFailureForTests
    {
        get => Volatile.Read(ref _moduleLoadFailureForTests);
        set => Volatile.Write(ref _moduleLoadFailureForTests, value);
    }

    /// <summary>Loads a module and transfers its finalization ownership to this kernel table. CudaKernels Dispose/
    /// finalizer releases every adopted module, preventing independently queued child finalizers from unloading
    /// modules before their owning kernel table is cleaned.</summary>
    private CudaModule LoadOwnedModule(string path, int maxRegisters = 0)
    {
        Exception? injectedFailure = Volatile.Read(ref _moduleLoadFailureForTests)?.Invoke(path);
        if (injectedFailure is not null) throw injectedFailure;
        CudaModule module = CudaModule.LoadFromFile(path, maxRegisters);
        try
        {
            _ownedModules.Add(module);
        }
        catch (Exception adoptionFailure)
        {
            try { module.Dispose(); }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "CUDA module adoption and rollback both failed.", adoptionFailure, cleanupFailure);
            }
            throw;
        }
        GC.SuppressFinalize(module);
        return module;
    }

    /// <summary>Loads all PTX kernels from the specified directory.</summary>
    public CudaKernels(string ptxDir)
    {
        try
        {
            if (!Directory.Exists(ptxDir))
                throw new DirectoryNotFoundException($"PTX directory not found: {ptxDir}");

        // ── F32 modules ──────────────────────────────────────────────────
        _elementwiseModule = LoadOwnedModule(Path.Combine(ptxDir, "elementwise_f32.ptx"));
        _addF32 = _elementwiseModule.GetFunction("elementwise_add_f32");
        _mulF32 = _elementwiseModule.GetFunction("elementwise_mul_f32");
        _scaleF32 = _elementwiseModule.GetFunction("elementwise_scale_f32");
        _siluF32 = _elementwiseModule.GetFunction("elementwise_silu_f32");
        _geluF32 = _elementwiseModule.GetFunction("elementwise_gelu_f32");
        _clampF32 = _elementwiseModule.GetFunction("elementwise_clamp_f32");

        _groupnormModule = LoadOwnedModule(Path.Combine(ptxDir, "groupnorm_f32.ptx"));
        _groupnormF32 = _groupnormModule.GetFunction("groupnorm_f32");

        _layernormModule = LoadOwnedModule(Path.Combine(ptxDir, "layernorm_f32.ptx"));
        _layernormF32 = _layernormModule.GetFunction("layernorm_f32");

        _spatialModule = LoadOwnedModule(Path.Combine(ptxDir, "spatial_f32.ptx"));
        _upsampleNearest2dF32 = _spatialModule.GetFunction("upsample_nearest2d_f32");
        _im2colF32 = _spatialModule.GetFunction("im2col_f32");
        _col2biasAddF32 = _spatialModule.GetFunction("col2bias_add_f32");

        _im2colBandedModule = LoadOwnedModule(Path.Combine(ptxDir, "im2col_banded.ptx"));
        _im2colBandedF32 = _im2colBandedModule.GetFunction("im2col_banded_f32");
        _im2colBandedF16 = _im2colBandedModule.GetFunction("im2col_banded_f16");
        _im2colBandedBf16 = _im2colBandedModule.GetFunction("im2col_banded_bf16");

        _maxpool2dModule = LoadOwnedModule(Path.Combine(ptxDir, "maxpool2d.ptx"));
        _maxpool2dF32 = _maxpool2dModule.GetFunction("maxpool2d_f32");
        _maxpool2dF16 = _maxpool2dModule.GetFunction("maxpool2d_f16");

        _depthwiseConv2dModule = LoadOwnedModule(Path.Combine(ptxDir, "depthwise_conv2d.ptx"));
        _depthwiseConv2dF32 = _depthwiseConv2dModule.GetFunction("depthwise_conv2d_f32");
        _depthwiseConv2dF16 = _depthwiseConv2dModule.GetFunction("depthwise_conv2d_f16");

        _msdaModule = LoadOwnedModule(Path.Combine(ptxDir, "msda.ptx"));
        _msdaForwardF32 = _msdaModule.GetFunction("msda_forward_f32");

        _softmaxModule = LoadOwnedModule(Path.Combine(ptxDir, "softmax_f32.ptx"));
        _softmaxF32 = _softmaxModule.GetFunction("softmax_f32");

        _transposeModule = LoadOwnedModule(Path.Combine(ptxDir, "transpose_f32.ptx"));
        _transpose2dF32 = _transposeModule.GetFunction("transpose_2d_f32");
        _permute0213F32 = _transposeModule.GetFunction("permute_0213_f32");

        _wanRopeModule = LoadOwnedModule(Path.Combine(ptxDir, "wan_rope.ptx"));
        _wanRopeInterleaved = _wanRopeModule.GetFunction("wan_rope_interleaved");
        _wanRopeInterleavedPerHead = _wanRopeModule.GetFunction("wan_rope_interleaved_perhead");

        _wanVaeFramesModule = LoadOwnedModule(Path.Combine(ptxDir, "wan_vae_frames.ptx"));
        _wanVaeExtractFrame = _wanVaeFramesModule.GetFunction("wan_vae_extract_frame");
        _wanVaeWriteFrame = _wanVaeFramesModule.GetFunction("wan_vae_write_frame");
        _wanVaeExtractFrameBf16 = _wanVaeFramesModule.GetFunction("wan_vae_extract_frame_bf16");
        _wanVaeWriteFrameBf16 = _wanVaeFramesModule.GetFunction("wan_vae_write_frame_bf16");

        _wanVaeConv3dModule = LoadOwnedModule(Path.Combine(ptxDir, "wan_vae_conv3d.ptx"));
        _wanVaeBuildPadded = _wanVaeConv3dModule.GetFunction("wan_vae_build_padded");
        _wanVaeFillBias = _wanVaeConv3dModule.GetFunction("wan_vae_fill_bias");
        _wanVaeAccumulateTap = _wanVaeConv3dModule.GetFunction("wan_vae_accumulate_tap");
        _wanVaeBuildPaddedBf16 = _wanVaeConv3dModule.GetFunction("wan_vae_build_padded_bf16");
        _wanVaeFillBiasBf16 = _wanVaeConv3dModule.GetFunction("wan_vae_fill_bias_bf16");
        _wanVaeAccumulateTapBf16 = _wanVaeConv3dModule.GetFunction("wan_vae_accumulate_tap_bf16");
        _seedVr2PixelShuffle = _wanVaeConv3dModule.GetFunction("seedvr2_pixel_shuffle_f32");
        _seedVr2PixelShuffleBf16 = _wanVaeConv3dModule.GetFunction("seedvr2_pixel_shuffle_bf16");
        _seedVr2PadBr = _wanVaeConv3dModule.GetFunction("seedvr2_pad_br_f32");
        _seedVr2PadBrBf16 = _wanVaeConv3dModule.GetFunction("seedvr2_pad_br_bf16");

        _wanVaeNormModule = LoadOwnedModule(Path.Combine(ptxDir, "wan_vae_norm.ptx"));
        _wanVaeRmsNormChannel = _wanVaeNormModule.GetFunction("wan_vae_rms_norm_channel");
        _wanVaeRmsNormChannelBf16 = _wanVaeNormModule.GetFunction("wan_vae_rms_norm_channel_bf16");
        _wanVaeUnpatchify = _wanVaeNormModule.GetFunction("wan_vae_unpatchify");
        _wanVaeUnpatchifyBf16 = _wanVaeNormModule.GetFunction("wan_vae_unpatchify_bf16");
        _wanVaeSplitQkv = _wanVaeNormModule.GetFunction("wan_vae_split_qkv");
        _wanVaeTokensToFrame = _wanVaeNormModule.GetFunction("wan_vae_tokens_to_frame");

        _gegluModule = LoadOwnedModule(Path.Combine(ptxDir, "geglu_f32.ptx"));
        _gegluF32 = _gegluModule.GetFunction("geglu_f32");

        _broadcastAddModule = LoadOwnedModule(Path.Combine(ptxDir, "broadcast_add_f32.ptx"));
        _broadcastAddF32 = _broadcastAddModule.GetFunction("broadcast_add_f32");

        // Optional module: present only after src/HartsyInference.Cuda/Kernels/dit/build.sh has compiled stepcache.cu on a
        // CUDA-toolkit box. Absence is not an error — the step-cache feature reports unsupported instead.
        string stepCachePath = Path.Combine(ptxDir, "stepcache.ptx");
        if (File.Exists(stepCachePath))
        {
            _stepCacheModule = LoadOwnedModule(stepCachePath);
            _stepCacheRelL1F32 = _stepCacheModule.GetFunction("stepcache_rel_l1_f32");
            _stepCacheRelL1F16 = _stepCacheModule.GetFunction("stepcache_rel_l1_f16");
        }

        // Optional module: LTX-2.5 NA diffusion decoder (src/HartsyInference.Cuda/Kernels/ltx25vae/ltx25_na_decoder.cu).
        // Absence is not an error — the decoder falls back to the managed reference.
        string ltx25NaPath = Path.Combine(ptxDir, "ltx25_na_decoder.ptx");
        if (File.Exists(ltx25NaPath))
        {
            _ltx25NaModule = LoadOwnedModule(ltx25NaPath);
            _ltx25Na3dF32 = _ltx25NaModule.GetFunction("ltx25_na3d_f32");
            _ltx25NaRope3dF32 = _ltx25NaModule.GetFunction("ltx25_na_rope3d_f32");
            // A PTX predating the query-tiled rewrite still serves the two above, so missing tiled entries
            // demote to the untiled kernel rather than failing the whole module.
            try
            {
                _ltx25Na3dTiled8x8F32 = _ltx25NaModule.GetFunction("ltx25_na3d_tiled8x8_f32");
                _ltx25Na3dTiled4x8F32 = _ltx25NaModule.GetFunction("ltx25_na3d_tiled4x8_f32");
            }
            catch (CudaException)
            {
                _ltx25Na3dTiled8x8F32 = 0;
                _ltx25Na3dTiled4x8F32 = 0;
            }
            // Same demotion for a PTX predating the pixel-shuffle entry: the upsample falls back to the host loop.
            try { _ltx25PixelShuffleF32 = _ltx25NaModule.GetFunction("ltx25_pixel_shuffle_f32"); }
            catch (CudaException) { _ltx25PixelShuffleF32 = 0; }
        }

        // Optional module: W8A8 IMMA chain (src/HartsyInference.Cuda/Kernels/dequant/w8a8.cu). Absence is not an error.
        string w8a8Path = Path.Combine(ptxDir, "w8a8.ptx");
        if (File.Exists(w8a8Path))
        {
            _w8a8Module = LoadOwnedModule(w8a8Path);
            _w8a8QuantRowwiseF16 = _w8a8Module.GetFunction("w8a8_quant_rowwise_f16");
            _w8a8QuantRowwiseF32 = _w8a8Module.GetFunction("w8a8_quant_rowwise_f32");
            _w8a8DequantBiasF16 = _w8a8Module.GetFunction("w8a8_dequant_bias_f16");
            _w8a8DequantBiasF32 = _w8a8Module.GetFunction("w8a8_dequant_bias_f32");
            _w8a8DequantBiasStridedF16 = _w8a8Module.GetFunction("w8a8_dequant_bias_strided_f16");
            _w8a8DequantBiasStridedF32 = _w8a8Module.GetFunction("w8a8_dequant_bias_strided_f32");
        }

        // Optional module: ConvRot rotation (src/HartsyInference.Cuda/Kernels/dequant/convrot.cu). Absence is not an error.
        string convRotPath = Path.Combine(ptxDir, "convrot.ptx");
        if (File.Exists(convRotPath))
        {
            _convRotModule = LoadOwnedModule(convRotPath);
            _convRotRotateF16 = _convRotModule.GetFunction("convrot_rotate_f16");
            _convRotRotateF32 = _convRotModule.GetFunction("convrot_rotate_f32");
            _convRotQuantF16 = _convRotModule.GetFunction("convrot_quant_rowwise_f16");
            _convRotQuantGatedF16 = _convRotModule.GetFunction("convrot_quant_rowwise_gated_f16");
            _convRotQuantF32 = _convRotModule.GetFunction("convrot_quant_rowwise_f32");
        }

        // Optional module: fused-dequant int8 mma GEMM (Kernels/dequant/int8_mma_gemm.cu). Absence is not an error.
        string mmaPath = Path.Combine(ptxDir, "int8_mma_gemm.ptx");
        if (File.Exists(mmaPath))
        {
            // No register cap: at 128x256 the accumulator alone is 128 registers and 90 KB of shared already pins
            // the kernel to one block per SM, so capping could only force spills. Verify with NUM_REGS (attribute
            // 4) and LOCAL_SIZE_BYTES (3) after any tile change — CudaModule.LoadFromFile takes a cap if a future
            // smaller tile wants two blocks per SM, since the driver's PTX JIT ignores `.minnctapersm`.
            _int8MmaModule = LoadOwnedModule(mmaPath);
            _int8MmaGemmF16 = _int8MmaModule.GetFunction("int8_mma_gemm_dequant_f16");
            _int8MmaGemmF16Pad = _int8MmaModule.GetFunction("int8_mma_gemm_dequant_f16_pad");
            // Opt in to EXACTLY the mainloop's shared footprint, and only when it exceeds the 48 KB default —
            // never to the SM ceiling. This budget is what the driver uses to decide blocks-per-SM, so asking for
            // 99 KB "to leave room" makes two blocks arithmetically impossible, which silently overrides the
            // kernel's own `.minnctapersm 2` and lets ptxas spend all 256 registers per thread.
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
            // The two entry points have DIFFERENT footprints (the padded one carries 16 B of pad per row), so
            // each opts in to its own — a shared "max of both" would mis-budget the swizzled kernel's occupancy.
            if (Int8MmaSharedBytes > 48 * 1024)
                CudaDriverApi.cuFuncSetAttribute(_int8MmaGemmF16, 8, (int)Int8MmaSharedBytes);
            if (Int8MmaSharedBytesPad > 48 * 1024)
                CudaDriverApi.cuFuncSetAttribute(_int8MmaGemmF16Pad, 8, (int)Int8MmaSharedBytesPad);
        }

        // Optional module: NVFP4 dequant (src/HartsyInference.Cuda/Kernels/dequant/dequant_nvfp4_to_f16.cu). Absence is not an error.
        string nvfp4Path = Path.Combine(ptxDir, "dequant_nvfp4_to_f16.ptx");
        if (File.Exists(nvfp4Path))
        {
            _nvfp4Module = LoadOwnedModule(nvfp4Path);
            _nvfp4DequantF16 = _nvfp4Module.GetFunction("dequant_nvfp4_to_f16");
            _nvfp4DequantBf16 = _nvfp4Module.GetFunction("dequant_nvfp4_to_bf16");
        }

        // Optional module: SageAttention INT8 (src/HartsyInference.Cuda/Kernels/attention/build.sh). Absence is not an error.
        string sageAttnPath = Path.Combine(ptxDir, "sage_attn_int8.ptx");
        if (File.Exists(sageAttnPath))
        {
            _sageAttnModule = LoadOwnedModule(sageAttnPath);
            _sageKMeanF32 = _sageAttnModule.GetFunction("sage_k_mean_f32");
            _sageQuantQInt8F32 = _sageAttnModule.GetFunction("sage_quant_q_int8_f32");
            _sageQuantKInt8F32 = _sageAttnModule.GetFunction("sage_quant_k_int8_f32");
            _sageAttnInt8F32 = _sageAttnModule.GetFunction("sage_attn_int8_f32");
            _sageKMeanF16H = _sageAttnModule.GetFunction("sage_k_mean_f16h");
            _sageQuantQInt8F16H = _sageAttnModule.GetFunction("sage_quant_q_int8_f16h");
            _sageQuantKInt8F16H = _sageAttnModule.GetFunction("sage_quant_k_int8_f16h");

            // The register-resident v1 flash stage (same prologue kernels) — preferred when present.
            string sageV1Path = Path.Combine(ptxDir, "sage_attn_int8_v1.ptx");
            if (File.Exists(sageV1Path))
            {
                _sageAttnV1Module = LoadOwnedModule(sageV1Path);
                _sageAttnV1D128 = _sageAttnV1Module.GetFunction("sage_attn_int8_v1_d128_f32");
                _sageAttnV1D64 = _sageAttnV1Module.GetFunction("sage_attn_int8_v1_d64_f32");
                _sageAttnV1D128F16Acc = _sageAttnV1Module.GetFunction("sage_attn_int8_v1_d128_f16acc_f32");
                _sageAttnV1D64F16Acc = _sageAttnV1Module.GetFunction("sage_attn_int8_v1_d64_f16acc_f32");
                _sageAttnV1D128F16Io = _sageAttnV1Module.GetFunction("sage_attn_int8_v1_d128_f16io");
                _sageAttnV1D64F16Io = _sageAttnV1Module.GetFunction("sage_attn_int8_v1_d64_f16io");
                _sageVF16T = _sageAttnV1Module.GetFunction("sage_v_f16t");
                _sageVF16TH = _sageAttnV1Module.GetFunction("sage_v_f16t_h");
            }
        }

        // ── F16 modules ──────────────────────────────────────────────────
        _elementwiseF16Module = LoadOwnedModule(Path.Combine(ptxDir, "elementwise_f16.ptx"));
        _addF16 = _elementwiseF16Module.GetFunction("elementwise_add_f16");
        _mulF16 = _elementwiseF16Module.GetFunction("elementwise_mul_f16");
        _scaleF16 = _elementwiseF16Module.GetFunction("elementwise_scale_f16");
        _siluF16 = _elementwiseF16Module.GetFunction("elementwise_silu_f16");
        _geluF16 = _elementwiseF16Module.GetFunction("elementwise_gelu_f16");
        _clampF16 = _elementwiseF16Module.GetFunction("elementwise_clamp_f16");

        _groupnormF16Module = LoadOwnedModule(Path.Combine(ptxDir, "groupnorm_f16.ptx"));
        _groupnormF16 = _groupnormF16Module.GetFunction("groupnorm_f16");

        _layernormF16Module = LoadOwnedModule(Path.Combine(ptxDir, "layernorm_f16.ptx"));
        _layernormF16 = _layernormF16Module.GetFunction("layernorm_f16");

        _spatialF16Module = LoadOwnedModule(Path.Combine(ptxDir, "spatial_f16.ptx"));
        _upsampleNearest2dF16 = _spatialF16Module.GetFunction("upsample_nearest2d_f16");
        _im2colF16 = _spatialF16Module.GetFunction("im2col_f16");
        _col2biasAddF16 = _spatialF16Module.GetFunction("col2bias_add_f16");

        _softmaxF16Module = LoadOwnedModule(Path.Combine(ptxDir, "softmax_f16.ptx"));
        _softmaxF16 = _softmaxF16Module.GetFunction("softmax_f16");

        _transposeF16Module = LoadOwnedModule(Path.Combine(ptxDir, "transpose_f16.ptx"));
        _transpose2dF16 = _transposeF16Module.GetFunction("transpose_2d_f16");
        _permute0213F16 = _transposeF16Module.GetFunction("permute_0213_f16");

        _gegluF16Module = LoadOwnedModule(Path.Combine(ptxDir, "geglu_f16.ptx"));
        _gegluF16 = _gegluF16Module.GetFunction("geglu_f16");

        _broadcastAddF16Module = LoadOwnedModule(Path.Combine(ptxDir, "broadcast_add_f16.ptx"));
        _broadcastAddF16 = _broadcastAddF16Module.GetFunction("broadcast_add_f16");

        // ── BF16 modules (subset VAE needs; SDXL VAE F16 overflows) ──────
        _elementwiseBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "elementwise_bf16.ptx"));
        _addBf16 = _elementwiseBf16Module.GetFunction("elementwise_add_bf16");
        _mulBf16 = _elementwiseBf16Module.GetFunction("elementwise_mul_bf16");
        _scaleBf16 = _elementwiseBf16Module.GetFunction("elementwise_scale_bf16");
        _siluBf16 = _elementwiseBf16Module.GetFunction("elementwise_silu_bf16");
        _geluBf16 = _elementwiseBf16Module.GetFunction("elementwise_gelu_bf16");
        _clampBf16 = _elementwiseBf16Module.GetFunction("elementwise_clamp_bf16");

        _groupnormBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "groupnorm_bf16.ptx"));
        _groupnormBf16 = _groupnormBf16Module.GetFunction("groupnorm_bf16");

        _layernormBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "layernorm_bf16.ptx"));
        _layernormBf16 = _layernormBf16Module.GetFunction("layernorm_bf16");

        _spatialBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "spatial_bf16.ptx"));
        _upsampleNearest2dBf16 = _spatialBf16Module.GetFunction("upsample_nearest2d_bf16");
        _im2colBf16 = _spatialBf16Module.GetFunction("im2col_bf16");
        _col2biasAddBf16 = _spatialBf16Module.GetFunction("col2bias_add_bf16");

        _broadcastAddBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "broadcast_add_bf16.ptx"));
        _broadcastAddBf16 = _broadcastAddBf16Module.GetFunction("broadcast_add_bf16");

        _groupnormSiluBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "groupnorm_silu_bf16.ptx"));
        _groupnormSiluBf16 = _groupnormSiluBf16Module.GetFunction("groupnorm_silu_bf16");

        // ── Fused GroupNorm+SiLU ─────────────────────────────────────────
        _groupnormSiluModule = LoadOwnedModule(Path.Combine(ptxDir, "groupnorm_silu_f32.ptx"));
        _groupnormSiluF32 = _groupnormSiluModule.GetFunction("groupnorm_silu_f32");

        _groupnormSiluF16Module = LoadOwnedModule(Path.Combine(ptxDir, "groupnorm_silu_f16.ptx"));
        _groupnormSiluF16 = _groupnormSiluF16Module.GetFunction("groupnorm_silu_f16");

        // ── Cast ─────────────────────────────────────────────────────────
        _castModule = LoadOwnedModule(Path.Combine(ptxDir, "cast_f32_f16.ptx"));
        _castF32ToF16 = _castModule.GetFunction("cast_f32_to_f16");
        _castF16ToF32 = _castModule.GetFunction("cast_f16_to_f32");

        // ── FP8 Cast ─────────────────────────────────────────────────────
        _castF8Module = LoadOwnedModule(Path.Combine(ptxDir, "cast_f8e4m3_f16.ptx"));
        _castF8E4M3ToF16 = _castF8Module.GetFunction("cast_f8e4m3_to_f16");
        _castF16ToF8E4M3 = _castF8Module.GetFunction("cast_f16_to_f8e4m3");

        // ── FP8 Activation Quantization ──────────────────────────────────
        _fp8QuantModule = LoadOwnedModule(Path.Combine(ptxDir, "fp8_quant.ptx"));
        _fp8AbsMax = _fp8QuantModule.GetFunction("absmax_f32");
        _fp8AbsMaxFinalizeScale = _fp8QuantModule.GetFunction("absmax_finalize_scale");
        _fp8QuantF32ToE4M3 = _fp8QuantModule.GetFunction("quant_f32_e4m3");
        _fp8AbsMaxF16 = _fp8QuantModule.GetFunction("absmax_f16");
        _fp8QuantF16ToE4M3 = _fp8QuantModule.GetFunction("quant_f16_e4m3");

        // ── BF16 <-> F32 Cast ───────────────────────────────────────────
        _castBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "cast_bf16_f32.ptx"));
        _castBf16ToF32 = _castBf16Module.GetFunction("cast_bf16_to_f32");
        _castF32ToBf16 = _castBf16Module.GetFunction("cast_f32_to_bf16");

        // ── DiT glue (F32) ───────────────────────────────────────────────
        _ditF32Module = LoadOwnedModule(Path.Combine(ptxDir, "dit_f32.ptx"));
        _ditRmsNormF32 = _ditF32Module.GetFunction("dit_rmsnorm_f32");
        _ditAffineBroadcastF32 = _ditF32Module.GetFunction("dit_affine_broadcast_lastdim_f32");
        _ditGatedResidualF32 = _ditF32Module.GetFunction("dit_gated_residual_lastdim_f32");
        _ditModulation4F32 = _ditF32Module.GetFunction("dit_modulation4_f32");
        _ditCfgEulerF32 = _ditF32Module.GetFunction("dit_cfg_euler_f32");
        _ditCfgRenormReduceF32 = _ditF32Module.GetFunction("dit_cfg_renorm_reduce_f32");
        _ditCfgRenormFinalizeF32 = _ditF32Module.GetFunction("dit_cfg_renorm_finalize_f32");
        _ditCfgRenormEulerF32 = _ditF32Module.GetFunction("dit_cfg_renorm_euler_f32");
        _ditCfgNormalizedEulerF32 = _ditF32Module.GetFunction("dit_cfg_normalized_euler_f32");
        _ditAffineMixF32 = _ditF32Module.GetFunction("dit_affine_mix_f32");
        _ditMaskedAffineMixDenseF32 = _ditF32Module.GetFunction("dit_masked_affine_mix_dense_f32");
        _ditMaskedAffineMixPackedOuterF32 = _ditF32Module.GetFunction("dit_masked_affine_mix_packed_outer_f32");
        _ditMaskedAffineMixPackedInnerF32 = _ditF32Module.GetFunction("dit_masked_affine_mix_packed_inner_f32");
        _ditPatchifyF32 = _ditF32Module.GetFunction("dit_patchify_f32");
        _ditPatchifyU16 = _ditF32Module.GetFunction("dit_patchify_u16");
        _ditUnpatchifyF32 = _ditF32Module.GetFunction("dit_unpatchify_f32");
        _ditUnpatchifyU16 = _ditF32Module.GetFunction("dit_unpatchify_u16");
        _ditSplitSliceU32 = _ditF32Module.GetFunction("dit_split_slice_u32");
        _ditSplitSliceU16 = _ditF32Module.GetFunction("dit_split_slice_u16");
        _ditTanhF32 = _ditF32Module.GetFunction("dit_tanh_f32");
        _ditRopeF32 = _ditF32Module.GetFunction("dit_rope_f32");
        _ditRopeHeadMajorF32 = _ditF32Module.GetFunction("dit_rope_head_major_f32");

        // Separate module so rebuilding it can't perturb dit_f32.ptx's other 40 kernels. Optional:
        // absent PTX just leaves the handle 0 and callers fall back to the v1 kernel above.
        string ropeV2Path = Path.Combine(ptxDir, "dit_rope.ptx");
        if (File.Exists(ropeV2Path))
        {
            _ditRopeModule = LoadOwnedModule(ropeV2Path);
            _ditRopeHeadMajorV2F32 = _ditRopeModule.GetFunction("dit_rope_head_major_v2_f32");
        }

        string fp8EmitPath = Path.Combine(ptxDir, "dit_fp8emit.ptx");
        if (File.Exists(fp8EmitPath))
        {
            _ditFp8EmitModule = LoadOwnedModule(fp8EmitPath);
            _ditAffineBroadcastRowIndexedToFp8F32 =
                _ditFp8EmitModule.GetFunction("dit_affine_broadcast_rowindexed_to_fp8_f32");
        }
        _ltx2SplitRopeF32 = _ditF32Module.GetFunction("ltx2_split_rope_f32");
        _ltx2QkNormRopeHeadMajorF32 = _ditF32Module.GetFunction("ltx2_qk_norm_rope_headmajor_f32");
        _ltx2QkNormRopeTokenMajorF32 = _ditF32Module.GetFunction("ltx2_qk_norm_rope_tokenmajor_f32");
        _ltx2HeadGateF32 = _ditF32Module.GetFunction("ltx2_head_gate_f32");
        _ltx2RmsModulateF32 = _ditF32Module.GetFunction("ltx2_rms_modulate_f32");
        _ditSliceLastDimF32 = _ditF32Module.GetFunction("dit_slice_lastdim_f32");
        _ditRowScaleF32 = _ditF32Module.GetFunction("dit_row_scale_f32");
        _ditAddScalarF32 = _ditF32Module.GetFunction("dit_add_scalar_f32");
        _ditLayerNormNoAffineF32 = _ditF32Module.GetFunction("dit_layernorm_noaffine_f32");
        _ditMoeTopkGateF32 = _ditF32Module.GetFunction("dit_moe_topk_gate_f32");
        _ditRowGatedAccumF32 = _ditF32Module.GetFunction("dit_row_gated_accum_f32");
        _ditIndexAddF32 = _ditF32Module.GetFunction("dit_index_add_f32");
        _ditScatterRowsAfterF32 = _ditF32Module.GetFunction("dit_scatter_rows_after_f32");
        _ditSliceRowsF32 = _ditF32Module.GetFunction("dit_slice_rows_f32");
        _oasisSplitHeadsF32 = _ditF32Module.GetFunction("oasis_split_heads_f32");
        _oasisRopeInterleavedF32 = _ditF32Module.GetFunction("oasis_rope_interleaved_f32");
        _oasisMergeHeadsF32 = _ditF32Module.GetFunction("oasis_merge_heads_f32");
        _oasisUnpatchifyF32 = _ditF32Module.GetFunction("oasis_unpatchify_f32");
        _oasisAdaLnF32 = _ditF32Module.GetFunction("oasis_adaln_f32");
        _ditPixelQuantizeF32 = _ditF32Module.GetFunction("dit_pixel_quantize_f32");
        _triplaneGridSampleF32 = _ditF32Module.GetFunction("triplane_grid_sample_f32");
        _gegluErfF32 = _ditF32Module.GetFunction("geglu_erf_f32");
        _geluErfF32 = _ditF32Module.GetFunction("gelu_erf_f32");
        _convTranspose2dF32 = _ditF32Module.GetFunction("conv_transpose2d_f32");
        _ditConcat2F32 = _ditF32Module.GetFunction("dit_concat2_f32");
        _ditLayerNormModulateF32 = _ditF32Module.GetFunction("dit_layernorm_modulate_f32");
        _ditQkvSplitNormF32 = _ditF32Module.GetFunction("dit_qkv_split_norm_f32");
        _ditQkvSplitNormHeadMajorF32 = _ditF32Module.GetFunction("dit_qkv_split_norm_head_major_f32");
        _ditQkvSplitNormHeadMajorSubsetF32 = _ditF32Module.GetFunction("dit_qkv_split_norm_head_major_subset_f32");
        _fourierEmbedF32 = _ditF32Module.GetFunction("fourier_embed_f32");
        _conv3dF32 = _ditF32Module.GetFunction("conv3d_f32");
        _sparseScatterToGridF32 = _ditF32Module.GetFunction("sparse_scatter_to_grid_f32");
        _sparseGatherFromGridF32 = _ditF32Module.GetFunction("sparse_gather_from_grid_f32");
        _rowGatherF32 = _ditF32Module.GetFunction("row_gather_f32");
        _rowScatterAddF32 = _ditF32Module.GetFunction("row_scatter_add_f32");
        _ditAffineBroadcastRowIndexedF32 = _ditF32Module.GetFunction("dit_affine_broadcast_rowindexed_f32");
        _ditGatedResidualRowIndexedF32 = _ditF32Module.GetFunction("dit_gated_residual_rowindexed_f32");

        _mg3ActionModule = LoadOwnedModule(Path.Combine(ptxDir, "mg3_action.ptx"));
        _mg3SplitQkvTemporalF32 = _mg3ActionModule.GetFunction("mg3_split_qkv_temporal_f32");
        _mg3MergeTemporalF32 = _mg3ActionModule.GetFunction("mg3_merge_temporal_f32");
        _mg3RopeBatchedF32 = _mg3ActionModule.GetFunction("mg3_rope_batched_f32");
        _mg3KvExpandF32 = _mg3ActionModule.GetFunction("mg3_kv_expand_f32");
        _mg3MouseMlpConcatF32 = _mg3ActionModule.GetFunction("mg3_mouse_mlp_concat_f32");

        // ── DiT glue (F16 I/O, F32 accumulate) — DiT F16 activation path ─
        _ditF16Module = LoadOwnedModule(Path.Combine(ptxDir, "dit_f16.ptx"));
        _ditRmsNormF16 = _ditF16Module.GetFunction("dit_rmsnorm_f16");
        _ditLayerNormNoAffineF16 = _ditF16Module.GetFunction("dit_layernorm_noaffine_f16");
        _ditAffineBroadcastF16 = _ditF16Module.GetFunction("dit_affine_broadcast_lastdim_f16");
        _ditGatedResidualF16 = _ditF16Module.GetFunction("dit_gated_residual_lastdim_f16");
        _ditAddScalarF16 = _ditF16Module.GetFunction("dit_add_scalar_f16");
        _ditSigmoidF16 = _ditF16Module.GetFunction("dit_sigmoid_f16");
        _ditRopeInterleavedF16 = _ditF16Module.GetFunction("dit_rope_interleaved_f16");
        _ltx2SplitRopeF16 = _ditF16Module.GetFunction("ltx2_split_rope_f16");
        _ltx2QkNormRopeHeadMajorF16 = _ditF16Module.GetFunction("ltx2_qk_norm_rope_headmajor_f16");
        _ltx2QkNormRopeTokenMajorF16 = _ditF16Module.GetFunction("ltx2_qk_norm_rope_tokenmajor_f16");
        _ltx2QkNormRopeHeadMajorRopeF16 = _ditF16Module.GetFunction("ltx2_qk_norm_rope_headmajor_rf16_f16");
        _ltx2QkNormRopeTokenMajorRopeF16 = _ditF16Module.GetFunction("ltx2_qk_norm_rope_tokenmajor_rf16_f16");
        _ltx2HeadGateF16 = _ditF16Module.GetFunction("ltx2_head_gate_f16");
        _ltx2RmsModulateF16 = _ditF16Module.GetFunction("ltx2_rms_modulate_f16");
        _ditRopeF16 = _ditF16Module.GetFunction("dit_rope_f16");
        _ditSliceLastDimF16 = _ditF16Module.GetFunction("dit_slice_lastdim_f16");
        _ditConcat2F16 = _ditF16Module.GetFunction("dit_concat2_f16");
        _ditLayerNormModulateF16 = _ditF16Module.GetFunction("dit_layernorm_modulate_f16");
        _ditQkvSplitNormF16 = _ditF16Module.GetFunction("dit_qkv_split_norm_f16");
        _ditQkvSplitNormHeadMajorF16 = _ditF16Module.GetFunction("dit_qkv_split_norm_head_major_f16");
        _ditQkvSplitNormHeadMajorSubsetF16 = _ditF16Module.GetFunction("dit_qkv_split_norm_head_major_subset_f16");
        _ditRepeatKvF16 = _ditF16Module.GetFunction("dit_repeat_kv_f16");
        _ditSliceRowsF16 = _ditF16Module.GetFunction("dit_slice_rows_f16");
        _ditChwToHwcU8 = _ditF16Module.GetFunction("dit_chw_f32_to_hwc_u8");
        _ditAffineBroadcastRowIndexedF16 = _ditF16Module.GetFunction("dit_affine_broadcast_rowindexed_f16");
        _ditGatedResidualRowIndexedF16 = _ditF16Module.GetFunction("dit_gated_residual_rowindexed_f16");
        _ditGluActF16 = _ditF16Module.GetFunction("dit_glu_act_f16");

        // ── DiT glue (BF16 I/O, F32 accumulate) — DiT BF16 activation path ─
        _ditBf16Module = LoadOwnedModule(Path.Combine(ptxDir, "dit_bf16.ptx"));
        _ditRmsNormBf16 = _ditBf16Module.GetFunction("dit_rmsnorm_bf16");
        _ditGluActBf16 = _ditBf16Module.GetFunction("dit_glu_act_bf16");
        _ditGeGluBf16 = _ditBf16Module.GetFunction("dit_geglu_bf16");
        _ditAffineBroadcastRowIndexedBf16 = _ditBf16Module.GetFunction("dit_affine_broadcast_rowindexed_bf16");
        _ditGatedResidualRowIndexedBf16 = _ditBf16Module.GetFunction("dit_gated_residual_rowindexed_bf16");
        _ditAffineBroadcastBf16 = _ditBf16Module.GetFunction("dit_affine_broadcast_lastdim_bf16");
        _ditGatedResidualBf16 = _ditBf16Module.GetFunction("dit_gated_residual_lastdim_bf16");

        // ── Audio conv (codec/TTS Conv1d + ConvTranspose1d, F32) ─────────
        _audioConvF32Module = LoadOwnedModule(Path.Combine(ptxDir, "conv1d_f32.ptx"));
        _conv1dF32 = _audioConvF32Module.GetFunction("conv1d_f32");
        _convTranspose1dF32 = _audioConvF32Module.GetFunction("conv_transpose1d_f32");

        _audioActF32Module = LoadOwnedModule(Path.Combine(ptxDir, "audio_activations_f32.ptx"));
        _audioSigmoidF32 = _audioActF32Module.GetFunction("audio_sigmoid_f32");
        _audioMishF32 = _audioActF32Module.GetFunction("audio_mish_f32");
        _audioEluF32 = _audioActF32Module.GetFunction("audio_elu_f32");
        _audioLeakyReluF32 = _audioActF32Module.GetFunction("audio_leaky_relu_f32");
        _audioSnakeF32 = _audioActF32Module.GetFunction("audio_snake_f32");
        _audioSnakeBetaF32 = _audioActF32Module.GetFunction("audio_snake_beta_f32");
        _audioPreluF32 = _audioActF32Module.GetFunction("audio_prelu_f32");
        _audioRepeatTimeF32 = _audioActF32Module.GetFunction("audio_repeat_time_f32");

        _audioAdain1dF32Module = LoadOwnedModule(Path.Combine(ptxDir, "adain1d_f32.ptx"));
        _audioAdain1dF32 = _audioAdain1dF32Module.GetFunction("audio_adain1d_f32");

        // ── Language-model glue (F32) ────────────────────────────────────
        _lmF32Module = LoadOwnedModule(Path.Combine(ptxDir, "lm_f32.ptx"));
        _lmRepeatKvF32 = _lmF32Module.GetFunction("lm_repeat_kv_f32");
        _lmKvAppendF32 = _lmF32Module.GetFunction("lm_kv_append_f32");
        _lmKvAppendF16 = _lmF32Module.GetFunction("lm_kv_append_f16");
        _lmGatherRowsF32 = _lmF32Module.GetFunction("lm_gather_rows_f32");
        _lmScatterAddWeightedRowsF32 = _lmF32Module.GetFunction("lm_scatter_add_weighted_rows_f32");
        _lmArgMaxLastDimF32 = _lmF32Module.GetFunction("lm_argmax_lastdim_f32");
        _lmArgMaxStage1F32 = _lmF32Module.GetFunction("lm_argmax_lastdim_stage1_f32");
        _lmArgMaxStage2F32 = _lmF32Module.GetFunction("lm_argmax_lastdim_stage2_f32");
        _lmGluActF32 = _lmF32Module.GetFunction("lm_glu_act_f32");
        _lmQkvRopeScatterF32 = _lmF32Module.GetFunction("lm_qkv_rope_scatter_f32");
        _lmQkNormRopeScatterF32 = _lmF32Module.GetFunction("lm_qknorm_rope_scatter_f32");
        _lmRmsNormQ8_1F32 = _lmF32Module.GetFunction("lm_rmsnorm_q8_1_f32");
        _lmAddRmsNormQ8_1F32 = _lmF32Module.GetFunction("lm_add_rmsnorm_q8_1_f32");
        _lmGluActQ8_1F32 = _lmF32Module.GetFunction("lm_glu_act_q8_1_f32");
        _lmNormAddRmsNormQ8_1F32 = _lmF32Module.GetFunction("lm_norm_add_rmsnorm_q8_1_f32");
        _lmRmsNormAddF32 = _lmF32Module.GetFunction("lm_rmsnorm_add_f32");
        _lmPleGatherQ5KF32 = _lmF32Module.GetFunction("lm_ple_gather_q5k_f32");
        _lmSsmConvSplitStepF32 = _lmF32Module.GetFunction("lm_ssm_conv_split_step_f32");
        _lmSsmL2NormHeadsF32 = _lmF32Module.GetFunction("lm_ssm_l2norm_heads_f32");
        _lmSsmDeltaStepF32 = _lmF32Module.GetFunction("lm_ssm_delta_step_f32");
        _lmSsmDeltaScalarsF32 = _lmF32Module.GetFunction("lm_ssm_delta_scalars_f32");
        _lmSsmDeltaStepRowsF32 = _lmF32Module.GetFunction("lm_ssm_delta_step_rows_f32");
        _lmSsmDeltaStepWarpRowF32 = _lmF32Module.GetFunction("lm_ssm_delta_step_warprow_f32");
        _lmSsmDeltaGatedNormF32 = _lmF32Module.GetFunction("lm_ssm_delta_gated_norm_f32");
        _lmAddRmsNormF32 = _lmF32Module.GetFunction("lm_add_rmsnorm_f32");
        _lmRopeInterleavedF32 = _lmF32Module.GetFunction("lm_rope_interleaved_f32");
        _lmRopeDecodeSplitHalf = _lmF32Module.GetFunction("lm_rope_decode_splithalf");
        _lmRopeDecodeInterleaved = _lmF32Module.GetFunction("lm_rope_decode_interleaved");
        _lmEmbedGatherDecodeF32 = _lmF32Module.GetFunction("lm_embed_gather_decode_f32");
        _lmHistoryAppend = _lmF32Module.GetFunction("lm_history_append");
        _lmRepetitionPenaltyF32 = _lmF32Module.GetFunction("lm_repetition_penalty_f32");
        _lmKvSliceTimeF32 = _lmF32Module.GetFunction("lm_kv_slice_time_f32");
        _flashAttnF32Module = LoadOwnedModule(Path.Combine(ptxDir, "flash_attn_f32.ptx"));
        _flashAttnF32 = _flashAttnF32Module.GetFunction("lm_flash_attn_f32");
        _flashAttnF16Kv = _flashAttnF32Module.GetFunction("lm_flash_attn_f16kv_f32");
        _flashAttnF32SplitModule = LoadOwnedModule(Path.Combine(ptxDir, "flash_attn_f32_split.ptx"));
        _flashAttnF32Split = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f32_split");
        _flashAttnF32SplitF16Kv = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f16kv_f32_split");
        _flashAttnF32Combine = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f32_combine");
        _flashV2Module = LoadOwnedModule(Path.Combine(ptxDir, "flash_attn_v2_tf32.ptx"));
        _flashV2Tf32 = _flashV2Module.GetFunction("lm_flash_attn_v2_tf32");
        // Opt the fused flash kernel into >48 KB dynamic shared memory (K/V/S/O tiles ≈ 72 KB for D=128).
        // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8. Ignore failure (kernel launch will surface it).
        CudaDriverApi.cuFuncSetAttribute(_flashV2Tf32, 8, 96 * 1024);

        // ── GGUF Dequant ─────────────────────────────────────────────────
        _dequantQ8_0Module = LoadOwnedModule(Path.Combine(ptxDir, "dequant_q8_0_to_f16.ptx"));
        _dequantQ8_0ToF16 = _dequantQ8_0Module.GetFunction("dequant_q8_0_to_f16");
        _dequantQ4_0Module = LoadOwnedModule(Path.Combine(ptxDir, "dequant_q4_0_to_f16.ptx"));
        _dequantQ4_0ToF16 = _dequantQ4_0Module.GetFunction("dequant_q4_0_to_f16");
        _dequantQ5_0Module = LoadOwnedModule(Path.Combine(ptxDir, "dequant_q5_0_to_f16.ptx"));
        _dequantQ5_0ToF16 = _dequantQ5_0Module.GetFunction("dequant_q5_0_to_f16");
        _dequantQ4_KModule = LoadOwnedModule(Path.Combine(ptxDir, "dequant_q4_k_to_f16.ptx"));
        _dequantQ4_KToF16 = _dequantQ4_KModule.GetFunction("dequant_q4_k_to_f16");
        _dequantQ5_KModule = LoadOwnedModule(Path.Combine(ptxDir, "dequant_q5_k_to_f16.ptx"));
        _dequantQ5_KToF16 = _dequantQ5_KModule.GetFunction("dequant_q5_k_to_f16");
        _dequantQ6_KModule = LoadOwnedModule(Path.Combine(ptxDir, "dequant_q6_k_to_f16.ptx"));
        _dequantQ6_KToF16 = _dequantQ6_KModule.GetFunction("dequant_q6_k_to_f16");

        _mulMatVecQ4KModule = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q4k_f32.ptx"));
        _mulMatVecQ4KF32 = _mulMatVecQ4KModule.GetFunction("mul_mat_vec_q4k_f32");
        _mulMatVecQ6KModule = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q6k_f32.ptx"));
        _mulMatVecQ6KF32 = _mulMatVecQ6KModule.GetFunction("mul_mat_vec_q6k_f32");
        _mulMatVecQ8_0Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q8_0_f32.ptx"));
        _mulMatVecQ8_0F32 = _mulMatVecQ8_0Module.GetFunction("mul_mat_vec_q8_0_f32");
        // On by default (HARTSY_BF16_GEMV=0 disables). The PTX targets sm_80 like the engine's other lm/world
        // kernels, so it JITs on every GPU this engine already runs on; a genuine load failure is caught below
        // and falls back to cuBLAS. This is a large decode win — see the lm_head GEMV note in GenericTransformer.
        if (Environment.GetEnvironmentVariable("HARTSY_BF16_GEMV") != "0")
        {
            try
            {
                _mulMatVecF16Bf16Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_f16_bf16_f32.ptx"));
                _mulMatVecBf16F32 = _mulMatVecF16Bf16Module.GetFunction("mul_mat_vec_bf16_f32");
                _mulMatVecF16F32 = _mulMatVecF16Bf16Module.GetFunction("mul_mat_vec_f16_f32");
                HasFloatGemv = true;
            }
            catch (Exception ex)
            {
                _mulMatVecF16Bf16Module?.Dispose();
                _mulMatVecF16Bf16Module = null;
                _mulMatVecBf16F32 = 0; _mulMatVecF16F32 = 0; HasFloatGemv = false;
                HartsyInference.Core.Logging.Logs.Warning($"[Cuda] dense BF16/F16 decode GEMV kernel unavailable ({ex.Message}); using cuBLAS.");
            }
        }
        _mulMatVecQ5_0Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q5_0_f32.ptx"));
        _mulMatVecQ5_0F32 = _mulMatVecQ5_0Module.GetFunction("mul_mat_vec_q5_0_f32");
        _mulMatVecQ4_0Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q4_0_f32.ptx"));
        _mulMatVecQ4_0F32 = _mulMatVecQ4_0Module.GetFunction("mul_mat_vec_q4_0_f32");
        _mulMatVecQ5KModule = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q5k_f32.ptx"));
        _mulMatVecQ5KF32 = _mulMatVecQ5KModule.GetFunction("mul_mat_vec_q5k_f32");
        _quantActQ8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "quantize_activation_q8_1_f32.ptx"));
        _quantActQ8_1F32 = _quantActQ8_1Module.GetFunction("quantize_activation_q8_1_f32");
        _mulMatVecQ4KQ8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q4k_q8_1.ptx"));
        _mulMatVecQ4KQ8_1 = _mulMatVecQ4KQ8_1Module.GetFunction("mul_mat_vec_q4k_q8_1");
        _mulMatVecQ4KQ8_1Ksplit = _mulMatVecQ4KQ8_1Module.GetFunction("mul_mat_vec_q4k_q8_1_ksplit");
        _mulMatVecQ8_0Q8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q8_0_q8_1.ptx"));
        _mulMatVecQ8_0Q8_1 = _mulMatVecQ8_0Q8_1Module.GetFunction("mul_mat_vec_q8_0_q8_1");
        _mulMatVecQ8_0Q8_1Ksplit = _mulMatVecQ8_0Q8_1Module.GetFunction("mul_mat_vec_q8_0_q8_1_ksplit");
        _mulMatVecQ6KQ8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q6k_q8_1.ptx"));
        _mulMatVecQ6KQ8_1 = _mulMatVecQ6KQ8_1Module.GetFunction("mul_mat_vec_q6k_q8_1");
        _mulMatVecQ6KQ8_1Ksplit = _mulMatVecQ6KQ8_1Module.GetFunction("mul_mat_vec_q6k_q8_1_ksplit");
        _mulMatVecQ4_0Q8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q4_0_q8_1.ptx"));
        _mulMatVecQ4_0Q8_1 = _mulMatVecQ4_0Q8_1Module.GetFunction("mul_mat_vec_q4_0_q8_1");
        _mulMatVecQ5_0Q8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q5_0_q8_1.ptx"));
        _mulMatVecQ5_0Q8_1 = _mulMatVecQ5_0Q8_1Module.GetFunction("mul_mat_vec_q5_0_q8_1");
        _mulMatVecQ5_0Q8_1Ksplit = _mulMatVecQ5_0Q8_1Module.GetFunction("mul_mat_vec_q5_0_q8_1_ksplit");
        _mulMatVecQ5KQ8_1Module = LoadOwnedModule(Path.Combine(ptxDir, "mul_mat_vec_q5k_q8_1.ptx"));
        _mulMatVecQ5KQ8_1 = _mulMatVecQ5KQ8_1Module.GetFunction("mul_mat_vec_q5k_q8_1");
        }
        catch (Exception constructionFailure)
        {
            Exception? cleanupFailure = null;
            try { ReleaseOwnedModules(throwOnError: true); }
            catch (Exception error) { cleanupFailure = error; }
            Volatile.Write(ref _disposed, 1);
            GC.SuppressFinalize(this);
            if (cleanupFailure is not null)
            {
                throw new AggregateException(
                    "CUDA kernel-table construction and module rollback both failed.",
                    constructionFailure, cleanupFailure);
            }
            throw;
        }
    }

    // ── Private Launch Helpers ───────────────────────────────────────────

    private unsafe void LaunchBinaryImpl(nint func, ulong output, ulong a, ulong b, int count, nint stream)
    {
        ulong outArg = output, aArg = a, bArg = b;
        uint countArg = (uint)count;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &aArg;
        args[2] = &bArg;
        args[3] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchUnaryImpl(nint func, ulong output, ulong input, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)count;

        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchScaleImpl(nint func, ulong output, ulong input, float scalar, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        float scalarArg = scalar;
        uint countArg = (uint)count;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &scalarArg;
        args[3] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchClampImpl(nint func, ulong output, ulong input, float min, float max, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        float minArg = min, maxArg = max;
        uint countArg = (uint)count;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &minArg;
        args[3] = &maxArg;
        args[4] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches Conv1d (F32, channels-first [B,C,T]), one thread per output element; pass bias=0/hasBias=0 for no bias.</summary>
    public unsafe void LaunchConv1d(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int cIn, int cOut, int tIn, int tOut, int kernel, int stride, int padLeft,
        int dilation, int groups, int hasBias, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        int batchArg = batch, cInArg = cIn, cOutArg = cOut, tInArg = tIn, tOutArg = tOut;
        int kArg = kernel, strideArg = stride, padArg = padLeft, dilArg = dilation, groupsArg = groups, biasArg = hasBias;

        void** args = stackalloc void*[15];
        args[0] = &outArg; args[1] = &inArg; args[2] = &wArg; args[3] = &bArg;
        args[4] = &batchArg; args[5] = &cInArg; args[6] = &cOutArg; args[7] = &tInArg; args[8] = &tOutArg;
        args[9] = &kArg; args[10] = &strideArg; args[11] = &padArg; args[12] = &dilArg; args[13] = &groupsArg; args[14] = &biasArg;

        uint total = (uint)(batch * cOut * tOut);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _conv1dF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches ConvTranspose1d (F32, [B,C,T]); grouped like LaunchConv1d — groups==channels is depthwise (BigVGAN).</summary>
    public unsafe void LaunchConvTranspose1d(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int cIn, int cOut, int tIn, int tOut, int kernel, int stride, int padLeft,
        int dilation, int groups, int hasBias, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        int batchArg = batch, cInArg = cIn, cOutArg = cOut, tInArg = tIn, tOutArg = tOut;
        int kArg = kernel, strideArg = stride, padArg = padLeft, dilArg = dilation, groupsArg = groups, biasArg = hasBias;

        void** args = stackalloc void*[15];
        args[0] = &outArg; args[1] = &inArg; args[2] = &wArg; args[3] = &bArg;
        args[4] = &batchArg; args[5] = &cInArg; args[6] = &cOutArg; args[7] = &tInArg; args[8] = &tOutArg;
        args[9] = &kArg; args[10] = &strideArg; args[11] = &padArg; args[12] = &dilArg; args[13] = &groupsArg; args[14] = &biasArg;

        uint total = (uint)(batch * cOut * tOut);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            _convTranspose1dF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches an elementwise audio activation (Sigmoid) over <paramref name="count"/> F32 elements.</summary>
    public unsafe void LaunchAudioSigmoid(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_audioSigmoidF32, output, input, count, stream);

    public unsafe void LaunchAudioMish(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_audioMishF32, output, input, count, stream);

    /// <summary>Launches the audio Elu activation over <paramref name="count"/> F32 elements.</summary>
    public unsafe void LaunchAudioElu(ulong output, ulong input, float alpha, int count, nint stream)
        => LaunchScaleImpl(_audioEluF32, output, input, alpha, count, stream);

    /// <summary>Launches Leaky ReLU over count F32 elements (StyleTTS 2/Kokoro/HiFi-GAN/VITS use slope=0.2).</summary>
    public unsafe void LaunchAudioLeakyRelu(ulong output, ulong input, float slope, int count, nint stream)
        => LaunchScaleImpl(_audioLeakyReluF32, output, input, slope, count, stream);

    /// <summary>Launches Snake activation x + sin²(αx)/α over [B,C,T] F32, α per-channel (β≠0 uses the β-divisor variant).</summary>
    public unsafe void LaunchAudioSnake(ulong output, ulong input, ulong alpha, ulong beta,
        int batch, int channels, int timeDim, nint stream)
    {
        ulong outArg = output, inArg = input, alphaArg = alpha, betaArg = beta;
        int batchArg = batch, chArg = channels, tArg = timeDim;
        uint total = (uint)(batch * channels * timeDim);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        if (beta != 0)
        {
            void** args = stackalloc void*[7];
            args[0] = &outArg; args[1] = &inArg; args[2] = &alphaArg; args[3] = &betaArg;
            args[4] = &batchArg; args[5] = &chArg; args[6] = &tArg;
            CudaDriverApi.cuLaunchKernel(_audioSnakeBetaF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
        }
        else
        {
            void** args = stackalloc void*[6];
            args[0] = &outArg; args[1] = &inArg; args[2] = &alphaArg;
            args[3] = &batchArg; args[4] = &chArg; args[5] = &tArg;
            CudaDriverApi.cuLaunchKernel(_audioSnakeF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
        }
    }

    /// <summary>Launches parametric ReLU over [B,C,T] F32: x if x&gt;=0 else α·x, per-channel or shared α.</summary>
    public unsafe void LaunchAudioPrelu(ulong output, ulong input, ulong alpha,
        int batch, int channels, int timeDim, int alphaPerChannel, nint stream)
    {
        ulong outArg = output, inArg = input, alphaArg = alpha;
        int batchArg = batch, chArg = channels, tArg = timeDim, perChArg = alphaPerChannel;
        uint total = (uint)(batch * channels * timeDim);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        void** args = stackalloc void*[7];
        args[0] = &outArg; args[1] = &inArg; args[2] = &alphaArg;
        args[3] = &batchArg; args[4] = &chArg; args[5] = &tArg; args[6] = &perChArg;
        CudaDriverApi.cuLaunchKernel(_audioPreluF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches the frame-repeat over [B,C,inT] F32: each time-step duplicated numSamples times.</summary>
    public unsafe void LaunchAudioRepeatTime(ulong output, ulong input,
        int batch, int channels, int inT, int numSamples, nint stream)
    {
        ulong outArg = output, inArg = input;
        int batchArg = batch, chArg = channels, inTArg = inT, nsArg = numSamples;
        uint total = (uint)(batch * channels * inT * numSamples);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        void** args = stackalloc void*[6];
        args[0] = &outArg; args[1] = &inArg;
        args[2] = &batchArg; args[3] = &chArg; args[4] = &inTArg; args[5] = &nsArg;
        CudaDriverApi.cuLaunchKernel(_audioRepeatTimeF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches Adaptive InstanceNorm 1D over [B,C,T] F32: normalizes each row across T, then affines by (1+gamma)/beta.</summary>
    public unsafe void LaunchAudioAdaInstanceNorm1d(ulong output, ulong input, ulong gamma, ulong beta,
        int dim, int totalRows, int channels, bool perBatch, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, gammaArg = gamma, betaArg = beta;
        uint dimArg = (uint)dim, rowsArg = (uint)totalRows, chArg = (uint)channels;
        int perBatchArg = perBatch ? 1 : 0;
        float epsArg = eps;

        void** args = stackalloc void*[9];
        args[0] = &outArg; args[1] = &inArg; args[2] = &gammaArg; args[3] = &betaArg;
        args[4] = &dimArg; args[5] = &rowsArg; args[6] = &chArg; args[7] = &perBatchArg; args[8] = &epsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _audioAdain1dF32, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchGroupNormImpl(nint func, ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        uint batchArg = (uint)batch, chArg = (uint)channels, spatialArg = (uint)spatial, groupsArg = (uint)groups;
        float epsArg = eps;

        void** args = stackalloc void*[9];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &wArg;
        args[3] = &bArg;
        args[4] = &batchArg;
        args[5] = &chArg;
        args[6] = &spatialArg;
        args[7] = &groupsArg;
        args[8] = &epsArg;

        uint gridDim = (uint)(batch * groups);
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchLayerNormImpl(nint func, ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        uint normDimArg = (uint)normDim, rowsArg = (uint)totalRows;
        float epsArg = eps;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &wArg;
        args[3] = &bArg;
        args[4] = &normDimArg;
        args[5] = &rowsArg;
        args[6] = &epsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchRmsNormImpl(nint func, ulong output, ulong input, ulong weight,
        int normDim, int totalRows, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, wArg = weight;
        uint normDimArg = (uint)normDim, rowsArg = (uint)totalRows;
        float epsArg = eps;

        void** args = stackalloc void*[6];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &wArg;
        args[3] = &normDimArg;
        args[4] = &rowsArg;
        args[5] = &epsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchAffineBroadcastImpl(nint func, ulong output, ulong input, ulong scale, ulong shift,
        int seqLen, int dim, long total, nint stream)
    {
        ulong outArg = output, inArg = input, scaleArg = scale, shiftArg = shift;
        uint seqArg = (uint)seqLen, dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &scaleArg;
        args[3] = &shiftArg;
        args[4] = &seqArg;
        args[5] = &dimArg;
        args[6] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchAffineBroadcastRowIndexedImpl(nint func, ulong output, ulong input, ulong scaleTable,
        ulong shiftTable, ulong rowIndex, int dim, long total, nint stream)
    {
        ulong outArg = output, inArg = input, scaleArg = scaleTable, shiftArg = shiftTable, idxArg = rowIndex;
        uint dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &scaleArg;
        args[3] = &shiftArg;
        args[4] = &idxArg;
        args[5] = &dimArg;
        args[6] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the fused modulate-to-fp8 producer kernel is loaded.</summary>
    public bool HasAffineBroadcastRowIndexedToFp8 => _ditAffineBroadcastRowIndexedToFp8F32 != 0;

    /// <summary>Row-indexed affine broadcast writing e4m3 directly, so the consuming fp8 Linear needs no quantize pass.</summary>
    public unsafe void LaunchAffineBroadcastRowIndexedToFp8(ulong outputFp8, ulong input, ulong scaleTable,
        ulong shiftTable, ulong rowIndex, ulong inputScale, int dim, long total, nint stream)
    {
        ulong outArg = outputFp8, inArg = input, scaleArg = scaleTable, shiftArg = shiftTable;
        ulong idxArg = rowIndex, inScaleArg = inputScale;
        uint dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[8];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &scaleArg;
        args[3] = &shiftArg;
        args[4] = &idxArg;
        args[5] = &inScaleArg;
        args[6] = &dimArg;
        args[7] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            _ditAffineBroadcastRowIndexedToFp8F32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchGatedResidualRowIndexedImpl(nint func, ulong output, ulong residual, ulong value,
        ulong gateTable, ulong rowIndex, int dim, long total, nint stream)
    {
        ulong outArg = output, resArg = residual, valArg = value, gateArg = gateTable, idxArg = rowIndex;
        uint dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &resArg;
        args[2] = &valArg;
        args[3] = &gateArg;
        args[4] = &idxArg;
        args[5] = &dimArg;
        args[6] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchGatedResidualImpl(nint func, ulong output, ulong residual, ulong value, ulong gate,
        int seqLen, int dim, long total, nint stream)
    {
        ulong outArg = output, resArg = residual, valArg = value, gateArg = gate;
        uint seqArg = (uint)seqLen, dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &resArg;
        args[2] = &valArg;
        args[3] = &gateArg;
        args[4] = &seqArg;
        args[5] = &dimArg;
        args[6] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchRepeatKvImpl(nint func, ulong output, ulong input,
        int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint kvArg = (uint)kvHeads, groupArg = (uint)group, seqArg = (uint)seqLen, dimArg = (uint)headDim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[7];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &kvArg;
        args[3] = &groupArg;
        args[4] = &seqArg;
        args[5] = &dimArg;
        args[6] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchModulation4Impl(nint func, ulong scaleMsa, ulong gateMsa, ulong scaleMlp, ulong gateMlp,
        ulong proj, int dim, int batch, nint stream)
    {
        ulong sMsaArg = scaleMsa, gMsaArg = gateMsa, sMlpArg = scaleMlp, gMlpArg = gateMlp, projArg = proj;
        uint dimArg = (uint)dim, batchArg = (uint)batch;

        void** args = stackalloc void*[7];
        args[0] = &sMsaArg;
        args[1] = &gMsaArg;
        args[2] = &sMlpArg;
        args[3] = &gMlpArg;
        args[4] = &projArg;
        args[5] = &dimArg;
        args[6] = &batchArg;

        uint total = (uint)(batch * dim);
        uint gridDim = (total + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchCfgEulerImpl(nint func, ulong z, ulong pos, ulong neg,
        float guidance, float delta, int count, nint stream)
    {
        ulong zArg = z, posArg = pos, negArg = neg;
        float gArg = guidance, dArg = delta;
        uint countArg = (uint)count;

        void** args = stackalloc void*[6];
        args[0] = &zArg;
        args[1] = &posArg;
        args[2] = &negArg;
        args[3] = &gArg;
        args[4] = &dArg;
        args[5] = &countArg;

        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchRopeImpl(nint func, ulong x, ulong cos, ulong sin,
        int numHeads, int headDim, long totalVecs, int rotaryDim, nint stream)
    {
        // rotaryDim 0 (or >= headDim) = full rotary; else partial (rotate the first rotaryDim dims of each head).
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint headsArg = (uint)numHeads, headDimArg = (uint)headDim, rotArg = (uint)rdim;
        ulong vecsArg = (ulong)totalVecs;

        void** args = stackalloc void*[7];
        args[0] = &xArg;
        args[1] = &cosArg;
        args[2] = &sinArg;
        args[3] = &headsArg;
        args[4] = &headDimArg;
        args[5] = &vecsArg;
        args[6] = &rotArg;

        long threads = totalVecs * (rdim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchRowScaleImpl(nint func, ulong output, ulong input, ulong rowScale,
        int channels, long total, nint stream)
    {
        ulong outArg = output, inArg = input, scaleArg = rowScale;
        uint chArg = (uint)channels;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &scaleArg;
        args[3] = &chArg;
        args[4] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchLayerNormNoAffineImpl(nint func, ulong output, ulong input,
        int dim, int totalRows, float eps, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint dimArg = (uint)dim, rowsArg = (uint)totalRows;
        float epsArg = eps;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &dimArg;
        args[3] = &rowsArg;
        args[4] = &epsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchIndexAddImpl(nint func, ulong h, ulong table, ulong indices,
        int dim, long total, nint stream)
    {
        ulong hArg = h, tableArg = table, idxArg = indices;
        uint dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[5];
        args[0] = &hArg;
        args[1] = &tableArg;
        args[2] = &idxArg;
        args[3] = &dimArg;
        args[4] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchScatterRowsAfterImpl(nint func, ulong output, ulong input,
        int headRows, int dim, long total, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint headArg = (uint)headRows, dimArg = (uint)dim;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &headArg;
        args[3] = &dimArg;
        args[4] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchSliceRowsImpl(nint func, ulong output, ulong input,
        long elemOffset, long total, nint stream)
    {
        ulong outArg = output, inArg = input;
        ulong offArg = (ulong)elemOffset, totalArg = (ulong)total;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &offArg;
        args[3] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchSliceLastDimImpl(nint func, ulong output, ulong input,
        int outDim, int inDim, int offset, long total, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint outDimArg = (uint)outDim, inDimArg = (uint)inDim, offsetArg = (uint)offset;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[6];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &outDimArg;
        args[3] = &inDimArg;
        args[4] = &offsetArg;
        args[5] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchSoftmaxImpl(nint func, ulong data, int rowLen, int totalRows, nint stream)
    {
        ulong dataArg = data;
        uint rowLenArg = (uint)rowLen;
        uint totalRowsArg = (uint)totalRows;

        void** args = stackalloc void*[3];
        args[0] = &dataArg;
        args[1] = &rowLenArg;
        args[2] = &totalRowsArg;

        uint gridDim = (uint)totalRows;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchUpsampleImpl(nint func, ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint batchArg = (uint)batch, chArg = (uint)channels;
        uint inHArg = (uint)inH, inWArg = (uint)inW;
        uint outHArg = (uint)outH, outWArg = (uint)outW;
        uint scaleHArg = (uint)scaleH, scaleWArg = (uint)scaleW;

        void** args = stackalloc void*[10];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &batchArg;
        args[3] = &chArg;
        args[4] = &inHArg;
        args[5] = &inWArg;
        args[6] = &outHArg;
        args[7] = &outWArg;
        args[8] = &scaleHArg;
        args[9] = &scaleWArg;

        long totalElements = (long)batch * channels * outH * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches banded Im2Col for one batch element: fills the col buffer for output rows [ohBase, ohBase+bandRows).</summary>
    public unsafe void LaunchIm2ColBanded(DType dtype, ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outW, int ohBase, int bandRows, int batchChanOffset, nint stream)
    {
        nint func = dtype == DType.F16 ? _im2colBandedF16
            : dtype == DType.BF16 ? _im2colBandedBf16
            : _im2colBandedF32;
        ulong colArg = col, inArg = input;
        uint chArg = (uint)channels, inHArg = (uint)inH, inWArg = (uint)inW;
        uint kHArg = (uint)kH, kWArg = (uint)kW;
        uint padHArg = (uint)padH, padWArg = (uint)padW;
        uint strHArg = (uint)strideH, strWArg = (uint)strideW;
        uint outWArg = (uint)outW, ohBaseArg = (uint)ohBase, bandRowsArg = (uint)bandRows;
        uint chanOffArg = (uint)batchChanOffset;

        void** args = stackalloc void*[15];
        args[0] = &colArg;
        args[1] = &inArg;
        args[2] = &chArg;
        args[3] = &inHArg;
        args[4] = &inWArg;
        args[5] = &kHArg;
        args[6] = &kWArg;
        args[7] = &padHArg;
        args[8] = &padWArg;
        args[9] = &strHArg;
        args[10] = &strWArg;
        args[11] = &outWArg;
        args[12] = &ohBaseArg;
        args[13] = &bandRowsArg;
        args[14] = &chanOffArg;

        long totalElements = (long)channels * kH * kW * bandRows * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>2D max-pool over an NCHW tensor; one thread per output element. Dtype selects the F32 or F16 kernel.</summary>
    public unsafe void LaunchMaxPool2D(DType dtype, ulong output, ulong input,
        int n, int channels, int inH, int inW, int outH, int outW,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, nint stream)
    {
        nint func = dtype == DType.F16 ? _maxpool2dF16 : _maxpool2dF32;
        ulong outArg = output, inArg = input;
        uint nArg = (uint)n, chArg = (uint)channels, inHArg = (uint)inH, inWArg = (uint)inW;
        uint outHArg = (uint)outH, outWArg = (uint)outW;
        uint kHArg = (uint)kernelH, kWArg = (uint)kernelW;
        uint strHArg = (uint)strideH, strWArg = (uint)strideW;
        uint padHArg = (uint)padH, padWArg = (uint)padW;

        void** args = stackalloc void*[14];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &nArg;
        args[3] = &chArg;
        args[4] = &inHArg;
        args[5] = &inWArg;
        args[6] = &outHArg;
        args[7] = &outWArg;
        args[8] = &kHArg;
        args[9] = &kWArg;
        args[10] = &strHArg;
        args[11] = &strWArg;
        args[12] = &padHArg;
        args[13] = &padWArg;

        long totalElements = (long)n * channels * outH * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Multi-scale deformable attention forward. One thread per (query, head, channel); softmax over levels·points is folded in.</summary>
    /// <param name="shapes">Per-level (h,w), int[levels*2].</param>
    /// <param name="levelStart">Per-level flattened offset, int[levels].</param>
    public unsafe void LaunchMsdaForward(ulong output, ulong value, ulong sampOff, ulong attn, ulong refPoints,
        ulong shapes, ulong levelStart, int nq, int heads, int hd, int levels, int points, int coords,
        int refQueryStride, int refLevelStride, nint stream)
    {
        ulong outArg = output, valArg = value, offArg = sampOff, atArg = attn, refArg = refPoints;
        ulong shArg = shapes, lsArg = levelStart;
        uint nqArg = (uint)nq, hArg = (uint)heads, hdArg = (uint)hd, lvArg = (uint)levels;
        uint ptArg = (uint)points, coArg = (uint)coords, rqArg = (uint)refQueryStride, rlArg = (uint)refLevelStride;

        void** args = stackalloc void*[15];
        args[0] = &outArg;
        args[1] = &valArg;
        args[2] = &offArg;
        args[3] = &atArg;
        args[4] = &refArg;
        args[5] = &shArg;
        args[6] = &lsArg;
        args[7] = &nqArg;
        args[8] = &hArg;
        args[9] = &hdArg;
        args[10] = &lvArg;
        args[11] = &ptArg;
        args[12] = &coArg;
        args[13] = &rqArg;
        args[14] = &rlArg;

        long total = (long)nq * heads * hd;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            _msdaForwardF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Depthwise 2D conv (groups==channels) over NCHW; weight [C,1,kH,kW], bias optional, one thread per output element.</summary>
    public unsafe void LaunchDepthwiseConv2D(DType dtype, ulong output, ulong input, ulong weight, ulong bias,
        int hasBias, int n, int channels, int inH, int inW, int outH, int outW,
        int kH, int kW, int strideH, int strideW, int padH, int padW, nint stream)
    {
        nint func = dtype == DType.F16 ? _depthwiseConv2dF16 : _depthwiseConv2dF32;
        ulong outArg = output, inArg = input, wArg = weight, bArg = bias;
        int hasBiasArg = hasBias;
        uint nArg = (uint)n, chArg = (uint)channels, inHArg = (uint)inH, inWArg = (uint)inW;
        uint outHArg = (uint)outH, outWArg = (uint)outW;
        uint kHArg = (uint)kH, kWArg = (uint)kW;
        uint strHArg = (uint)strideH, strWArg = (uint)strideW;
        uint padHArg = (uint)padH, padWArg = (uint)padW;

        void** args = stackalloc void*[17];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &wArg;
        args[3] = &bArg;
        args[4] = &hasBiasArg;
        args[5] = &nArg;
        args[6] = &chArg;
        args[7] = &inHArg;
        args[8] = &inWArg;
        args[9] = &outHArg;
        args[10] = &outWArg;
        args[11] = &kHArg;
        args[12] = &kWArg;
        args[13] = &strHArg;
        args[14] = &strWArg;
        args[15] = &padHArg;
        args[16] = &padWArg;

        long totalElements = (long)n * channels * outH * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchIm2ColImpl(nint func, ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
    {
        ulong colArg = col, inArg = input;
        uint chArg = (uint)channels, inHArg = (uint)inH, inWArg = (uint)inW;
        uint kHArg = (uint)kH, kWArg = (uint)kW;
        uint padHArg = (uint)padH, padWArg = (uint)padW;
        uint strHArg = (uint)strideH, strWArg = (uint)strideW;
        uint outHArg = (uint)outH, outWArg = (uint)outW;
        uint batchOffsetArg = (uint)batchOffset;

        void** args = stackalloc void*[14];
        args[0] = &colArg;
        args[1] = &inArg;
        args[2] = &chArg;
        args[3] = &inHArg;
        args[4] = &inWArg;
        args[5] = &kHArg;
        args[6] = &kWArg;
        args[7] = &padHArg;
        args[8] = &padWArg;
        args[9] = &strHArg;
        args[10] = &strWArg;
        args[11] = &outHArg;
        args[12] = &outWArg;
        args[13] = &batchOffsetArg;

        long totalElements = (long)channels * kH * kW * outH * outW;
        uint gridDim = (uint)((totalElements + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchBiasAddImpl(nint func, ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
    {
        ulong outArg = output, biasArg = bias;
        uint outChArg = (uint)outChannels, spatialArg = (uint)spatial, totalArg = (uint)totalElements;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &biasArg;
        args[2] = &outChArg;
        args[3] = &spatialArg;
        args[4] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchTranspose2DImpl(nint func, ulong output, ulong input, int d1, int d2, int totalElements, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint d1Arg = (uint)d1, d2Arg = (uint)d2, totalArg = (uint)totalElements;

        void** args = stackalloc void*[5];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &d1Arg;
        args[3] = &d2Arg;
        args[4] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchPermute0213Impl(nint func, ulong output, ulong input, int s, int h, int d, int totalElements, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint sArg = (uint)s, hArg = (uint)h, dArg = (uint)d, totalArg = (uint)totalElements;

        void** args = stackalloc void*[6];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &sArg;
        args[3] = &hArg;
        args[4] = &dArg;
        args[5] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchGeGluImpl(nint func, ulong output, ulong input, int innerDim, int outputElements, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint innerDimArg = (uint)innerDim;
        uint outElemArg = (uint)outputElements;

        void** args = stackalloc void*[4];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &innerDimArg;
        args[3] = &outElemArg;

        uint gridDim = ((uint)outputElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchBroadcastAddImpl(nint func, ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
    {
        ulong hiddenArg = hidden, biasArg = bias;
        uint chArg = (uint)channels, spatialArg = (uint)spatial, totalArg = (uint)totalElements;

        void** args = stackalloc void*[5];
        args[0] = &hiddenArg;
        args[1] = &biasArg;
        args[2] = &chArg;
        args[3] = &spatialArg;
        args[4] = &totalArg;

        uint gridDim = ((uint)totalElements + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Elementwise Launches ────────────────────────────────────────────

    /// <summary>Launches elementwise add: output[i] = a[i] + b[i] (F32)</summary>
    public void LaunchAdd(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_addF32, output, a, b, count, stream);

    /// <summary>Whether the optional stepcache.ptx module was found and loaded (see src/HartsyInference.Cuda/Kernels/dit/stepcache.cu).</summary>
    public bool HasStepCacheKernels => _stepCacheModule is not null;

    /// <summary>Launches the feature-cache gate reduction: sums[0] += Σ|a−b|, sums[1] += Σ|b| over count elements.</summary>
    /// <param name="sums">Caller-pre-zeroed 2-float output buffer.</param>
    /// <param name="isF16">Selects the F32 or F16 operand kernel.</param>
    public unsafe void LaunchStepCacheRelL1(ulong sums, ulong a, ulong b, long count, bool isF16, nint stream)
    {
        if (_stepCacheModule is null)
            throw new InvalidOperationException("stepcache.ptx is not loaded; gate on HasStepCacheKernels first.");

        ulong aArg = a, bArg = b, sumsArg = sums;
        ulong countArg = (ulong)count;

        void** args = stackalloc void*[4];
        args[0] = &aArg;
        args[1] = &bArg;
        args[2] = &sumsArg;
        args[3] = &countArg;

        // Grid-stride kernel: cap blocks so the per-block atomic tail stays negligible while still
        // saturating the SMs at DiT activation sizes (seqLen×hidden ≈ 15M elements).
        uint gridDim = (uint)Math.Min(1024, (count + BlockSize - 1) / BlockSize);
        nint func = isF16 ? _stepCacheRelL1F16 : _stepCacheRelL1F32;
        CudaDriverApi.cuLaunchKernel(
            func, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the optional ltx25_na_decoder.ptx module was found and loaded
    /// (src/HartsyInference.Cuda/Kernels/ltx25vae/ltx25_na_decoder.cu).</summary>
    public bool HasLtx25NaKernels => _ltx25NaModule is not null;

    /// <summary>Whether the loaded ltx25_na_decoder.ptx carries the pixel-shuffle entry.</summary>
    public bool HasLtx25PixelShuffle => _ltx25PixelShuffleF32 != 0;

    /// <summary>Shared-memory bytes <see cref="LaunchLtx25Na3d"/> needs: the q row plus one score per window position.</summary>
    public static int Ltx25Na3dSharedBytes(int headDim, int kt, int kh, int kw) => (headDim + kt * kh * kw) * sizeof(float);

    /// <summary>3D neighborhood attention over [batch,T,H,W,heads,headDim]; one block per (batch,t,h,w,head) query.
    /// Kernel extents must already be clamped to their axis length, as the managed reference does.</summary>
    public unsafe void LaunchLtx25Na3d(ulong output, ulong q, ulong k, ulong v,
        int batch, int dimT, int dimH, int dimW, int heads, int headDim,
        int kt, int kh, int kw, float scale, nint stream)
    {
        if (_ltx25NaModule is null)
            throw new InvalidOperationException("ltx25_na_decoder.ptx is not loaded; gate on HasLtx25NaKernels first.");

        ulong outArg = output, qArg = q, kArg = k, vArg = v;
        uint bArg = (uint)batch, tArg = (uint)dimT, hArg = (uint)dimH, wArg = (uint)dimW;
        uint headsArg = (uint)heads, hdArg = (uint)headDim;
        uint ktArg = (uint)kt, khArg = (uint)kh, kwArg = (uint)kw;
        float scaleArg = scale;

        void** args = stackalloc void*[14];
        args[0] = &outArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bArg; args[5] = &tArg; args[6] = &hArg; args[7] = &wArg;
        args[8] = &headsArg; args[9] = &hdArg;
        args[10] = &ktArg; args[11] = &khArg; args[12] = &kwArg; args[13] = &scaleArg;

        // One block per query·head. Threads split the window in the score pass and head_dim in the value pass,
        // so a block wider than either is wasted; 256 covers head_dim 64 and keeps 8 warps on the window.
        uint blockDim = 256;
        uint gridDim = (uint)((long)batch * dimT * dimH * dimW * heads);
        uint sharedBytes = (uint)Ltx25Na3dSharedBytes(headDim, kt, kh, kw);
        CudaDriverApi.cuLaunchKernel(_ltx25Na3dF32, gridDim, 1, 1, blockDim, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Query-tile height/width the tiled na3d kernels stage a shared window for, or (0, 0) when the shape
    /// is not one they cover — head_dim must be 64, and a tile only pays for itself once it holds real queries.</summary>
    public (int H, int W) Ltx25Na3dTile(int dimH, int dimW, int headDim)
    {
        if (_ltx25Na3dTiled8x8F32 == 0 || headDim != 64) return (0, 0);
        // A tile wider than the volume is all padding, and the union it stages is then the whole volume anyway.
        if (dimH >= 8 && dimW >= 8) return (8, 8);
        if (dimH >= 4 && dimW >= 8) return (4, 8);
        return (0, 0);
    }

    /// <summary>Query-tiled 3D neighborhood attention: one block per (batch, t, h-tile, w-tile, head), streaming the
    /// tile's union window through shared memory. Same contract as <see cref="LaunchLtx25Na3d"/>; the tile must come
    /// from <see cref="Ltx25Na3dTile"/>.</summary>
    public unsafe void LaunchLtx25Na3dTiled(ulong output, ulong q, ulong k, ulong v,
        int batch, int dimT, int dimH, int dimW, int heads, int headDim,
        int kt, int kh, int kw, float scale, int tileH, int tileW, nint stream)
    {
        nint func = tileH == 8 ? _ltx25Na3dTiled8x8F32 : _ltx25Na3dTiled4x8F32;
        if (func == 0)
            throw new InvalidOperationException("ltx25_na3d_tiled*_f32 is not loaded; gate on Ltx25Na3dTile first.");

        ulong outArg = output, qArg = q, kArg = k, vArg = v;
        uint bArg = (uint)batch, tArg = (uint)dimT, hArg = (uint)dimH, wArg = (uint)dimW;
        uint headsArg = (uint)heads, hdArg = (uint)headDim;
        uint ktArg = (uint)kt, khArg = (uint)kh, kwArg = (uint)kw;
        float scaleArg = scale;

        void** args = stackalloc void*[14];
        args[0] = &outArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bArg; args[5] = &tArg; args[6] = &hArg; args[7] = &wArg;
        args[8] = &headsArg; args[9] = &hdArg;
        args[10] = &ktArg; args[11] = &khArg; args[12] = &kwArg; args[13] = &scaleArg;

        long hTiles = (dimH + tileH - 1) / tileH, wTiles = (dimW + tileW - 1) / tileW;
        uint gridDim = (uint)((long)batch * dimT * hTiles * wTiles * heads);
        CudaDriverApi.cuLaunchKernel(func, gridDim, 1, 1, 256, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>3-axis interleaved rope on x[1,t,h,w,heads,headDim], in place. cos/sin tables are per axis,
    /// laid out [axisLength, pairs]; the three head chunks sit at offsets 0 / offsetH / offsetW.</summary>
    /// <summary>Launches the channels-last 3D pixel shuffle over one temporal chunk; one thread per SOURCE element,
    /// so <paramref name="total"/> is the projection's element count.</summary>
    public unsafe void LaunchLtx25PixelShuffle(ulong dst, ulong src, int start, int h, int w,
        int strideT, int strideH, int strideW, int outChannels, int dropped, int outH, int outW,
        long total, nint stream)
    {
        if (_ltx25PixelShuffleF32 == 0)
            throw new InvalidOperationException("ltx25_pixel_shuffle_f32 is not loaded; gate on HasLtx25PixelShuffle first.");

        ulong dstArg = dst, srcArg = src;
        int startArg = start, hArg = h, wArg = w;
        int p1Arg = strideT, p2Arg = strideH, p3Arg = strideW;
        int ocArg = outChannels, dropArg = dropped, ohArg = outH, owArg = outW;
        ulong totalArg = (ulong)total;

        void** args = stackalloc void*[13];
        args[0] = &dstArg; args[1] = &srcArg;
        args[2] = &startArg; args[3] = &hArg; args[4] = &wArg;
        args[5] = &p1Arg; args[6] = &p2Arg; args[7] = &p3Arg;
        args[8] = &ocArg; args[9] = &dropArg; args[10] = &ohArg; args[11] = &owArg;
        args[12] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ltx25PixelShuffleF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchLtx25NaRope3d(ulong x,
        ulong cosT, ulong sinT, ulong cosH, ulong sinH, ulong cosW, ulong sinW,
        int dimT, int dimH, int dimW, int heads, int headDim,
        int pairsT, int pairsH, int pairsW, int offsetH, int offsetW, nint stream)
    {
        if (_ltx25NaModule is null)
            throw new InvalidOperationException("ltx25_na_decoder.ptx is not loaded; gate on HasLtx25NaKernels first.");

        ulong xArg = x, ctArg = cosT, stArg = sinT, chArg = cosH, shArg = sinH, cwArg = cosW, swArg = sinW;
        uint tArg = (uint)dimT, hArg = (uint)dimH, wArg = (uint)dimW;
        uint headsArg = (uint)heads, hdArg = (uint)headDim;
        uint ptArg = (uint)pairsT, phArg = (uint)pairsH, pwArg = (uint)pairsW;
        uint ohArg = (uint)offsetH, owArg = (uint)offsetW;

        void** args = stackalloc void*[17];
        args[0] = &xArg;
        args[1] = &ctArg; args[2] = &stArg; args[3] = &chArg; args[4] = &shArg; args[5] = &cwArg; args[6] = &swArg;
        args[7] = &tArg; args[8] = &hArg; args[9] = &wArg;
        args[10] = &headsArg; args[11] = &hdArg;
        args[12] = &ptArg; args[13] = &phArg; args[14] = &pwArg;
        args[15] = &ohArg; args[16] = &owArg;

        long threads = (long)dimT * dimH * dimW * heads * (pairsT + pairsH + pairsW);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ltx25NaRope3dF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the optional w8a8.ptx module was found and loaded (src/HartsyInference.Cuda/Kernels/dequant/w8a8.cu).</summary>
    public bool HasW8A8Kernels => _w8a8Module is not null;

    /// <summary>W8A8 prologue — per-row dynamic INT8 quant: x[rows,cols] → q8 + rowScale[rows] (256-thread block per row).</summary>
    /// <param name="invScale">Optional SmoothQuant 1/s_j per-channel vector [cols] (X_hat = X/s); pass 0 for none.</param>
    public unsafe void LaunchW8A8QuantRowwise(ulong q8, ulong rowScale, ulong x, int rows, int cols, nint stream,
        bool srcF16, ulong invScale = 0)
    {
        if (_w8a8Module is null) throw new InvalidOperationException("w8a8.ptx not present in the Ptx folder.");
        ulong xArg = x, qArg = q8, sArg = rowScale, isArg = invScale;
        uint colsArg = (uint)cols;
        void** args = stackalloc void*[5];
        args[0] = &xArg; args[1] = &qArg; args[2] = &sArg; args[3] = &colsArg; args[4] = &isArg;
        CudaDriverApi.cuLaunchKernel(srcF16 ? _w8a8QuantRowwiseF16 : _w8a8QuantRowwiseF32,
            (uint)rows, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>W8A8 epilogue — dequants the int32 GEMM result with rowScale·wScale (+ optional bias) into F16 or F32.</summary>
    public unsafe void LaunchW8A8DequantBias(ulong output, ulong d32, ulong rowScale, ulong wScale, ulong bias,
        int rows, int cols, nint stream, bool outF16, uint actMode = 0)
    {
        if (_w8a8Module is null) throw new InvalidOperationException("w8a8.ptx not present in the Ptx folder.");
        ulong dArg = d32, rsArg = rowScale, wsArg = wScale, bArg = bias, oArg = output;
        uint rowsArg = (uint)rows, colsArg = (uint)cols, actArg = actMode;
        void** args = stackalloc void*[8];
        args[0] = &dArg; args[1] = &rsArg; args[2] = &wsArg; args[3] = &bArg; args[4] = &oArg;
        args[5] = &rowsArg; args[6] = &colsArg; args[7] = &actArg;
        long n = (long)rows * cols;
        uint grid = (uint)Math.Min((n + 255) / 256, 65535);
        CudaDriverApi.cuLaunchKernel(outF16 ? _w8a8DequantBiasF16 : _w8a8DequantBiasF32,
            grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Dequant epilogue writing a column SLICE into a wider destination row. <paramref name="d32"/> is
    /// a contiguous <c>[rows, cols]</c> int32 tile; <paramref name="wScale"/>/<paramref name="bias"/> must be
    /// pre-offset to that slice. Lets the caller tile the GEMM over N so the accumulator stays in L2.</summary>
    public unsafe void LaunchW8A8DequantBiasStrided(ulong output, ulong d32, ulong rowScale, ulong wScale, ulong bias,
        int rows, int cols, int outStride, nint stream, bool outF16, uint actMode = 0)
    {
        if (_w8a8Module is null) throw new InvalidOperationException("w8a8.ptx not present in the Ptx folder.");
        ulong dArg = d32, rsArg = rowScale, wsArg = wScale, bArg = bias, oArg = output;
        uint rowsArg = (uint)rows, colsArg = (uint)cols, strideArg = (uint)outStride, actArg = actMode;
        void** args = stackalloc void*[9];
        args[0] = &dArg; args[1] = &rsArg; args[2] = &wsArg; args[3] = &bArg; args[4] = &oArg;
        args[5] = &rowsArg; args[6] = &colsArg; args[7] = &strideArg; args[8] = &actArg;
        long n = (long)rows * cols;
        uint grid = (uint)Math.Min((n + 255) / 256, 65535);
        CudaDriverApi.cuLaunchKernel(outF16 ? _w8a8DequantBiasStridedF16 : _w8a8DequantBiasStridedF32,
            grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the optional convrot.ptx module was found and loaded (src/HartsyInference.Cuda/Kernels/dequant/convrot.cu).</summary>
    public bool HasConvRotKernels => _convRotModule is not null;

    /// <summary>ConvRot rotation — out = x @ H per contiguous <paramref name="group"/>-wide slice, over <c>count</c>
    /// elements (a multiple of <paramref name="group"/>). Out-of-place; <paramref name="output"/> may not alias x.</summary>
    public unsafe void LaunchConvRotRotate(ulong output, ulong x, long count, int group, nint stream, bool srcF16)
    {
        if (_convRotModule is null) throw new InvalidOperationException("convrot.ptx not present in the Ptx folder.");
        if (group < 4 || count % group != 0)
            throw new ArgumentException($"ConvRot needs count ({count}) to be a multiple of group ({group}).", nameof(count));

        // 1024 shared floats per block regardless of group size, so a small group just packs more groups per block.
        uint quarter = (uint)(group >> 2);
        uint groupsPerBlock = quarter >= 256 ? 1u : Math.Max(1u, 256u / quarter);
        ulong totalGroups = (ulong)(count / group);
        uint sharedBytes = (uint)((long)groupsPerBlock * group * sizeof(float));

        ulong xArg = x, oArg = output;
        uint groupArg = (uint)group, gpbArg = groupsPerBlock;
        ulong totalArg = totalGroups;
        void** args = stackalloc void*[5];
        args[0] = &xArg; args[1] = &oArg; args[2] = &groupArg; args[3] = &gpbArg; args[4] = &totalArg;
        ulong blocks = (totalGroups + groupsPerBlock - 1) / groupsPerBlock;
        CudaDriverApi.cuLaunchKernel(srcF16 ? _convRotRotateF16 : _convRotRotateF32,
            (uint)blocks, 1, 1, 256, 1, 1, sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Shared-memory floats the fused ConvRot+quant kernel may use for one row. 8192 floats = 32 KB,
    /// which keeps several blocks per SM on Ada; wider rows keep the two-kernel path.</summary>
    private const int FusedConvRotMaxCols = 8192;

    /// <summary>Whether <see cref="LaunchConvRotQuantRowwise"/> can serve this row width — the fused kernel
    /// stages the whole row in shared memory, so <paramref name="cols"/> is bounded and must divide by group.</summary>
    /// <remarks><paramref name="group"/> must be a power of FOUR: the Hadamard is kron(h4, ..., h4), so the
    /// rotation is exactly log4(group) radix-4 stages and a group like 128 has no valid decomposition.</remarks>
    public bool HasFusedConvRotQuant(int cols, int group) =>
        _convRotModule is not null && cols > 0 && cols <= FusedConvRotMaxCols
        && group >= 4 && cols % group == 0
        && (group & (group - 1)) == 0 && (group & 0x55555554) != 0;

    /// <summary>Fused ConvRot rotation + per-row dynamic int8 quantization: replaces
    /// <see cref="LaunchConvRotRotate"/> followed by <see cref="LaunchW8A8QuantRowwise"/>, dropping the
    /// full-width rotated intermediate (7 bytes/element of traffic down to 3). Bit-identical to that pair.</summary>
    /// <summary>Whether the gated variant can serve this head layout: F16 source (the DiT activation dtype the
    /// gate kernel writes), a head count the per-block sigmoid cache holds, and a power-of-two head dim (the
    /// inner loop shifts rather than divides).</summary>
    public bool HasFusedGatedConvRotQuant(int cols, int group, bool srcF16, int heads, int headDim) =>
        HasFusedConvRotQuant(cols, group) && srcF16 && heads > 0 && heads <= 128
        && headDim > 0 && (headDim & (headDim - 1)) == 0 && heads * headDim == cols;

    /// <inheritdoc cref="LaunchConvRotQuantRowwise(ulong,ulong,ulong,int,int,int,nint,bool)"/>
    /// <summary>Fused ConvRot + quant with LTX-2's per-head output gate applied on load, replacing a separate
    /// <see cref="LaunchLtx2HeadGate"/> pass over the same tensor. Bit-identical to that pair.</summary>
    public unsafe void LaunchConvRotQuantRowwiseGated(ulong q, ulong rowScale, ulong x, int rows, int cols,
        int group, nint stream, ulong logits, int heads, int headDim)
    {
        if (!HasFusedGatedConvRotQuant(cols, group, srcF16: true, heads, headDim))
            throw new InvalidOperationException(
                $"fused gated ConvRot+quant unavailable for cols={cols}, group={group}, heads={heads}, headDim={headDim}.");
        ulong xArg = x, qArg = q, sArg = rowScale, lArg = logits;
        uint colsArg = (uint)cols, groupArg = (uint)group, headsArg = (uint)heads;
        uint shiftArg = (uint)System.Numerics.BitOperations.TrailingZeroCount((uint)headDim);
        void** args = stackalloc void*[8];
        args[0] = &xArg; args[1] = &qArg; args[2] = &sArg; args[3] = &colsArg; args[4] = &groupArg;
        args[5] = &lArg; args[6] = &headsArg; args[7] = &shiftArg;
        uint gatedShared = (uint)((long)cols * sizeof(float));
        CudaDriverApi.cuLaunchKernel(_convRotQuantGatedF16,
            (uint)rows, 1, 1, 256, 1, 1, gatedShared, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchConvRotQuantRowwise(ulong q, ulong rowScale, ulong x, int rows, int cols,
        int group, nint stream, bool srcF16)
    {
        if (!HasFusedConvRotQuant(cols, group))
            throw new InvalidOperationException($"fused ConvRot+quant unavailable for cols={cols}, group={group}.");
        ulong xArg = x, qArg = q, sArg = rowScale;
        uint colsArg = (uint)cols, groupArg = (uint)group;
        void** args = stackalloc void*[5];
        args[0] = &xArg; args[1] = &qArg; args[2] = &sArg; args[3] = &colsArg; args[4] = &groupArg;
        uint sharedBytes = (uint)((long)cols * sizeof(float));
        CudaDriverApi.cuLaunchKernel(srcF16 ? _convRotQuantF16 : _convRotQuantF32,
            (uint)rows, 1, 1, 256, 1, 1, sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the fused-dequant int8 mma GEMM can serve this shape. The kernel tiles 128x128x64 with
    /// no N/K remainder handling (M is predicated), so K and N must be whole multiples; every LTX-2.5 DiT shape
    /// is. Callers fall back to <see cref="Int8GemmExecutor"/> + the separate dequant kernel otherwise.</summary>
    /// <summary>The fused mma GEMM's CUfunction, for occupancy/register diagnostics. 0 when the module is absent.</summary>
    internal nint Int8MmaGemmFunction => _int8MmaGemmF16;

    /// <summary>The padded control entry point's CUfunction. Its register count is the check that the A/B baseline
    /// really is the shipped kernel — if it moved, the control was contaminated by the edit and the delta is void.</summary>
    internal nint Int8MmaGemmPadFunction => _int8MmaGemmF16Pad;

    /// <summary>Dynamic shared bytes the SWIZZLED fused mma GEMM needs: STAGES × (BM + BN) rows × BK, unpadded.
    /// MUST track <c>int8_mma_gemm.cu</c> — it is both the launch argument and the occupancy budget the driver
    /// reasons about, so an over-estimate here costs registers, not just address space.</summary>
    internal const uint Int8MmaSharedBytes = 3u * (128u + 256u) * 64u;

    /// <summary>Dynamic shared bytes the PADDED control entry point needs (rows × <c>BK+16</c>). The two kernels
    /// must never be launched with each other's budget: over-budgeting the swizzled one throws away exactly the
    /// occupancy headroom the unpadded layout buys, and under-budgeting the padded one corrupts its last stage.</summary>
    internal const uint Int8MmaSharedBytesPad = 3u * (128u + 256u) * 80u;

    /// <summary>Selects the swizzled (default) or padded-control operand layout — <c>HARTSY_INT8_MMA_SWIZZLE=0</c>
    /// picks the padded kernel the swizzle replaced. This is an A/B control, NOT the feature kill switch; that is
    /// <c>HARTSY_INT8_FUSED_MMA=0</c>, which drops to cuBLASLt + a separate dequant entirely.</summary>
    internal static readonly bool Int8MmaSwizzle = Environment.GetEnvironmentVariable("HARTSY_INT8_MMA_SWIZZLE") != "0";

    /// <summary>N tile of the fused mma GEMM; N must be a whole multiple (M is predicated, N and K are not).</summary>
    private const int Int8MmaTileN = 256;

    public bool HasInt8MmaGemm(int m, int n, int k) =>
        _int8MmaModule is not null && m > 0 && n % Int8MmaTileN == 0 && k % 64 == 0;

    /// <summary>Fused int8 GEMM + W8A8 dequant: <c>D[M,N] = act(actScale[m]·wScale[n]·(A·Bᵀ) + bias[n])</c>,
    /// F16 out. Replaces the cuBLASLt GEMM + <c>w8a8_dequant_bias</c> pair, so the int32 accumulator never
    /// reaches HBM. <paramref name="actMode"/> matches that kernel's (0 none, 1 gelu-tanh).</summary>
    public unsafe void LaunchInt8MmaGemmDequant(ulong d, ulong a, ulong b, ulong actScale, ulong wScale,
        ulong bias, int m, int n, int k, uint actMode, nint stream)
        => LaunchInt8MmaGemmDequant(d, a, b, actScale, wScale, bias, m, n, k, actMode, stream, Int8MmaSwizzle);

    /// <param name="swizzle">Layout arm. Exposed so a benchmark can time both entry points against the same cold
    /// buffers in ONE process — the env var is read once at type init, so an env-only switch would force separate
    /// processes and reintroduce the between-run spread the comparison is trying to resolve.</param>
    /// <inheritdoc cref="LaunchInt8MmaGemmDequant(ulong,ulong,ulong,ulong,ulong,ulong,int,int,int,uint,nint)"/>
    public unsafe void LaunchInt8MmaGemmDequant(ulong d, ulong a, ulong b, ulong actScale, ulong wScale,
        ulong bias, int m, int n, int k, uint actMode, nint stream, bool swizzle)
    {
        if (!HasInt8MmaGemm(m, n, k))
            throw new InvalidOperationException($"fused int8 mma GEMM unavailable for {m}x{n}x{k}.");
        ulong dA = d, aA = a, bA = b, asA = actScale, wsA = wScale, biA = bias;
        uint mA = (uint)m, nA = (uint)n, kA = (uint)k, actA = actMode;
        void** args = stackalloc void*[10];
        args[0] = &dA; args[1] = &aA; args[2] = &bA; args[3] = &asA; args[4] = &wsA; args[5] = &biA;
        args[6] = &mA; args[7] = &nA; args[8] = &kA; args[9] = &actA;
        // Function and shared budget are picked together — they are not independently valid.
        nint fn = swizzle ? _int8MmaGemmF16 : _int8MmaGemmF16Pad;
        uint shared = swizzle ? Int8MmaSharedBytes : Int8MmaSharedBytesPad;
        uint gridX = (uint)(n / Int8MmaTileN), gridY = (uint)((m + 127) / 128);
        CudaDriverApi.cuLaunchKernel(fn, gridX, gridY, 1, 256, 1, 1, shared, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the optional dequant_nvfp4_to_f16.ptx module was found and loaded (src/HartsyInference.Cuda/Kernels/dequant).</summary>
    public bool HasNvfp4Kernels => _nvfp4Module is not null;

    /// <summary>NVFP4 unpack — packed <c>[rows, halfCols]</c> E2M1 bytes × swizzled E4M3 block scales × the two
    /// scalars → dense F16 or BF16 <c>[rows, 2·halfCols]</c>.</summary>
    /// <param name="paddedCols">Last-dim length of the stored block-scale tensor (its swizzle stride).</param>
    /// <param name="scaleFactor">The block-scale tensor's own <c>Fp8ScaleFactor</c>; passed separately from
    /// <paramref name="globalScale"/> so the product is formed in the host reference's order.</param>
    public unsafe void LaunchNvfp4Dequant(ulong output, ulong weight, ulong blockScale,
        int rows, int halfCols, int paddedCols, float scaleFactor, float globalScale, nint stream, bool outBf16)
    {
        if (_nvfp4Module is null) throw new InvalidOperationException("dequant_nvfp4_to_f16.ptx not present in the Ptx folder.");
        ulong wArg = weight, sArg = blockScale, oArg = output;
        uint rowsArg = (uint)rows, halfColsArg = (uint)halfCols, paddedArg = (uint)paddedCols;
        float sfArg = scaleFactor, gsArg = globalScale;
        void** args = stackalloc void*[8];
        args[0] = &wArg; args[1] = &sArg; args[2] = &oArg;
        args[3] = &rowsArg; args[4] = &halfColsArg; args[5] = &paddedArg;
        args[6] = &sfArg; args[7] = &gsArg;
        const uint BlockSize = 256;
        uint gridX = (uint)(((long)halfCols + BlockSize - 1) / BlockSize);
        uint gridY = (uint)Math.Min(rows, 65535);
        CudaDriverApi.cuLaunchKernel(outBf16 ? _nvfp4DequantBf16 : _nvfp4DequantF16,
            gridX, gridY, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the optional sage_attn_int8.ptx module was found and loaded (src/HartsyInference.Cuda/Kernels/attention).</summary>
    public bool HasSageAttentionKernels => _sageAttnModule is not null;

    /// <summary>SageAttention prologue 1/3 — K per-channel mean over the sequence: kmean[b,h,d]. Grid (H,B), block 128.</summary>
    public unsafe void LaunchSageKMean(ulong kmean, ulong k, int b, int h, int skv, int d, nint stream, bool srcF16 = false)
    {
        ThrowIfNoSageModule();
        ulong kmeanArg = kmean, kArg = k;
        uint bArg = (uint)b, hArg = (uint)h, skvArg = (uint)skv, dArg = (uint)d;
        void** args = stackalloc void*[6];
        args[0] = &kmeanArg; args[1] = &kArg; args[2] = &bArg; args[3] = &hArg; args[4] = &skvArg; args[5] = &dArg;
        CudaDriverApi.cuLaunchKernel(srcF16 ? _sageKMeanF16H : _sageKMeanF32, (uint)h, (uint)b, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>SageAttention prologue 2/3 — per-row absmax INT8 quant of Q with the softmax scale folded into the row scale.</summary>
    public unsafe void LaunchSageQuantQ(ulong q8, ulong qscale, ulong q, int b, int h, int sq, int d, float attnScale, nint stream, bool srcF16 = false)
    {
        ThrowIfNoSageModule();
        ulong q8Arg = q8, qsArg = qscale, qArg = q;
        uint bArg = (uint)b, hArg = (uint)h, sqArg = (uint)sq, dArg = (uint)d;
        float scaleArg = attnScale;
        void** args = stackalloc void*[8];
        args[0] = &q8Arg; args[1] = &qsArg; args[2] = &qArg;
        args[3] = &bArg; args[4] = &hArg; args[5] = &sqArg; args[6] = &dArg; args[7] = &scaleArg;
        uint rows = (uint)((long)b * h * sq);
        CudaDriverApi.cuLaunchKernel(srcF16 ? _sageQuantQInt8F16H : _sageQuantQInt8F32, (rows + 3) / 4, 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>SageAttention prologue 3/3 — per-row absmax INT8 quant of (K − kmean) for outlier-recovering channel smoothing.</summary>
    public unsafe void LaunchSageQuantK(ulong k8, ulong kscale, ulong k, ulong kmean, int b, int h, int skv, int d, nint stream, bool srcF16 = false)
    {
        ThrowIfNoSageModule();
        ulong k8Arg = k8, ksArg = kscale, kArg = k, kmeanArg = kmean;
        uint bArg = (uint)b, hArg = (uint)h, skvArg = (uint)skv, dArg = (uint)d;
        void** args = stackalloc void*[8];
        args[0] = &k8Arg; args[1] = &ksArg; args[2] = &kArg; args[3] = &kmeanArg;
        args[4] = &bArg; args[5] = &hArg; args[6] = &skvArg; args[7] = &dArg;
        uint rows = (uint)((long)b * h * skv);
        CudaDriverApi.cuLaunchKernel(srcF16 ? _sageQuantKInt8F16H : _sageQuantKInt8F32, (rows + 3) / 4, 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Whether the register-resident v1 flash kernel is available (no Sq%32 restriction).</summary>
    public bool HasSageV1 => _sageAttnV1Module is not null;

    /// <summary>True when the v1 path will actually be dispatched (module present, not forced off via
    /// HARTSY_SAGE_V0=1) — the caller must then provide the pre-transposed F16 V workspace.</summary>
    public bool UseSageV1 => _sageAttnV1Module is not null && Environment.GetEnvironmentVariable("HARTSY_SAGE_V0") != "1";

    /// <summary>v1 prologue — one-shot V transpose+cast: [B,H,Skv,D] f32 → [B,H,D,skvPad] f16, amortizing per-tile re-transpose.</summary>
    public unsafe void LaunchSageVF16T(ulong vt16, ulong v, int b, int h, int skv, int d, nint stream, bool srcF16 = false)
    {
        ThrowIfNoSageModule();
        ulong vtArg = vt16, vArg = v;
        uint bArg = (uint)b, hArg = (uint)h, skvArg = (uint)skv, dArg = (uint)d;
        void** args = stackalloc void*[6];
        args[0] = &vtArg; args[1] = &vArg; args[2] = &bArg; args[3] = &hArg; args[4] = &skvArg; args[5] = &dArg;
        // skvPad = Skv rounded up to 8 (16-byte cp.async source alignment), amortizing the flash kernel's
        // otherwise-repeated per-query-tile re-transpose (Sq/64 times).
        long skvPad = (skv + 7L) & ~7L;
        if (skvPad >= 2048 && (skvPad & (skvPad - 1)) == 0) skvPad += 8;   // anti-aliasing pad (see sage_skv_pad)
        long total = (long)b * h * d * skvPad;
        uint grid = (uint)Math.Min(4096, (total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(srcF16 ? _sageVF16TH : _sageVF16T, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>SageAttention main kernel — fused INT8-QK^T flash attention (online softmax); dispatches v1 when compiled, else v0.</summary>
    public unsafe void LaunchSageAttnInt8(ulong outPtr, ulong q8, ulong qscale, ulong k8, ulong kscale, ulong v,
        ulong vt16, int b, int h, int sq, int skv, int d, nint stream, bool f16Io = false)
    {
        ThrowIfNoSageModule();
        ulong outArg = outPtr, q8Arg = q8, qsArg = qscale, k8Arg = k8, ksArg = kscale, vArg = v;
        uint bArg = (uint)b, hArg = (uint)h, sqArg = (uint)sq, skvArg = (uint)skv;
        // Register-resident mma.sync v1 (grid ceil(Sq/64)×H×B, block 128, ~24.3 KB SMEM, any Sq) when compiled,
        // else the wmma v0 (grid Sq/32×H×B, block 64 — caller must gate Sq % 32 == 0 for v0).
        // HARTSY_SAGE_V0=1 forces v0 for debugging.
        if (UseSageV1)
        {
            if (vt16 == 0) throw new ArgumentException("v1 path requires the pre-transposed V workspace (LaunchSageVF16T).");
            ulong vtArg = vt16;
            void** args = stackalloc void*[10];
            args[0] = &outArg; args[1] = &q8Arg; args[2] = &qsArg; args[3] = &k8Arg; args[4] = &ksArg; args[5] = &vtArg;
            args[6] = &bArg; args[7] = &hArg; args[8] = &sqArg; args[9] = &skvArg;
            // SMEM: TWO cp.async stages of (K8 s8 [BC*D] + Vt f16 [D*BC] + kscale f32 [BC]), BR=64/BC=32
            // in-kernel (BC=32 keeps d128 at 155 regs / 0 spills / 3 blocks-per-SM; 2 stages ≈ 24.8 KB < 48 KB).
            const int bc = 32;
            uint smemBytes = (uint)(2 * (bc * d + d * bc * 2 + bc * sizeof(float)));
            // E1 experiment knob: HARTSY_SAGE_PV=f16acc selects the F16-accumulate PV variant (2× PV mma
            // rate on GeForce Ampere; P pre-scaled 1/16 in-kernel for overflow headroom to ~349k keys).
            bool f16Acc = Environment.GetEnvironmentVariable("HARTSY_SAGE_PV") == "f16acc";
            if (f16Io && !UseSageV1)
                throw new InvalidOperationException("F16-ingest Sage requires the v1 module (HARTSY_SAGE_V0=1 is F32-only).");
            nint func = f16Io
                ? (d == 128 ? _sageAttnV1D128F16Io : _sageAttnV1D64F16Io)   // native-F16 contract implies f16acc PV
                : d == 128
                    ? (f16Acc ? _sageAttnV1D128F16Acc : _sageAttnV1D128)
                    : (f16Acc ? _sageAttnV1D64F16Acc : _sageAttnV1D64);
            CudaDriverApi.cuLaunchKernel(func, (uint)((sq + 63) / 64), (uint)h, (uint)b, 128, 1, 1,
                smemBytes, stream, (nint)args, 0).ThrowOnError();
            return;
        }
        uint dArg = (uint)d;
        void** args0 = stackalloc void*[11];
        args0[0] = &outArg; args0[1] = &q8Arg; args0[2] = &qsArg; args0[3] = &k8Arg; args0[4] = &ksArg; args0[5] = &vArg;
        args0[6] = &bArg; args0[7] = &hArg; args0[8] = &sqArg; args0[9] = &skvArg; args0[10] = &dArg;
        // SMEM: Vsh f32 [16*D] + Ssh f32 [32*16] + Osh f32 [32*D] + K8sh s8 [16*D] (kernel-side BR=32/BC=16).
        uint smemBytes0 = (uint)(16 * d * sizeof(float) + 32 * 16 * sizeof(float) + 32 * d * sizeof(float) + 16 * d);
        CudaDriverApi.cuLaunchKernel(_sageAttnInt8F32, (uint)(sq / 32), (uint)h, (uint)b, 64, 1, 1,
            smemBytes0, stream, (nint)args0, 0).ThrowOnError();
    }

    private void ThrowIfNoSageModule()
    {
        if (_sageAttnModule is null)
            throw new InvalidOperationException("sage_attn_int8.ptx is not loaded; gate on HasSageAttentionKernels first.");
    }

    /// <summary>Launches elementwise add: output[i] = a[i] + b[i] (F16)</summary>
    public void LaunchAddF16(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_addF16, output, a, b, count, stream);

    /// <summary>Launches elementwise add: output[i] = a[i] + b[i] (BF16)</summary>
    public void LaunchAddBf16(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_addBf16, output, a, b, count, stream);

    /// <summary>Launches elementwise multiply: output[i] = a[i] * b[i] (F32)</summary>
    public void LaunchMul(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_mulF32, output, a, b, count, stream);

    /// <summary>Launches elementwise multiply: output[i] = a[i] * b[i] (F16)</summary>
    public void LaunchMulF16(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_mulF16, output, a, b, count, stream);

    /// <summary>Launches elementwise multiply: output[i] = a[i] * b[i] (BF16)</summary>
    public void LaunchMulBf16(ulong output, ulong a, ulong b, int count, nint stream)
        => LaunchBinaryImpl(_mulBf16, output, a, b, count, stream);

    /// <summary>Launches elementwise scale: output[i] = input[i] * scalar (F32)</summary>
    public void LaunchScale(ulong output, ulong input, float scalar, int count, nint stream)
        => LaunchScaleImpl(_scaleF32, output, input, scalar, count, stream);

    /// <summary>Launches elementwise scale: output[i] = input[i] * scalar (F16)</summary>
    public void LaunchScaleF16(ulong output, ulong input, float scalar, int count, nint stream)
        => LaunchScaleImpl(_scaleF16, output, input, scalar, count, stream);

    /// <summary>Launches elementwise scale: output[i] = input[i] * scalar (BF16)</summary>
    public void LaunchScaleBf16(ulong output, ulong input, float scalar, int count, nint stream)
        => LaunchScaleImpl(_scaleBf16, output, input, scalar, count, stream);

    /// <summary>Launches SiLU activation: output[i] = input[i] * sigmoid(input[i]) (F32)</summary>
    public void LaunchSilu(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_siluF32, output, input, count, stream);

    /// <summary>Launches SiLU activation: output[i] = input[i] * sigmoid(input[i]) (F16)</summary>
    public void LaunchSiluF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_siluF16, output, input, count, stream);

    /// <summary>Launches SiLU activation: output[i] = input[i] * sigmoid(input[i]) (BF16)</summary>
    public void LaunchSiluBf16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_siluBf16, output, input, count, stream);

    /// <summary>Launches GELU activation (F32)</summary>
    public void LaunchGelu(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_geluF32, output, input, count, stream);

    /// <summary>Launches GELU activation (F16)</summary>
    public void LaunchGeluF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_geluF16, output, input, count, stream);

    /// <summary>Launches GELU activation (BF16)</summary>
    public void LaunchGeluBf16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_geluBf16, output, input, count, stream);

    /// <summary>Launches elementwise clamp: output[i] = clamp(input[i], min, max) (F32)</summary>
    public void LaunchClamp(ulong output, ulong input, float min, float max, int count, nint stream)
        => LaunchClampImpl(_clampF32, output, input, min, max, count, stream);

    /// <summary>Launches elementwise clamp: output[i] = clamp(input[i], min, max) (F16)</summary>
    public void LaunchClampF16(ulong output, ulong input, float min, float max, int count, nint stream)
        => LaunchClampImpl(_clampF16, output, input, min, max, count, stream);

    /// <summary>Launches elementwise clamp: output[i] = clamp(input[i], min, max) (BF16)</summary>
    public void LaunchClampBf16(ulong output, ulong input, float min, float max, int count, nint stream)
        => LaunchClampImpl(_clampBf16, output, input, min, max, count, stream);

    // ── Normalization Launches ──────────────────────────────────────────

    /// <summary>Launches GroupNorm (F32). Each block handles one (batch, group) pair.</summary>
    public void LaunchGroupNorm(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormF32, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches GroupNorm with F16 I/O, FP32 accumulation.</summary>
    public void LaunchGroupNormF16(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormF16, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches GroupNorm with BF16 I/O, FP32 accumulation, for SDXL VAE where F16 activations overflow (±65504).</summary>
    public void LaunchGroupNormBf16(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormBf16, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches LayerNorm (F32). Each block handles one row.</summary>
    public void LaunchLayerNorm(ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
        => LaunchLayerNormImpl(_layernormF32, output, input, weight, bias, normDim, totalRows, eps, stream);

    /// <summary>Launches LayerNorm with F16 I/O, FP32 accumulation.</summary>
    public void LaunchLayerNormF16(ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
        => LaunchLayerNormImpl(_layernormF16, output, input, weight, bias, normDim, totalRows, eps, stream);

    /// <summary>Launches LayerNorm with BF16 I/O, FP32 accumulation.</summary>
    public void LaunchLayerNormBf16(ulong output, ulong input, ulong weight, ulong bias,
        int normDim, int totalRows, float eps, nint stream)
        => LaunchLayerNormImpl(_layernormBf16, output, input, weight, bias, normDim, totalRows, eps, stream);

    // ── DiT glue Launches (F32) ─────────────────────────────────────────

    /// <summary>Launches RMSNorm: one block per row, reduces over normDim, applies weight (also serves per-head QK-RMSNorm).</summary>
    public void LaunchRmsNorm(ulong output, ulong input, ulong weight, int normDim, int totalRows, float eps, nint stream)
        => LaunchRmsNormImpl(_ditRmsNormF32, output, input, weight, normDim, totalRows, eps, stream);

    /// <summary>RMSNorm with F16 I/O (F32 accumulate, F32 weight). Same launch geometry as the F32 kernel.</summary>
    public void LaunchRmsNormF16(ulong output, ulong input, ulong weight, int normDim, int totalRows, float eps, nint stream)
        => LaunchRmsNormImpl(_ditRmsNormF16, output, input, weight, normDim, totalRows, eps, stream);

    /// <summary>RMSNorm with BF16 I/O (F32 accumulate, F32 weight). Same launch geometry as the F32 kernel.</summary>
    public void LaunchRmsNormBf16(ulong output, ulong input, ulong weight, int normDim, int totalRows, float eps, nint stream)
        => LaunchRmsNormImpl(_ditRmsNormBf16, output, input, weight, normDim, totalRows, eps, stream);

    /// <summary>Launches broadcast affine over the last dim: out = in*scale + shift (shift optional, pass 0 to skip).</summary>
    public void LaunchAffineBroadcastLastDim(ulong output, ulong input, ulong scale, ulong shift, int seqLen, int dim, long total, nint stream)
        => LaunchAffineBroadcastImpl(_ditAffineBroadcastF32, output, input, scale, shift, seqLen, dim, total, stream);

    /// <summary>Broadcast affine over the last dim, F16 activation I/O (scale/shift stay F32).</summary>
    public void LaunchAffineBroadcastLastDimF16(ulong output, ulong input, ulong scale, ulong shift, int seqLen, int dim, long total, nint stream)
        => LaunchAffineBroadcastImpl(_ditAffineBroadcastF16, output, input, scale, shift, seqLen, dim, total, stream);

    /// <summary>Broadcast affine over the last dim, BF16 activation I/O (scale/shift stay F32).</summary>
    public void LaunchAffineBroadcastLastDimBf16(ulong output, ulong input, ulong scale, ulong shift, int seqLen, int dim, long total, nint stream)
        => LaunchAffineBroadcastImpl(_ditAffineBroadcastBf16, output, input, scale, shift, seqLen, dim, total, stream);

    /// <summary>Row-indexed affine with the table gather and the <c>1 +</c> folded in (shift optional, pass 0 to skip).</summary>
    public void LaunchAffineBroadcastRowIndexed(ulong output, ulong input, ulong scaleTable, ulong shiftTable, ulong rowIndex, int dim, long total, nint stream)
        => LaunchAffineBroadcastRowIndexedImpl(_ditAffineBroadcastRowIndexedF32, output, input, scaleTable, shiftTable, rowIndex, dim, total, stream);

    /// <summary>Row-indexed affine, F16 activation I/O (the modulation table stays F32).</summary>
    public void LaunchAffineBroadcastRowIndexedF16(ulong output, ulong input, ulong scaleTable, ulong shiftTable, ulong rowIndex, int dim, long total, nint stream)
        => LaunchAffineBroadcastRowIndexedImpl(_ditAffineBroadcastRowIndexedF16, output, input, scaleTable, shiftTable, rowIndex, dim, total, stream);

    /// <summary>Row-indexed affine, BF16 activation I/O (the modulation table stays F32).</summary>
    public void LaunchAffineBroadcastRowIndexedBf16(ulong output, ulong input, ulong scaleTable, ulong shiftTable, ulong rowIndex, int dim, long total, nint stream)
        => LaunchAffineBroadcastRowIndexedImpl(_ditAffineBroadcastRowIndexedBf16, output, input, scaleTable, shiftTable, rowIndex, dim, total, stream);

    /// <summary>Launches GQA K/V head repeat (block pattern): [B,Hkv,L,D] → [B,Hkv*group,L,D].</summary>
    public void LaunchRepeatKv(ulong output, ulong input, int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
        => LaunchRepeatKvImpl(_lmRepeatKvF32, output, input, kvHeads, group, seqLen, headDim, total, stream);

    /// <summary>GQA K/V head repeat with F16 I/O (pure copy — no accumulation).</summary>
    public void LaunchRepeatKvF16(ulong output, ulong input, int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
        => LaunchRepeatKvImpl(_ditRepeatKvF16, output, input, kvHeads, group, seqLen, headDim, total, stream);

    /// <summary>GQA K/V head repeat with BF16 I/O; reuses the 16-bit bit-copy kernel without numeric conversion.</summary>
    public void LaunchRepeatKvBf16(ulong output, ulong input, int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
        => LaunchRepeatKvImpl(_ditRepeatKvF16, output, input, kvHeads, group, seqLen, headDim, total, stream);

    private static void ValidateFlashAttentionHeadDim(int headDim)
    {
        if (headDim <= 0 || headDim > 1024)
            throw new ArgumentOutOfRangeException(nameof(headDim), "FlashAttention requires headDim in [1, 1024].");
    }

    /// <summary>Launches FlashAttention (one block per (b, q-head, q-row); blockDim is the next power of two at
    /// least 32 and at least headDim; shared memory matches blockDim floats). <paramref name="lk"/> is the K/V
    /// buffer seq stride; <paramref name="kvLen"/> the valid key count.</summary>
    public unsafe void LaunchFlashAttention(
        ulong outPtr, ulong q, ulong k, ulong v,
        int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen, int kvGroup,
        bool causal, int qOffset, float scale, float softcap, ulong sink, int slidingWindow,
        ulong alibiSlopes, nint stream, ulong dPos = 0)
    {
        ValidateFlashAttentionHeadDim(headDim);
        ulong outArg = outPtr, qArg = q, kArg = k, vArg = v, sinkArg = sink, alibiArg = alibiSlopes, dPosArg = dPos;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim;
        uint hkvArg = (uint)hkv, lkArg = (uint)lk, kvLenArg = (uint)kvLen, grpArg = (uint)kvGroup;
        int causalArg = causal ? 1 : 0, offArg = qOffset, swArg = slidingWindow;
        float scaleArg = scale, softcapArg = softcap;

        void** args = stackalloc void*[20];
        args[0] = &outArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bArg; args[5] = &hqArg; args[6] = &tqArg; args[7] = &dArg;
        args[8] = &hkvArg; args[9] = &lkArg; args[10] = &kvLenArg; args[11] = &grpArg;
        args[12] = &causalArg; args[13] = &offArg; args[14] = &scaleArg; args[15] = &softcapArg;
        args[16] = &sinkArg; args[17] = &swArg; args[18] = &alibiArg; args[19] = &dPosArg;

        // At least one full warp is required by the full-mask shuffle. Padding lanes contribute zero.
        uint blockThreads = 32;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq);
        uint sharedBytes = blockThreads * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _flashAttnF32, gridDim, 1, 1, blockThreads, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Same as <see cref="LaunchFlashAttention"/> but <paramref name="k"/>/<paramref name="v"/> point at
    /// F16-storage KV buffers — the kernel upconverts to F32 on load; Q/out/scores/softmax/accumulate are
    /// unchanged. v1 scope: monolithic path only (no split-K, no graph-decode variant) — callers must not reach
    /// this with an F16 KV buffer via those paths.</summary>
    public unsafe void LaunchFlashAttentionF16Kv(
        ulong outPtr, ulong q, ulong k, ulong v,
        int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen, int kvGroup,
        bool causal, int qOffset, float scale, float softcap, ulong sink, int slidingWindow,
        ulong alibiSlopes, nint stream, ulong dPos = 0)
    {
        ValidateFlashAttentionHeadDim(headDim);
        ulong outArg = outPtr, qArg = q, kArg = k, vArg = v, sinkArg = sink, alibiArg = alibiSlopes, dPosArg = dPos;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim;
        uint hkvArg = (uint)hkv, lkArg = (uint)lk, kvLenArg = (uint)kvLen, grpArg = (uint)kvGroup;
        int causalArg = causal ? 1 : 0, offArg = qOffset, swArg = slidingWindow;
        float scaleArg = scale, softcapArg = softcap;

        void** args = stackalloc void*[20];
        args[0] = &outArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bArg; args[5] = &hqArg; args[6] = &tqArg; args[7] = &dArg;
        args[8] = &hkvArg; args[9] = &lkArg; args[10] = &kvLenArg; args[11] = &grpArg;
        args[12] = &causalArg; args[13] = &offArg; args[14] = &scaleArg; args[15] = &softcapArg;
        args[16] = &sinkArg; args[17] = &swArg; args[18] = &alibiArg; args[19] = &dPosArg;

        uint blockThreads = 32;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq);
        uint sharedBytes = blockThreads * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _flashAttnF16Kv, gridDim, 1, 1, blockThreads, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused FlashAttention-2, TF32 tensor cores, F32 accumulate — no-mask MHA, D∈{64,128}, Hkv==Hq.
    /// Q/out [B,Hq,Sq,D], K/V [B,Hq,Skv,D]. Sq must contain complete 32-row tiles. Grid (Sq/32,Hq,B);
    /// block 64 (2 warps); BR=32/BC=16 tiles.</summary>
    public unsafe void LaunchFlashAttentionV2Tf32(ulong outPtr, ulong q, ulong k, ulong v,
        int b, int hq, int sq, int skv, int d, float scale, nint stream)
    {
        const int BR = 32, BC = 16;   // must match flash_attn_v2_tf32.cu #defines
        if (outPtr == 0 || q == 0 || k == 0 || v == 0)
            throw new ArgumentException("FlashAttention-v2 requires non-null device pointers.");
        if (b <= 0 || hq <= 0 || skv <= 0)
            throw new ArgumentOutOfRangeException(nameof(b), "FlashAttention-v2 dimensions must be positive.");
        if (b > 65_535)
            throw new ArgumentOutOfRangeException(nameof(b), "FlashAttention-v2 batch count must fit CUDA grid Z (maximum 65535).");
        if (hq > 65_535)
            throw new ArgumentOutOfRangeException(nameof(hq), "FlashAttention-v2 head count must fit CUDA grid Y (maximum 65535).");
        if (sq <= 0 || sq % BR != 0)
            throw new ArgumentOutOfRangeException(nameof(sq), "FlashAttention-v2 requires complete 32-row query tiles.");
        if (d != 64 && d != 128)
            throw new ArgumentOutOfRangeException(nameof(d), "FlashAttention-v2 supports head dimensions 64 and 128 only.");
        if (!float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), "FlashAttention-v2 scale must be finite.");

        ulong oArg = outPtr, qArg = q, kArg = k, vArg = v;
        uint bA = (uint)b, hA = (uint)hq, sqA = (uint)sq, skvA = (uint)skv, dA = (uint)d;
        float scA = scale;
        void** args = stackalloc void*[10];
        args[0] = &oArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bA; args[5] = &hA; args[6] = &sqA; args[7] = &skvA; args[8] = &dA; args[9] = &scA;
        uint grid = (uint)(sq / BR);
        uint smem = (uint)((BC * d + BC * d + BR * BC + BR * d) * sizeof(float));
        CudaDriverApi.cuLaunchKernel(_flashV2Tf32, grid, (uint)hq, (uint)b, 64, 1, 1,
            smem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Flash-decoding split phase (plain path: no sink/alibi/softcap/window). Launches
    /// <c>batch·hq·tq·splits</c> blocks; each computes the partial online-softmax state (m, l, Σp·V) for
    /// its key chunk into the scratch buffers. <paramref name="chunk"/> = ceil(kvLen / splits).</summary>
    public unsafe void LaunchFlashAttentionSplit(ulong partialM, ulong partialL, ulong partialAcc,
        ulong q, ulong k, ulong v, int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen,
        int kvGroup, bool causal, int qOffset, float scale, int splits, int chunk, nint stream, ulong dPos = 0,
        float softcap = 0f, int slidingWindow = 0, bool f16Kv = false)
    {
        ValidateFlashAttentionHeadDim(headDim);
        ulong pmArg = partialM, plArg = partialL, paArg = partialAcc, qArg = q, kArg = k, vArg = v, dPosArg = dPos;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim;
        uint hkvArg = (uint)hkv, lkArg = (uint)lk, kvLenArg = (uint)kvLen, grpArg = (uint)kvGroup;
        int causalArg = causal ? 1 : 0, offArg = qOffset;
        float scaleArg = scale, softcapArg = softcap;
        int windowArg = slidingWindow;
        uint gArg = (uint)splits, chunkArg = (uint)chunk;

        void** args = stackalloc void*[22];
        args[0] = &pmArg; args[1] = &plArg; args[2] = &paArg; args[3] = &qArg; args[4] = &kArg; args[5] = &vArg;
        args[6] = &bArg; args[7] = &hqArg; args[8] = &tqArg; args[9] = &dArg;
        args[10] = &hkvArg; args[11] = &lkArg; args[12] = &kvLenArg; args[13] = &grpArg;
        args[14] = &causalArg; args[15] = &offArg; args[16] = &scaleArg; args[17] = &softcapArg; args[18] = &windowArg;
        args[19] = &gArg; args[20] = &chunkArg; args[21] = &dPosArg;

        uint blockThreads = 32;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq * splits);
        uint sharedBytes = blockThreads * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            f16Kv ? _flashAttnF32SplitF16Kv : _flashAttnF32Split, gridDim, 1, 1, blockThreads, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Flash-decoding combine phase: merges the <paramref name="splits"/> chunk partials per query
    /// into the final output via online softmax. Launches <c>batch·hq·tq</c> blocks.</summary>
    public unsafe void LaunchFlashAttentionCombine(ulong outPtr, ulong partialM, ulong partialL,
        ulong partialAcc, int batch, int hq, int tq, int headDim, int splits, nint stream)
    {
        ValidateFlashAttentionHeadDim(headDim);
        ulong outArg = outPtr, pmArg = partialM, plArg = partialL, paArg = partialAcc;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim, gArg = (uint)splits;

        void** args = stackalloc void*[9];
        args[0] = &outArg; args[1] = &pmArg; args[2] = &plArg; args[3] = &paArg;
        args[4] = &bArg; args[5] = &hqArg; args[6] = &tqArg; args[7] = &dArg; args[8] = &gArg;

        uint blockThreads = 32;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq);
        CudaDriverApi.cuLaunchKernel(
            _flashAttnF32Combine, gridDim, 1, 1, blockThreads, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches in-place KV-cache append: copies newKv [1,H,tNew,D] into buffer [1,H,maxSeq,D] at offset.</summary>
    public unsafe void LaunchKvAppend(
        ulong buffer, ulong newKv, int heads, int maxSeq, int tNew, int headDim, int offset,
        nint stream, ulong dPos = 0)
    {
        ulong bufArg = buffer, newArg = newKv, dPosArg = dPos;
        uint hArg = (uint)heads, maxArg = (uint)maxSeq, tArg = (uint)tNew, dArg = (uint)headDim, offArg = (uint)offset;
        ulong total = (ulong)heads * (ulong)tNew * (ulong)headDim;
        ulong totalArg = total;

        void** args = stackalloc void*[9];
        args[0] = &bufArg; args[1] = &newArg; args[2] = &hArg; args[3] = &maxArg;
        args[4] = &tArg; args[5] = &dArg; args[6] = &offArg; args[7] = &totalArg; args[8] = &dPosArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmKvAppendF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Same as <see cref="LaunchKvAppend"/> but the resident buffer is F16-storage (halved VRAM) —
    /// the source stays F32 (straight out of the K/V projection); the kernel converts on write.</summary>
    public unsafe void LaunchKvAppendF16(
        ulong buffer, ulong newKv, int heads, int maxSeq, int tNew, int headDim, int offset,
        nint stream, ulong dPos = 0)
    {
        ulong bufArg = buffer, newArg = newKv, dPosArg = dPos;
        uint hArg = (uint)heads, maxArg = (uint)maxSeq, tArg = (uint)tNew, dArg = (uint)headDim, offArg = (uint)offset;
        ulong total = (ulong)heads * (ulong)tNew * (ulong)headDim;
        ulong totalArg = total;

        void** args = stackalloc void*[9];
        args[0] = &bufArg; args[1] = &newArg; args[2] = &hArg; args[3] = &maxArg;
        args[4] = &tArg; args[5] = &dArg; args[6] = &offArg; args[7] = &totalArg; args[8] = &dPosArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmKvAppendF16, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches KV time-range extraction: output[1,h,t,d] = input[1,h,start+t,d] for t in [0,len).</summary>
    public unsafe void LaunchKvSliceTime(ulong output, ulong input, int heads, int tIn, int headDim, int start, int len, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint hArg = (uint)heads, tInArg = (uint)tIn, dArg = (uint)headDim, startArg = (uint)start, lenArg = (uint)len;
        ulong total = (ulong)heads * (ulong)len * (ulong)headDim;
        ulong totalArg = total;

        void** args = stackalloc void*[8];
        args[0] = &outArg; args[1] = &inArg; args[2] = &hArg; args[3] = &tInArg;
        args[4] = &dArg; args[5] = &startArg; args[6] = &lenArg; args[7] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmKvSliceTimeF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches MoE row-gather: output[m,j] = input[rowIndices[m], j]. total = M·K.</summary>
    public unsafe void LaunchGatherRows(ulong outPtr, ulong inPtr, ulong idxPtr, int k, ulong total, nint stream)
    {
        ulong outArg = outPtr, inArg = inPtr, idxArg = idxPtr;
        uint kArg = (uint)k;
        ulong totalArg = total;
        void** args = stackalloc void*[5];
        args[0] = &outArg; args[1] = &inArg; args[2] = &idxArg; args[3] = &kArg; args[4] = &totalArg;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmGatherRowsF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches MoE weighted scatter-add: output[rowIndices[m], j] += scales[m]·input[m,j]. total = M·K.</summary>
    public unsafe void LaunchScatterAddWeightedRows(ulong outPtr, ulong inPtr, ulong idxPtr, ulong scalePtr, int k, ulong total, nint stream)
    {
        ulong outArg = outPtr, inArg = inPtr, idxArg = idxPtr, scaleArg = scalePtr;
        uint kArg = (uint)k;
        ulong totalArg = total;
        void** args = stackalloc void*[6];
        args[0] = &outArg; args[1] = &inArg; args[2] = &idxArg; args[3] = &scaleArg; args[4] = &kArg; args[5] = &totalArg;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmScatterAddWeightedRowsF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused graph-decode QKV epilogue: ropes q/k and scatters q,k,v into their caches at pos.</summary>
    public unsafe void LaunchQkvRopeScatter(ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn,
        ulong cos, ulong sin, int nq, int nkv, int headDim, int rotaryDim, bool interleaved, int maxSeq, ulong devicePos, nint stream)
    {
        ulong qA = qOut, kA = kCache, vA = vCache, qiA = qIn, kiA = kIn, viA = vIn, cA = cos, sA = sin, dpA = devicePos;
        uint nqA = (uint)nq, nkvA = (uint)nkv, dA = (uint)headDim, rdA = (uint)rotaryDim, msA = (uint)maxSeq;
        int ilA = interleaved ? 1 : 0;
        void** args = stackalloc void*[15];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &qiA; args[4] = &kiA; args[5] = &viA;
        args[6] = &cA; args[7] = &sA;
        args[8] = &nqA; args[9] = &nkvA; args[10] = &dA; args[11] = &rdA; args[12] = &ilA; args[13] = &msA; args[14] = &dpA;
        long total = ((long)nq + 2L * nkv) * headDim;
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmQkvRopeScatterF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused per-head QK-RMSNorm + RoPE + KV-scatter for QK-norm models' graph decode step
    /// (bit-identical to slice + 2× RmsNorm + rope-scatter — reduction copied verbatim from
    /// dit_rmsnorm_f32 at its production geometry, one 256-thread block per head).</summary>
    public unsafe void LaunchQkNormRopeScatter(ulong qOut, ulong kCache, ulong vCache, ulong qIn, ulong kIn, ulong vIn,
        ulong qNormW, ulong kNormW, ulong cos, ulong sin, int nq, int nkv, int headDim, int rotaryDim,
        bool interleaved, float eps, int maxSeq, ulong devicePos, nint stream)
    {
        ulong qA = qOut, kA = kCache, vA = vCache, qiA = qIn, kiA = kIn, viA = vIn, qwA = qNormW, kwA = kNormW,
            cA = cos, sA = sin, dpA = devicePos;
        uint nqA = (uint)nq, nkvA = (uint)nkv, dA = (uint)headDim, rdA = (uint)rotaryDim, msA = (uint)maxSeq;
        int ilA = interleaved ? 1 : 0;
        float epsA = eps;
        void** args = stackalloc void*[18];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &qiA; args[4] = &kiA; args[5] = &viA;
        args[6] = &qwA; args[7] = &kwA; args[8] = &cA; args[9] = &sA;
        args[10] = &nqA; args[11] = &nkvA; args[12] = &dA; args[13] = &rdA; args[14] = &ilA;
        args[15] = &epsA; args[16] = &msA; args[17] = &dpA;
        uint grid = (uint)(nq + 2 * nkv);
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(_lmQkNormRopeScatterF32, grid, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused residual-add + RMSNorm (bit-identical to Add followed by the RmsNorm kernel — same
    /// reduction structure and launch geometry).</summary>
    public unsafe void LaunchAddRmsNorm(ulong residOut, ulong normOut, ulong a, ulong b, ulong weight,
        int normDim, int totalRows, float eps, nint stream)
    {
        ulong rA = residOut, nA = normOut, aA = a, bA = b, wA = weight;
        uint dimA = (uint)normDim, rowsA = (uint)totalRows;
        float epsA = eps;
        void** args = stackalloc void*[8];
        args[0] = &rA; args[1] = &nA; args[2] = &aA; args[3] = &bA; args[4] = &wA;
        args[5] = &dimA; args[6] = &rowsA; args[7] = &epsA;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(_lmAddRmsNormF32, (uint)totalRows, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused gated-FFN activation epilogue over the concatenated [gate | up] projection output
    /// (act 0 = SiLU, 1 = GELU-tanh): one elementwise pass, no slice/activation/multiply intermediates.</summary>
    public unsafe void LaunchGluAct(ulong comb, ulong gateUp, int rows, int ff, bool gelu, nint stream)
        => LaunchGluActImpl(_lmGluActF32, comb, gateUp, rows, ff, gelu, stream);

    /// <summary>Gated-FFN activation epilogue with BF16 I/O (F32 compute). Same launch geometry as the F32 kernel.</summary>
    public unsafe void LaunchGluActBf16(ulong comb, ulong gateUp, int rows, int ff, bool gelu, nint stream)
        => LaunchGluActImpl(_ditGluActBf16, comb, gateUp, rows, ff, gelu, stream);

    /// <summary>Gated-FFN activation epilogue with F16 I/O (F32 compute). Same launch geometry as the F32 kernel.</summary>
    public unsafe void LaunchGluActF16(ulong comb, ulong gateUp, int rows, int ff, bool gelu, nint stream)
        => LaunchGluActImpl(_ditGluActF16, comb, gateUp, rows, ff, gelu, stream);

    private unsafe void LaunchGluActImpl(nint func, ulong comb, ulong gateUp, int rows, int ff, bool gelu, nint stream)
    {
        ulong combA = comb, guA = gateUp;
        uint rowsA = (uint)rows, ffA = (uint)ff;
        int actA = gelu ? 1 : 0;
        void** args = stackalloc void*[5];
        args[0] = &combA; args[1] = &guA; args[2] = &rowsA; args[3] = &ffA; args[4] = &actA;
        long total = (long)rows * ff;
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>RMSNorm + Q8_1 sidecar emission in one launch (bit-identical F32 output to the plain
    /// RmsNorm kernel — same body and geometry; sidecar math verbatim from quantize_activation_q8_1_f32).</summary>
    public unsafe void LaunchRmsNormQ8(ulong output, ulong xq, ulong xd, ulong xs, ulong input, ulong weight,
        int normDim, int totalRows, float eps, nint stream)
    {
        ulong outA = output, xqA = xq, xdA = xd, xsA = xs, inA = input, wA = weight;
        uint dimA = (uint)normDim, rowsA = (uint)totalRows;
        float epsA = eps;
        void** args = stackalloc void*[9];
        args[0] = &outA; args[1] = &xqA; args[2] = &xdA; args[3] = &xsA; args[4] = &inA; args[5] = &wA;
        args[6] = &dimA; args[7] = &rowsA; args[8] = &epsA;
        CudaDriverApi.cuLaunchKernel(_lmRmsNormQ8_1F32, (uint)totalRows, 1, 1, BlockSize, 1, 1,
            BlockSize * sizeof(float), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Residual-add + RMSNorm + Q8_1 sidecar emission in one launch (bit-identical to
    /// LaunchAddRmsNorm's outputs; sidecar on the normOut).</summary>
    public unsafe void LaunchAddRmsNormQ8(ulong residOut, ulong normOut, ulong xq, ulong xd, ulong xs,
        ulong a, ulong b, ulong weight, int normDim, int totalRows, float eps, nint stream)
    {
        ulong rA = residOut, nA = normOut, xqA = xq, xdA = xd, xsA = xs, aA = a, bA = b, wA = weight;
        uint dimA = (uint)normDim, rowsA = (uint)totalRows;
        float epsA = eps;
        void** args = stackalloc void*[11];
        args[0] = &rA; args[1] = &nA; args[2] = &xqA; args[3] = &xdA; args[4] = &xsA;
        args[5] = &aA; args[6] = &bA; args[7] = &wA;
        args[8] = &dimA; args[9] = &rowsA; args[10] = &epsA;
        CudaDriverApi.cuLaunchKernel(_lmAddRmsNormQ8_1F32, (uint)totalRows, 1, 1, BlockSize, 1, 1,
            BlockSize * sizeof(float), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused GLU epilogue + Q8_1 sidecar emission (bit-identical comb output to LaunchGluAct).
    /// rows*ff must be a multiple of 32 (each warp quantizes its own aligned 32-block).</summary>
    public unsafe void LaunchGluActQ8(ulong comb, ulong xq, ulong xd, ulong xs, ulong gateUp,
        int rows, int ff, bool gelu, nint stream)
    {
        ulong combA = comb, xqA = xq, xdA = xd, xsA = xs, guA = gateUp;
        uint rowsA = (uint)rows, ffA = (uint)ff;
        int actA = gelu ? 1 : 0;
        void** args = stackalloc void*[8];
        args[0] = &combA; args[1] = &xqA; args[2] = &xdA; args[3] = &xsA; args[4] = &guA;
        args[5] = &rowsA; args[6] = &ffA; args[7] = &actA;
        long total = (long)rows * ff;
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmGluActQ8_1F32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Sandwich fusion: residOut = a + rmsnorm(b)*w1; normOut = rmsnorm(residOut)*w2 (+ optional
    /// Q8_1 sidecar when xq != 0). Bit-identical to PostSublayer-RmsNorm followed by AddRmsNorm(+Q8).</summary>
    public unsafe void LaunchNormAddRmsNormQ8(ulong residOut, ulong normOut, ulong xq, ulong xd, ulong xs,
        ulong a, ulong b, ulong w1, ulong w2, int normDim, int totalRows, float eps, nint stream)
    {
        ulong rA = residOut, nA = normOut, xqA = xq, xdA = xd, xsA = xs, aA = a, bA = b, w1A = w1, w2A = w2;
        uint dimA = (uint)normDim, rowsA = (uint)totalRows;
        float epsA = eps;
        void** args = stackalloc void*[12];
        args[0] = &rA; args[1] = &nA; args[2] = &xqA; args[3] = &xdA; args[4] = &xsA;
        args[5] = &aA; args[6] = &bA; args[7] = &w1A; args[8] = &w2A;
        args[9] = &dimA; args[10] = &rowsA; args[11] = &epsA;
        CudaDriverApi.cuLaunchKernel(_lmNormAddRmsNormQ8_1F32, (uint)totalRows, 1, 1, BlockSize, 1, 1,
            BlockSize * sizeof(float), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Sandwich fusion: output = a + rmsnorm(b)*w. Bit-identical to RmsNorm followed by Add.</summary>
    public unsafe void LaunchRmsNormAdd(ulong output, ulong a, ulong b, ulong weight,
        int normDim, int totalRows, float eps, nint stream)
    {
        ulong oA = output, aA = a, bA = b, wA = weight;
        uint dimA = (uint)normDim, rowsA = (uint)totalRows;
        float epsA = eps;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &aA; args[2] = &bA; args[3] = &wA;
        args[4] = &dimA; args[5] = &rowsA; args[6] = &epsA;
        CudaDriverApi.cuLaunchKernel(_lmRmsNormAddF32, (uint)totalRows, 1, 1, BlockSize, 1, 1,
            BlockSize * sizeof(float), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Gemma-4 PLE: dequantizes ONE row of the device-resident Q5_K per-layer token table by the
    /// DEVICE token id and scales by sqrt(ple) — values bit-identical to the CPU codec (same expression order).</summary>
    public unsafe void LaunchPleGatherQ5K(ulong output, ulong table, int width, float scale, ulong deviceTokenId, nint stream)
    {
        ulong oA = output, tA = table, dA = deviceTokenId;
        uint wA = (uint)width;
        float sA = scale;
        void** args = stackalloc void*[5];
        args[0] = &oA; args[1] = &tA; args[2] = &wA; args[3] = &sA; args[4] = &dA;
        uint grid = (uint)((width + 255) / 256);
        CudaDriverApi.cuLaunchKernel(_lmPleGatherQ5KF32, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Qwen3.5 SSM decode: causal depthwise conv step + SiLU + q/k/v split; shifts the
    /// device-persistent history ring in place.</summary>
    public unsafe void LaunchSsmConvSplitStep(ulong q, ulong k, ulong v, ulong qkvMixed, ulong history, ulong convW,
        int convDim, int keyDim, int valueDim, int kernel, nint stream)
    {
        ulong qA = q, kA = k, vA = v, mA = qkvMixed, hA = history, wA = convW;
        uint cdA = (uint)convDim, kdA = (uint)keyDim, vdA = (uint)valueDim, knA = (uint)kernel;
        void** args = stackalloc void*[10];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &mA; args[4] = &hA; args[5] = &wA;
        args[6] = &cdA; args[7] = &kdA; args[8] = &vdA; args[9] = &knA;
        uint grid = (uint)((convDim + 255) / 256);
        CudaDriverApi.cuLaunchKernel(_lmSsmConvSplitStepF32, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-head L2 normalize in place (block per head) with a post-scale.</summary>
    public unsafe void LaunchSsmL2NormHeads(ulong x, int heads, int dim, float eps, float postScale, nint stream)
    {
        ulong xA = x;
        uint dimA = (uint)dim;
        float epsA = eps, scA = postScale;
        void** args = stackalloc void*[4];
        args[0] = &xA; args[1] = &dimA; args[2] = &epsA; args[3] = &scA;
        CudaDriverApi.cuLaunchKernel(_lmSsmL2NormHeadsF32, (uint)heads, 1, 1, 256, 1, 1, 256 * sizeof(float), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Qwen3.5 delta-rule recurrence + gated RMSNorm, one block per value head; state is
    /// device-persistent across tokens.</summary>
    private static readonly bool _ssmDeltaRowParallel = Environment.GetEnvironmentVariable("HARTSY_SSM_DELTA_V2") != "0";
    // Byte-verified 4/4 prompts vs the legacy kernel and 2.7× faster in the isolated-graph
    // microbenchmark (20.5 vs 55 µs/call — the block-per-row shape's 512-byte blocks can't hide
    // DRAM latency; a plain Add over the same tensors streams at 7.4 µs). An earlier e2e A/B
    // wrongly concluded "no gain": it predated the per-head scalar hoist, whose per-thread
    // transcendentals were masking the schedule win. Kill-switch HARTSY_SSM_DELTA_WARPROW=0.
    private static readonly bool _ssmDeltaWarpRow = Environment.GetEnvironmentVariable("HARTSY_SSM_DELTA_WARPROW") != "0";

    public unsafe void LaunchSsmDeltaStep(ulong output, ulong state, ulong q, ulong k, ulong v, ulong z,
        ulong alphaRaw, ulong betaRaw, ulong dtBias, ulong ssmA, ulong normW,
        int hv, int sv, int sk, int repeat, float eps, nint stream, ulong oScratch = 0)
    {
        // Row-parallel split (grid = sv×hv recurrence + head-wide gated-norm tail): the legacy
        // block-per-head kernel serializes sv tree-reductions behind block syncs (192 µs/layer on
        // Qwen3.5-0.8B, ~50% of the whole decode step). Bit-identical values; needs the caller's
        // [hv*sv]-float scratch for the pre-norm readout. sk ≤ 1024 = the rows kernel's per-thread
        // register cache bound (CACHE_COLS·blockDim). Kill-switch HARTSY_SSM_DELTA_V2=0.
        if (_ssmDeltaRowParallel && oScratch != 0 && sk <= 1024 && hv <= 256)
        {
            // Per-head gate scalars first (one tiny launch): the row kernels then load 2 floats
            // per head instead of every thread evaluating 4 precise transcendentals — which, at
            // row-parallel thread counts, dominated the kernel (~42 µs shape-invariant floor).
            // Scalars live in the caller's scratch after the [hv*sv] readout floats.
            ulong scalars = oScratch + (ulong)(hv * sv) * sizeof(float);
            ulong scA = scalars, alA2 = alphaRaw, beA2 = betaRaw, dtA2 = dtBias, saA2 = ssmA;
            uint hvA = (uint)hv;
            void** scalarArgs = stackalloc void*[6];
            scalarArgs[0] = &scA; scalarArgs[1] = &alA2; scalarArgs[2] = &beA2;
            scalarArgs[3] = &dtA2; scalarArgs[4] = &saA2; scalarArgs[5] = &hvA;
            CudaDriverApi.cuLaunchKernel(_lmSsmDeltaScalarsF32, 1, 1, 1, 256, 1, 1, 0, stream, (nint)scalarArgs, 0).ThrowOnError();

            ulong osA = oScratch, stA2 = state, qA2 = q, kA2 = k, vA2 = v, scA2 = scalars;
            uint svA2 = (uint)sv, skA2 = (uint)sk, rpA2 = (uint)repeat;
            void** rowsArgs = stackalloc void*[9];
            rowsArgs[0] = &osA; rowsArgs[1] = &stA2; rowsArgs[2] = &qA2; rowsArgs[3] = &kA2;
            rowsArgs[4] = &vA2; rowsArgs[5] = &scA2;
            rowsArgs[6] = &svA2; rowsArgs[7] = &skA2; rowsArgs[8] = &rpA2;
            uint redBytes = 256 * sizeof(float);
            // sk ≤ 256: warp-per-row shape (8 rows/block, sync-free warp-simulated tree — same
            // values, fewer blocks than the block-per-row shape on Qwen3.5's 128-wide rows).
            if (sk <= 256 && _ssmDeltaWarpRow)
            {
                uint rowBlocks = ((uint)sv + 7u) / 8u;
                CudaDriverApi.cuLaunchKernel(
                    _lmSsmDeltaStepWarpRowF32, rowBlocks, (uint)hv, 1, 256, 1, 1,
                    0, stream, (nint)rowsArgs, 0).ThrowOnError();
            }
            else
            {
                CudaDriverApi.cuLaunchKernel(
                    _lmSsmDeltaStepRowsF32, (uint)sv, (uint)hv, 1, 256, 1, 1,
                    redBytes, stream, (nint)rowsArgs, 0).ThrowOnError();
            }

            ulong outA = output, osA2 = oScratch, zA2 = z, nwA2 = normW;
            uint svA3 = (uint)sv;
            float epsA2 = eps;
            void** normArgs = stackalloc void*[6];
            normArgs[0] = &outA; normArgs[1] = &osA2; normArgs[2] = &zA2; normArgs[3] = &nwA2;
            normArgs[4] = &svA3; normArgs[5] = &epsA2;
            CudaDriverApi.cuLaunchKernel(_lmSsmDeltaGatedNormF32, (uint)hv, 1, 1, 256, 1, 1, redBytes, stream, (nint)normArgs, 0).ThrowOnError();
            return;
        }

        ulong oA = output, stA = state, qA = q, kA = k, vA = v, zA = z, alA = alphaRaw, beA = betaRaw,
            dtA = dtBias, saA = ssmA, nwA = normW;
        uint svA = (uint)sv, skA = (uint)sk, rpA = (uint)repeat;
        float epsA = eps;
        void** args = stackalloc void*[15];
        args[0] = &oA; args[1] = &stA; args[2] = &qA; args[3] = &kA; args[4] = &vA; args[5] = &zA;
        args[6] = &alA; args[7] = &beA; args[8] = &dtA; args[9] = &saA; args[10] = &nwA;
        args[11] = &svA; args[12] = &skA; args[13] = &rpA; args[14] = &epsA;
        uint sharedMem = (uint)((sv + 256) * sizeof(float));
        CudaDriverApi.cuLaunchKernel(_lmSsmDeltaStepF32, (uint)hv, 1, 1, 256, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-row argmax over the last dim: idxPtr[r] = argmax_c inPtr[r,c]; two-stage reduction for large C when scratch given.</summary>
    public unsafe void LaunchArgMaxLastDim(ulong idxPtr, ulong inPtr, int rows, int c, nint stream, ulong scratch = 0)
    {
        uint blockThreads = BlockSize;   // 256, power of two; threads beyond C just hold -FLT_MAX sentinels
        uint sharedBytes = blockThreads * (uint)(sizeof(float) + sizeof(int));
        // Large-C decode argmax (LLM vocab): the single-block kernel reads the whole row through one SM
        // (~180us at C=152k). Two-stage — G chunk blocks then a combine block — is bit-identical (same
        // first-max tie-break at every level) and saturates the GPU. Needs an 8*64-byte scratch for the
        // G partial (value, index) pairs; single-block fallback when the caller has none.
        if (rows == 1 && c >= 32768 && scratch != 0)
        {
            const uint maxG = 64;   // scratch = 64 floats @ 0 + 64 ints @ 256
            uint g = Math.Min(maxG, ((uint)c + 2047) / 2048);
            uint chunk = ((uint)c + g - 1) / g;
            g = ((uint)c + chunk - 1) / chunk;
            ulong pVal = scratch, pIdx = scratch + maxG * sizeof(float);
            ulong inArg1 = inPtr; uint cArg1 = (uint)c, chunkArg = chunk;
            void** args1 = stackalloc void*[5];
            args1[0] = &pVal; args1[1] = &pIdx; args1[2] = &inArg1; args1[3] = &cArg1; args1[4] = &chunkArg;
            CudaDriverApi.cuLaunchKernel(_lmArgMaxStage1F32, g, 1, 1, blockThreads, 1, 1, sharedBytes, stream, (nint)args1, 0).ThrowOnError();
            ulong outArg2 = idxPtr; uint gArg = g;
            void** args2 = stackalloc void*[4];
            args2[0] = &outArg2; args2[1] = &pVal; args2[2] = &pIdx; args2[3] = &gArg;
            CudaDriverApi.cuLaunchKernel(_lmArgMaxStage2F32, 1, 1, 1, blockThreads, 1, 1, sharedBytes, stream, (nint)args2, 0).ThrowOnError();
            return;
        }
        ulong outArg = idxPtr, inArg = inPtr;
        uint cArg = (uint)c;
        void** args = stackalloc void*[3];
        args[0] = &outArg; args[1] = &inArg; args[2] = &cArg;
        CudaDriverApi.cuLaunchKernel(_lmArgMaxLastDimF32, (uint)rows, 1, 1, blockThreads, 1, 1, sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches gated residual over the last dim: out[b,s,d] = residual[b,s,d] + gate[b,d]*value[b,s,d].</summary>
    public void LaunchGatedResidualLastDim(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total, nint stream)
        => LaunchGatedResidualImpl(_ditGatedResidualF32, output, residual, value, gate, seqLen, dim, total, stream);

    /// <summary>Gated residual over the last dim, F16 activation I/O (gate stays F32).</summary>
    public void LaunchGatedResidualLastDimF16(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total, nint stream)
        => LaunchGatedResidualImpl(_ditGatedResidualF16, output, residual, value, gate, seqLen, dim, total, stream);

    /// <summary>Gated residual over the last dim, BF16 activation I/O (gate stays F32).</summary>
    public void LaunchGatedResidualLastDimBf16(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total, nint stream)
        => LaunchGatedResidualImpl(_ditGatedResidualBf16, output, residual, value, gate, seqLen, dim, total, stream);

    /// <summary>Row-indexed gated residual with the gate-table gather folded in.</summary>
    public void LaunchGatedResidualRowIndexed(ulong output, ulong residual, ulong value, ulong gateTable, ulong rowIndex, int dim, long total, nint stream)
        => LaunchGatedResidualRowIndexedImpl(_ditGatedResidualRowIndexedF32, output, residual, value, gateTable, rowIndex, dim, total, stream);

    /// <summary>Row-indexed gated residual, F16 activation I/O (the gate table stays F32).</summary>
    public void LaunchGatedResidualRowIndexedF16(ulong output, ulong residual, ulong value, ulong gateTable, ulong rowIndex, int dim, long total, nint stream)
        => LaunchGatedResidualRowIndexedImpl(_ditGatedResidualRowIndexedF16, output, residual, value, gateTable, rowIndex, dim, total, stream);

    /// <summary>Row-indexed gated residual, BF16 activation I/O (the gate table stays F32).</summary>
    public void LaunchGatedResidualRowIndexedBf16(ulong output, ulong residual, ulong value, ulong gateTable, ulong rowIndex, int dim, long total, nint stream)
        => LaunchGatedResidualRowIndexedImpl(_ditGatedResidualRowIndexedBf16, output, residual, value, gateTable, rowIndex, dim, total, stream);

    /// <summary>Sigmoid with F16 I/O (F32 exp), for the Krea2 attention output gate.</summary>
    public unsafe void LaunchDitSigmoidF16(ulong output, ulong input, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)count;
        void** args = stackalloc void*[3];
        args[0] = &outArg; args[1] = &inArg; args[2] = &countArg;
        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_ditSigmoidF16, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches AdaLN modulation split: proj[B,4D] → (1+scale_msa, tanh(gate_msa), 1+scale_mlp, tanh(gate_mlp)), each [B,D].</summary>
    public void LaunchModulation4(ulong scaleMsa, ulong gateMsa, ulong scaleMlp, ulong gateMlp, ulong proj, int dim, int batch, nint stream)
        => LaunchModulation4Impl(_ditModulation4F32, scaleMsa, gateMsa, scaleMlp, gateMlp, proj, dim, batch, stream);

    /// <summary>Launches CFG combine + Euler step in-place on z: z[i] += delta*(guidance*pos[i] + (1-guidance)*neg[i]).</summary>
    public void LaunchCfgEuler(ulong z, ulong pos, ulong neg, float guidance, float delta, int count, nint stream)
        => LaunchCfgEulerImpl(_ditCfgEulerF32, z, pos, neg, guidance, delta, count, stream);

    /// <summary>Elementwise affine mix: output = xScale*x + yScale*y.</summary>
    public unsafe void LaunchAffineMix(
        ulong output, ulong x, ulong y, float xScale, float yScale, long count, nint stream)
    {
        ulong outputArg = output, xArg = x, yArg = y, countArg = (ulong)count;
        float xScaleArg = xScale, yScaleArg = yScale;
        void** args = stackalloc void*[6];
        args[0] = &outputArg;
        args[1] = &xArg;
        args[2] = &yArg;
        args[3] = &xScaleArg;
        args[4] = &yScaleArg;
        args[5] = &countArg;
        uint gridDim = (uint)Math.Min((countArg + BlockSize - 1) / BlockSize, 65_535UL);
        CudaDriverApi.cuLaunchKernel(
            _ditAffineMixF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>NCHW mask-broadcast affine replacement, in-place on target.</summary>
    public unsafe void LaunchMaskedAffineMixDense(
        ulong target, ulong source, ulong noise, ulong mask, float sourceScale, float noiseScale,
        long batch, long channels, long spatial, nint stream)
    {
        ulong targetArg = target, sourceArg = source, noiseArg = noise, maskArg = mask;
        ulong batchArg = (ulong)batch, channelsArg = (ulong)channels, spatialArg = (ulong)spatial;
        float sourceScaleArg = sourceScale, noiseScaleArg = noiseScale;
        void** args = stackalloc void*[9];
        args[0] = &targetArg;
        args[1] = &sourceArg;
        args[2] = &noiseArg;
        args[3] = &maskArg;
        args[4] = &sourceScaleArg;
        args[5] = &noiseScaleArg;
        args[6] = &batchArg;
        args[7] = &channelsArg;
        args[8] = &spatialArg;
        uint gridX = (uint)Math.Min((spatialArg + BlockSize - 1) / BlockSize, 65_535UL);
        uint gridY = (uint)Math.Min(batchArg, 65_535UL);
        CudaDriverApi.cuLaunchKernel(
            _ditMaskedAffineMixDenseF32, gridX, gridY, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Packed [B,S,F] mask-broadcast affine replacement, in-place on target.</summary>
    public unsafe void LaunchMaskedAffineMixPacked(
        ulong target, ulong source, ulong noise, ulong mask, float sourceScale, float noiseScale,
        long tokens, long featureDimension, long patchArea, bool channelInner, nint stream)
    {
        ulong targetArg = target, sourceArg = source, noiseArg = noise, maskArg = mask;
        ulong tokensArg = (ulong)tokens, featureArg = (ulong)featureDimension, patchArg = (ulong)patchArea;
        float sourceScaleArg = sourceScale, noiseScaleArg = noiseScale;
        void** args = stackalloc void*[9];
        args[0] = &targetArg;
        args[1] = &sourceArg;
        args[2] = &noiseArg;
        args[3] = &maskArg;
        args[4] = &sourceScaleArg;
        args[5] = &noiseScaleArg;
        args[6] = &tokensArg;
        args[7] = &featureArg;
        args[8] = &patchArg;
        uint gridDim = (uint)Math.Min(tokensArg, 65_535UL);
        nint kernel = channelInner ? _ditMaskedAffineMixPackedInnerF32 : _ditMaskedAffineMixPackedOuterF32;
        CudaDriverApi.cuLaunchKernel(
            kernel, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Global-renormalized CFG plus Euler update. Scratch layout is two double partial arrays followed
    /// by one float scale; all three stages are ordered on <paramref name="stream"/> and inputs remain unchanged.</summary>
    public unsafe void LaunchCfgRenormEuler(
        ulong z, ulong cond, ulong uncond, ulong scratch, int partialCount,
        float guidance, float delta, float renormMin, long count, nint stream)
    {
        ulong condPartials = scratch;
        ulong guidedPartials = scratch + (ulong)partialCount * sizeof(double);
        ulong scale = guidedPartials + (ulong)partialCount * sizeof(double);

        ulong condArg = cond, uncondArg = uncond, condPartialsArg = condPartials;
        ulong guidedPartialsArg = guidedPartials, countArg = (ulong)count;
        float guidanceArg = guidance;
        void** reduceArgs = stackalloc void*[6];
        reduceArgs[0] = &condArg;
        reduceArgs[1] = &uncondArg;
        reduceArgs[2] = &condPartialsArg;
        reduceArgs[3] = &guidedPartialsArg;
        reduceArgs[4] = &guidanceArg;
        reduceArgs[5] = &countArg;
        CudaDriverApi.cuLaunchKernel(
            _ditCfgRenormReduceF32, (uint)partialCount, 1, 1, BlockSize, 1, 1,
            2 * BlockSize * sizeof(double), stream, (nint)reduceArgs, 0).ThrowOnError();

        ulong scaleArg = scale;
        uint partialCountArg = (uint)partialCount;
        float renormMinArg = renormMin;
        void** finalizeArgs = stackalloc void*[5];
        finalizeArgs[0] = &scaleArg;
        finalizeArgs[1] = &condPartialsArg;
        finalizeArgs[2] = &guidedPartialsArg;
        finalizeArgs[3] = &partialCountArg;
        finalizeArgs[4] = &renormMinArg;
        CudaDriverApi.cuLaunchKernel(
            _ditCfgRenormFinalizeF32, 1, 1, 1, BlockSize, 1, 1,
            2 * BlockSize * sizeof(double), stream, (nint)finalizeArgs, 0).ThrowOnError();

        ulong zArg = z;
        float deltaArg = delta;
        void** updateArgs = stackalloc void*[7];
        updateArgs[0] = &zArg;
        updateArgs[1] = &condArg;
        updateArgs[2] = &uncondArg;
        updateArgs[3] = &scaleArg;
        updateArgs[4] = &guidanceArg;
        updateArgs[5] = &deltaArg;
        updateArgs[6] = &countArg;
        CudaDriverApi.cuLaunchKernel(
            _ditCfgRenormEulerF32, (uint)partialCount, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)updateArgs, 0).ThrowOnError();
    }

    /// <summary>Last-dimension-normalized CFG plus Euler update. Each block cooperatively reduces one
    /// logical row and applies its broadcast norm ratio to the in-place latent update.</summary>
    public unsafe void LaunchCfgNormalizedEuler(
        ulong z, ulong cond, ulong uncond, float guidance, float delta, float eps,
        long rows, long lastDim, nint stream)
    {
        ulong zArg = z, condArg = cond, uncondArg = uncond;
        float guidanceArg = guidance, deltaArg = delta, epsArg = eps;
        ulong rowsArg = (ulong)rows, lastDimArg = (ulong)lastDim;
        void** args = stackalloc void*[8];
        args[0] = &zArg;
        args[1] = &condArg;
        args[2] = &uncondArg;
        args[3] = &guidanceArg;
        args[4] = &deltaArg;
        args[5] = &epsArg;
        args[6] = &rowsArg;
        args[7] = &lastDimArg;
        uint threads = 32;
        while (threads < (ulong)lastDim && threads < BlockSize) threads <<= 1;
        uint gridDim = (uint)Math.Min(rows, 65_535L);
        CudaDriverApi.cuLaunchKernel(
            _ditCfgNormalizedEulerF32, gridDim, 1, 1, threads, 1, 1,
            2 * threads * sizeof(double), stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches patchify: NCHW <c>[B,C,H,W]</c> → token grid <c>[B,(H/p)·(W/p),C·p²]</c> (F32).
    /// <paramref name="innerChannelFastest"/> selects (ph,pw,c) when true and (c,ph,pw) when false.</summary>
    public unsafe void LaunchDitPatchify(ulong output, ulong input, int batch, int channels, int height, int width,
        int patch, bool innerChannelFastest, nint stream)
        => LaunchDitPatchifyImpl(
            _ditPatchifyF32, output, input, batch, channels, height, width, patch, innerChannelFastest, stream);

    /// <summary>16-bit payload variant shared by F16 and BF16; performs no floating-point conversion.</summary>
    public unsafe void LaunchDitPatchifyU16(ulong output, ulong input, int batch, int channels, int height, int width,
        int patch, bool innerChannelFastest, nint stream)
        => LaunchDitPatchifyImpl(
            _ditPatchifyU16, output, input, batch, channels, height, width, patch, innerChannelFastest, stream);

    private unsafe void LaunchDitPatchifyImpl(nint kernel, ulong output, ulong input, int batch, int channels,
        int height, int width, int patch, bool innerChannelFastest, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint chArg = (uint)channels, hArg = (uint)height, wArg = (uint)width, pArg = (uint)patch;
        uint innerArg = innerChannelFastest ? 1u : 0u;
        ulong totalArg = (ulong)batch * (ulong)channels * (ulong)height * (ulong)width;
        void** args = stackalloc void*[8];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &chArg;
        args[3] = &hArg;
        args[4] = &wArg;
        args[5] = &pArg;
        args[6] = &innerArg;
        args[7] = &totalArg;
        uint gridDim = (uint)Math.Min((totalArg + BlockSize - 1) / BlockSize, 65_535UL);
        CudaDriverApi.cuLaunchKernel(
            kernel, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches unpatchify: token grid [B,hP·wP,C·p²] → pixel latent [B,C,H,W] (F32).
    /// <paramref name="innerChannelFastest"/>: token inner order (ph, pw, c) when true (Z-Image), (c, ph, pw) when false (Krea2).</summary>
    public unsafe void LaunchDitUnpatchify(ulong output, ulong input, int batch, int channels, int hPacked,
        int wPacked, int patch, bool innerChannelFastest, nint stream)
        => LaunchDitUnpatchifyImpl(
            _ditUnpatchifyF32, output, input, batch, channels, hPacked, wPacked, patch, innerChannelFastest, stream);

    /// <summary>16-bit payload variant shared by F16 and BF16; performs no floating-point conversion.</summary>
    public unsafe void LaunchDitUnpatchifyU16(ulong output, ulong input, int batch, int channels, int hPacked,
        int wPacked, int patch, bool innerChannelFastest, nint stream)
        => LaunchDitUnpatchifyImpl(
            _ditUnpatchifyU16, output, input, batch, channels, hPacked, wPacked, patch, innerChannelFastest, stream);

    /// <summary>
    /// Copies one split chunk from logical <c>[outer,inputDimension,inner]</c> storage. The 16-bit variant is
    /// shared by F16 and BF16 and, like the 32-bit variant, treats every element as an opaque payload word.
    /// </summary>
    public unsafe void LaunchSplitSlice(
        bool sixteenBit,
        ulong output,
        ulong input,
        long outer,
        long inputDimension,
        long outputDimension,
        long inner,
        long splitOffset,
        nint stream)
    {
        ulong outputArg = output, inputArg = input;
        ulong inputStrideArg = checked((ulong)inputDimension * (ulong)inner);
        ulong outputStrideArg = checked((ulong)outputDimension * (ulong)inner);
        ulong sourceOffsetArg = checked((ulong)splitOffset * (ulong)inner);
        ulong blocksPerOuterArg = 1UL + (outputStrideArg - 1UL) / BlockSize;
        ulong totalBlocksArg = checked((ulong)outer * blocksPerOuterArg);
        void** args = stackalloc void*[7];
        args[0] = &outputArg;
        args[1] = &inputArg;
        args[2] = &inputStrideArg;
        args[3] = &outputStrideArg;
        args[4] = &sourceOffsetArg;
        args[5] = &blocksPerOuterArg;
        args[6] = &totalBlocksArg;
        uint gridDim = (uint)Math.Min(totalBlocksArg, 65_535UL);
        CudaDriverApi.cuLaunchKernel(
            sixteenBit ? _ditSplitSliceU16 : _ditSplitSliceU32,
            gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchDitUnpatchifyImpl(nint kernel, ulong output, ulong input, int batch, int channels,
        int hPacked, int wPacked, int patch, bool innerChannelFastest, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint chArg = (uint)channels, hpArg = (uint)hPacked, wpArg = (uint)wPacked, pArg = (uint)patch;
        uint innerArg = innerChannelFastest ? 1u : 0u;
        ulong totalArg = (ulong)batch * (ulong)channels * (ulong)hPacked * (ulong)wPacked
            * (ulong)patch * (ulong)patch;
        void** args = stackalloc void*[8];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &chArg;
        args[3] = &hpArg;
        args[4] = &wpArg;
        args[5] = &pArg;
        args[6] = &innerArg;
        args[7] = &totalArg;
        uint gridDim = (uint)Math.Min((totalArg + BlockSize - 1) / BlockSize, 65_535UL);
        CudaDriverApi.cuLaunchKernel(
            kernel, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches tanh: output[i] = tanh(input[i]) (F32).</summary>
    public void LaunchTanh(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_ditTanhF32, output, input, count, stream);

    /// <summary>Launches in-place rotary embedding on x [B,L,numHeads,headDim]; cos/sin [B,L,headDim].</summary>
    public void LaunchRope(ulong x, ulong cos, ulong sin, int numHeads, int headDim, long totalVecs, nint stream, int rotaryDim = 0)
        => LaunchRopeImpl(_ditRopeF32, x, cos, sin, numHeads, headDim, totalVecs, rotaryDim, stream);

    /// <summary>Rotate-half rotary embedding with F16 x (cos/sin stay F32). Same geometry as the F32 kernel.</summary>
    public void LaunchRopeF16(ulong x, ulong cos, ulong sin, int numHeads, int headDim, long totalVecs, nint stream, int rotaryDim = 0)
        => LaunchRopeImpl(_ditRopeF16, x, cos, sin, numHeads, headDim, totalVecs, rotaryDim, stream);

    /// <summary>Launches in-place rotary embedding on head-major x [B,heads,seq,headDim]; cos/sin stay [B,seq,headDim].</summary>
    /// <summary>Whether the division-free head-major rope kernel is available and fits this shape.</summary>
    /// <remarks>Needs the token count in gridDim.x and (batch,head) in gridDim.y, so it declines shapes that
    /// would blow the 65535 gridDim.y ceiling or need a block wider than 1024.</remarks>
    public bool CanUseRopeHeadMajorV2(int heads, int headDim, int seq, int batch, int rotaryDim)
    {
        if (_ditRopeHeadMajorV2F32 == 0) return false;
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim >> 1;
        if (half <= 0) return false;
        int padHalf = 1;
        while (padHalf < half) padHalf <<= 1;
        return padHalf <= 1024 && (long)batch * heads <= 65535 && seq > 0;
    }

    /// <summary>Head-major rope with block-indexed tokens; bit-identical to <see cref="LaunchRopeHeadMajor"/>.</summary>
    public unsafe void LaunchRopeHeadMajorV2(ulong x, ulong cos, ulong sin, int heads, int headDim, int seq,
        int batch, nint stream, int rotaryDim = 0)
    {
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim >> 1;
        // Scalar, one pair per thread. A float4 variant (four pairs, 128-bit loads) measured SLOWER --
        // 169 ms/step against this kernel's 124 -- so the kernel is latency-bound on occupancy, not on
        // access width. Do not re-try it without first raising occupancy.
        int units = half;
        int padUnits = 1, padShift = 0;
        while (padUnits < units) { padUnits <<= 1; padShift++; }
        int vecsPerBlock = Math.Max(1, 256 / padUnits);
        int blockDimX = vecsPerBlock * padUnits;

        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint headDimArg = (uint)headDim, halfArg = (uint)half, seqArg = (uint)seq;
        uint headsArg = (uint)heads, padShiftArg = (uint)padShift;

        void** args = stackalloc void*[8];
        args[0] = &xArg;
        args[1] = &cosArg;
        args[2] = &sinArg;
        args[3] = &headDimArg;
        args[4] = &halfArg;
        args[5] = &seqArg;
        args[6] = &headsArg;
        args[7] = &padShiftArg;

        uint gridX = (uint)((seq + vecsPerBlock - 1) / vecsPerBlock);
        uint gridY = (uint)(batch * heads);
        CudaDriverApi.cuLaunchKernel(
            _ditRopeHeadMajorV2F32, gridX, gridY, 1, (uint)blockDimX, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchRopeHeadMajor(ulong x, ulong cos, ulong sin, int heads, int headDim, int seq,
        long totalVecs, nint stream, int rotaryDim = 0)
    {
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint headsArg = (uint)heads, headDimArg = (uint)headDim, rotArg = (uint)rdim, seqArg = (uint)seq;
        ulong vecsArg = (ulong)totalVecs;

        void** args = stackalloc void*[8];
        args[0] = &xArg;
        args[1] = &cosArg;
        args[2] = &sinArg;
        args[3] = &headsArg;
        args[4] = &headDimArg;
        args[5] = &vecsArg;
        args[6] = &rotArg;
        args[7] = &seqArg;

        long threads = totalVecs * (rdim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            _ditRopeHeadMajorF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches in-place interleaved (GPT-J) rotary embedding on x [B,L,numHeads,headDim];
    /// cos/sin [B,L,headDim]. Rotates adjacent pairs (2i,2i+1) by frequency i (not the split-half convention).
    /// <paramref name="rotaryDim"/> 0 (or &gt;=headDim) rotates every pair (full rotary); otherwise only pairs
    /// inside [0, rotaryDim) rotate — dims [rotaryDim, headDim) pass through (GLM-4's partial rotary).</summary>
    public unsafe void LaunchRopeInterleaved(ulong x, ulong cos, ulong sin, int numHeads, int headDim, int rotaryDim, long totalVecs, nint stream)
    {
        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint headsArg = (uint)numHeads, headDimArg = (uint)headDim, rotaryDimArg = (uint)rotaryDim;
        ulong vecsArg = (ulong)totalVecs;
        void** args = stackalloc void*[7];
        args[0] = &xArg;
        args[1] = &cosArg;
        args[2] = &sinArg;
        args[3] = &headsArg;
        args[4] = &headDimArg;
        args[5] = &rotaryDimArg;
        args[6] = &vecsArg;
        long threads = totalVecs * (headDim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            _lmRopeInterleavedF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Graph-capture decode: split-half RoPE on a single decode-step Q/K tensor [1,numHeads,1,headDim],
    /// reading the rotation row from a precomputed full table via devicePos[1] (qOffset). Grid is fixed
    /// (position-independent) — same devicePos convention as LaunchKvAppend/LaunchFlashAttention.</summary>
    public unsafe void LaunchRopeDecodeSplitHalf(
        ulong x, ulong cosTable, ulong sinTable, int numHeads, int headDim, int rotaryDim,
        ulong devicePos, nint stream)
    {
        ulong xArg = x, cosArg = cosTable, sinArg = sinTable, posArg = devicePos;
        uint headsArg = (uint)numHeads, headDimArg = (uint)headDim, rotArg = (uint)rotaryDim;
        void** args = stackalloc void*[7];
        args[0] = &xArg; args[1] = &cosArg; args[2] = &sinArg;
        args[3] = &headsArg; args[4] = &headDimArg; args[5] = &rotArg; args[6] = &posArg;
        int rdim = rotaryDim == 0 ? headDim : Math.Min(rotaryDim, headDim);
        long threads = (long)numHeads * (rdim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmRopeDecodeSplitHalf, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Graph-capture decode: interleaved (GPT-J) RoPE on a single decode-step Q/K tensor, position
    /// from devicePos[1]. Same table/convention as <see cref="LaunchRopeDecodeSplitHalf"/>.</summary>
    public unsafe void LaunchRopeDecodeInterleaved(ulong x, ulong cosTable, ulong sinTable, int numHeads, int headDim, ulong devicePos, nint stream)
    {
        ulong xArg = x, cosArg = cosTable, sinArg = sinTable, posArg = devicePos;
        uint headsArg = (uint)numHeads, headDimArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &xArg; args[1] = &cosArg; args[2] = &sinArg;
        args[3] = &headsArg; args[4] = &headDimArg; args[5] = &posArg;
        long threads = (long)numHeads * (headDim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmRopeDecodeInterleaved, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Graph-capture decode: gathers one row (the token in <paramref name="tokenId"/>, a persistent
    /// 1-int device buffer) from a GPU-resident embedding table into <paramref name="output"/> [1,1,hidden].
    /// Fixed grid — capturable.</summary>
    public unsafe void LaunchEmbedGatherDecode(ulong output, ulong embedTable, ulong tokenId, int hidden, nint stream)
    {
        ulong outArg = output, embArg = embedTable, tokArg = tokenId;
        uint hiddenArg = (uint)hidden;
        void** args = stackalloc void*[4];
        args[0] = &outArg; args[1] = &embArg; args[2] = &tokArg; args[3] = &hiddenArg;
        uint gridDim = (uint)(((uint)hidden + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmEmbedGatherDecodeF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Graph-capture decode: appends the current device token id into the repetition-penalty
    /// history buffer and increments its counter. Fixed 1×1×1 launch — capturable.</summary>
    public unsafe void LaunchHistoryAppend(ulong history, ulong historyCount, ulong tokenId, nint stream)
    {
        ulong histArg = history, countArg = historyCount, tokArg = tokenId;
        void** args = stackalloc void*[3];
        args[0] = &histArg; args[1] = &countArg; args[2] = &tokArg;
        CudaDriverApi.cuLaunchKernel(_lmHistoryAppend, 1, 1, 1, 1, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Graph-capture decode: applies HF-convention repetition penalty to every token id in
    /// <paramref name="history"/>[0, *<paramref name="historyCount"/>) sequentially (matches
    /// <c>RepetitionPenaltyStep</c>'s CPU semantics exactly). Fixed 1×1×1 launch — capturable.</summary>
    public unsafe void LaunchRepetitionPenalty(ulong logits, ulong history, ulong historyCount, float penalty, int vocabSize, nint stream)
    {
        ulong logitsArg = logits, histArg = history, countArg = historyCount;
        float penaltyArg = penalty;
        uint vocabArg = (uint)vocabSize;
        void** args = stackalloc void*[5];
        args[0] = &logitsArg; args[1] = &histArg; args[2] = &countArg; args[3] = &penaltyArg; args[4] = &vocabArg;
        CudaDriverApi.cuLaunchKernel(_lmRepetitionPenaltyF32, 1, 1, 1, 1, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches last-dim slice: out[row,d] = in[row, offset+d], in row stride = inDim.</summary>
    public void LaunchSliceLastDim(ulong output, ulong input, int outDim, int inDim, int offset, long total, nint stream)
        => LaunchSliceLastDimImpl(_ditSliceLastDimF32, output, input, outDim, inDim, offset, total, stream);

    /// <summary>Last-dim slice with F16 I/O (pure copy). Same geometry as the F32 kernel.</summary>
    public void LaunchSliceLastDimF16(ulong output, ulong input, int outDim, int inDim, int offset, long total, nint stream)
        => LaunchSliceLastDimImpl(_ditSliceLastDimF16, output, input, outDim, inDim, offset, total, stream);

    /// <summary>Launches per-row scalar multiply: out[row,c] = in[row,c] * rowScale[row] (token masking).</summary>
    public void LaunchRowScale(ulong output, ulong input, ulong rowScale, int channels, long total, nint stream)
        => LaunchRowScaleImpl(_ditRowScaleF32, output, input, rowScale, channels, total, stream);

    /// <summary>Launches add-scalar: out[i] = in[i] + c (F32).</summary>
    public void LaunchAddScalar(ulong output, ulong input, float c, int count, nint stream)
        => LaunchScaleImpl(_ditAddScalarF32, output, input, c, count, stream);

    /// <summary>Pixel-quantize to 256 levels (DIAMOND EDM output; enables a drain-free graph-capturable step).</summary>
    public void LaunchPixelQuantize(ulong output, ulong input, int count, nint stream)
        => LaunchScaleImpl(_ditPixelQuantizeF32, output, input, 0f, count, stream);

    /// <summary>Oasis head split: frame-major qkv[token,3·dim] → out[b,heads,seq,headDim] (part q/k/v, spatial/temporal).</summary>
    public unsafe void LaunchOasisSplitHeads(
        ulong output, ulong qkv, int frames, int sp, int heads, int headDim, int part,
        bool temporal, nint stream)
    {
        ulong outArg = output, qkvArg = qkv;
        uint fArg = (uint)frames, spArg = (uint)sp, hArg = (uint)heads, hdArg = (uint)headDim, pArg = (uint)part, tArg = temporal ? 1u : 0u;
        void** args = stackalloc void*[8];
        args[0] = &outArg; args[1] = &qkvArg; args[2] = &fArg; args[3] = &spArg;
        args[4] = &hArg; args[5] = &hdArg; args[6] = &pArg; args[7] = &tArg;
        long threads = (long)frames * sp * heads * headDim;
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_oasisSplitHeadsF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Interleaved (2i,2i+1) partial rope on x[b,heads,seq,headDim]; cos/sin are [seq,rotDim].</summary>
    public unsafe void LaunchOasisRopeInterleaved(ulong x, ulong cos, ulong sin, int batch, int heads, int seq, int headDim, int rotDim, nint stream)
    {
        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint bArg = (uint)batch, hArg = (uint)heads, sArg = (uint)seq, hdArg = (uint)headDim, rArg = (uint)rotDim;
        void** args = stackalloc void*[8];
        args[0] = &xArg; args[1] = &cosArg; args[2] = &sinArg; args[3] = &bArg;
        args[4] = &hArg; args[5] = &sArg; args[6] = &hdArg; args[7] = &rArg;
        long threads = (long)batch * heads * seq * (rotDim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_oasisRopeInterleavedF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Oasis head merge: attn[b,heads,seq,headDim] → out[token,dim] (inverse of split).</summary>
    public unsafe void LaunchOasisMergeHeads(ulong output, ulong attn, int frames, int sp, int heads, int headDim, bool temporal, nint stream)
    {
        ulong outArg = output, attnArg = attn;
        uint fArg = (uint)frames, spArg = (uint)sp, hArg = (uint)heads, hdArg = (uint)headDim, tArg = temporal ? 1u : 0u;
        void** args = stackalloc void*[7];
        args[0] = &outArg; args[1] = &attnArg; args[2] = &fArg; args[3] = &spArg;
        args[4] = &hArg; args[5] = &hdArg; args[6] = &tArg;
        long threads = (long)frames * sp * heads * headDim;
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_oasisMergeHeadsF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused Oasis adaLN: LayerNorm(x)·(1+scale)+shift, scale/shift sliced from mod per frame (5 kernels → 1).</summary>
    public unsafe void LaunchOasisAdaLn(
        ulong output, ulong input, ulong mod, int dim, int sp, int totalRows, int modStride,
        int shiftOff, int scaleOff, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, modArg = mod;
        uint dimArg = (uint)dim, spArg = (uint)sp, rowsArg = (uint)totalRows, msArg = (uint)modStride, shArg = (uint)shiftOff, scArg = (uint)scaleOff;
        float epsArg = eps;
        void** args = stackalloc void*[10];
        args[0] = &outArg; args[1] = &inArg; args[2] = &modArg; args[3] = &dimArg; args[4] = &spArg;
        args[5] = &rowsArg; args[6] = &msArg; args[7] = &shArg; args[8] = &scArg; args[9] = &epsArg;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(_oasisAdaLnF32, (uint)totalRows, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Oasis per-frame unpatchify: proj[t·sp, c·p²] → out[t,c,H,W] with [py,px,ci] inner layout.</summary>
    public unsafe void LaunchOasisUnpatchify(ulong output, ulong proj, int frames, int channels, int gh, int gw, int patch, nint stream)
    {
        ulong outArg = output, projArg = proj;
        uint fArg = (uint)frames, cArg = (uint)channels, ghArg = (uint)gh, gwArg = (uint)gw, pArg = (uint)patch;
        void** args = stackalloc void*[7];
        args[0] = &outArg; args[1] = &projArg; args[2] = &fArg; args[3] = &cArg;
        args[4] = &ghArg; args[5] = &gwArg; args[6] = &pArg;
        long threads = (long)frames * channels * (gh * patch) * (gw * patch);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_oasisUnpatchifyF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>TripoSR triplane grid-sample: writes <paramref name="count"/>×[3·C] features to <paramref name="outF"/>.
    /// <paramref name="coords"/> may be 0 when <paramref name="gridRes"/>&gt;0 (coords generated in-kernel).</summary>
    public unsafe void LaunchTriplaneGridSample(ulong outF, ulong planes, ulong coords, ulong chunkStart,
        int count, int channels, int planeH, int planeW, float radius, int gridRes, nint stream)
    {
        ulong outArg = outF, planesArg = planes, coordsArg = coords, startArg = chunkStart;
        uint cntArg = (uint)count, chArg = (uint)channels, hArg = (uint)planeH, wArg = (uint)planeW, gridArg = (uint)gridRes;
        float radArg = radius;
        void** args = stackalloc void*[10];
        args[0] = &outArg; args[1] = &planesArg; args[2] = &coordsArg; args[3] = &startArg; args[4] = &cntArg;
        args[5] = &chArg; args[6] = &hArg; args[7] = &wArg; args[8] = &radArg; args[9] = &gridArg;
        uint gridDim = (uint)(((long)count + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_triplaneGridSampleF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>2D transposed convolution (gather form). <paramref name="bias"/> may be 0.</summary>
    public unsafe void LaunchConvTranspose2d(ulong outp, ulong input, ulong weight, ulong bias,
        int n, int cIn, int cOut, int iH, int iW, int oH, int oW, int kH, int kW, int sH, int sW, int pH, int pW, nint stream)
    {
        ulong outArg = outp, inArg = input, wArg = weight, bArg = bias;
        int nArg = n, ciArg = cIn, coArg = cOut, ihArg = iH, iwArg = iW, ohArg = oH, owArg = oW;
        int khArg = kH, kwArg = kW, shArg = sH, swArg = sW, phArg = pH, pwArg = pW;
        void** args = stackalloc void*[18];
        args[0] = &outArg; args[1] = &inArg; args[2] = &wArg; args[3] = &bArg; args[4] = &nArg;
        args[5] = &ciArg; args[6] = &coArg; args[7] = &ihArg; args[8] = &iwArg; args[9] = &ohArg; args[10] = &owArg;
        args[11] = &khArg; args[12] = &kwArg; args[13] = &shArg; args[14] = &swArg; args[15] = &phArg; args[16] = &pwArg;
        long threads = (long)n * cOut * oH * oW;
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_convTranspose2dF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>GEGLU with exact erf gate: <paramref name="outp"/>[rows,inner] = proj[:,:inner]·gelu_erf(proj[:,inner:]).</summary>
    public unsafe void LaunchGegluErf(ulong outp, ulong proj, long rows, int inner, nint stream)
    {
        ulong outArg = outp, projArg = proj, rowsArg = (ulong)rows;
        uint innerArg = (uint)inner;
        void** args = stackalloc void*[4];
        args[0] = &outArg; args[1] = &projArg; args[2] = &rowsArg; args[3] = &innerArg;
        long threads = rows * inner;
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_gegluErfF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Exact (erf) GELU, elementwise: <paramref name="outp"/>[i] = 0.5·x·(1+erf(x/√2)).</summary>
    public unsafe void LaunchGeluErf(ulong outp, ulong input, long count, nint stream)
    {
        ulong outArg = outp, inArg = input, countArg = (ulong)count;
        void** args = stackalloc void*[3];
        args[0] = &outArg; args[1] = &inArg; args[2] = &countArg;
        uint gridDim = (uint)((count + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_geluErfF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Two-input concat along the [outer, aDim|bDim, inner] middle axis in ONE launch (replaces the
    /// per-outer-slice memcpy loop). <paramref name="f16"/> selects the F16/F32 activation kernel.</summary>
    public unsafe void LaunchConcat2(bool f16, ulong outp, ulong a, ulong b, int outer, int aDim, int bDim, int inner, nint stream)
    {
        ulong outArg = outp, aArg = a, bArg = b;
        uint outerArg = (uint)outer, aArg2 = (uint)aDim, bArg2 = (uint)bDim, innerArg = (uint)inner;
        void** args = stackalloc void*[7];
        args[0] = &outArg; args[1] = &aArg; args[2] = &bArg;
        args[3] = &outerArg; args[4] = &aArg2; args[5] = &bArg2; args[6] = &innerArg;
        long threads = (long)outer * (aDim + bDim) * inner;
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(f16 ? _ditConcat2F16 : _ditConcat2F32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused adaLN modulation: out = (1+scale)·LayerNormNoAffine(in) + shift (one block per row).</summary>
    public unsafe void LaunchLayerNormModulate(bool f16, ulong output, ulong input, ulong scale, ulong shift,
        int dim, int seqLen, int totalRows, float eps, nint stream)
    {
        ulong outArg = output, inArg = input, scArg = scale, shArg = shift;
        uint dimArg = (uint)dim, seqArg = (uint)seqLen, rowsArg = (uint)totalRows; float epsArg = eps;
        void** args = stackalloc void*[8];
        args[0] = &outArg; args[1] = &inArg; args[2] = &scArg; args[3] = &shArg;
        args[4] = &dimArg; args[5] = &seqArg; args[6] = &rowsArg; args[7] = &epsArg;
        uint sharedMem = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(f16 ? _ditLayerNormModulateF16 : _ditLayerNormModulateF32,
            (uint)totalRows, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    private static uint QkvNormBlockSize(int headDim)
    {
        uint threads = 32;
        while (threads < (uint)headDim && threads < BlockSize)
            threads <<= 1;
        return threads;
    }

    /// <summary>Fused QKV split + per-head QK-RMSNorm with arbitrary positive head dimensions.</summary>
    public unsafe void LaunchQkvSplitNorm(bool f16, ulong q, ulong k, ulong v, ulong qkv, ulong qW, ulong kW,
        int tokens, int heads, int headDim, float eps, nint stream)
    {
        ulong qArg = q, kArg = k, vArg = v, qkvArg = qkv, qwArg = qW, kwArg = kW;
        uint tArg = (uint)tokens, hArg = (uint)heads, hdArg = (uint)headDim; float epsArg = eps;
        void** args = stackalloc void*[10];
        args[0] = &qArg; args[1] = &kArg; args[2] = &vArg; args[3] = &qkvArg; args[4] = &qwArg; args[5] = &kwArg;
        args[6] = &tArg; args[7] = &hArg; args[8] = &hdArg; args[9] = &epsArg;
        uint grid = (uint)((long)tokens * heads);
        uint block = QkvNormBlockSize(headDim);
        uint sharedMem = 2u * block * sizeof(float);
        CudaDriverApi.cuLaunchKernel(f16 ? _ditQkvSplitNormF16 : _ditQkvSplitNormF32,
            grid, 1, 1, block, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Head-major twin of <see cref="LaunchQkvSplitNorm"/>: q/k/v come out as [batch, heads, seq, headDim].</summary>
    /// <remarks><paramref name="packStride"/> is how many <c>heads*headDim</c>-wide segments the packed source holds
    /// and the slot arguments say which segment each output reads; a negative slot skips that output (its pointer may
    /// be 0). The full fused call is <c>packStride: 3, qSlot: 0, kSlot: 1, vSlot: 2</c>.</remarks>
    public unsafe void LaunchQkvSplitNormHeadMajor(bool f16, ulong q, ulong k, ulong v, ulong qkv, ulong qW, ulong kW,
        int tokens, int heads, int headDim, int seq, float eps, nint stream,
        int packStride = 3, int qSlot = 0, int kSlot = 1, int vSlot = 2)
    {
        ulong qArg = q, kArg = k, vArg = v, qkvArg = qkv, qwArg = qW, kwArg = kW;
        uint tArg = (uint)tokens, hArg = (uint)heads, hdArg = (uint)headDim, sArg = (uint)seq; float epsArg = eps;
        uint psArg = (uint)packStride; int qsArg = qSlot, ksArg = kSlot, vsArg = vSlot;
        // The fused call goes to the ORIGINAL kernel with its original argument list. Routing it through the
        // subset twin instead moved real generation output: the slot guards change that kernel's codegen, and
        // the sub-chunkRows dispatch exists precisely to keep this path bit-identical.
        bool full = packStride == 3 && qSlot == 0 && kSlot == 1 && vSlot == 2;
        void** args = stackalloc void*[15];
        args[0] = &qArg; args[1] = &kArg; args[2] = &vArg; args[3] = &qkvArg; args[4] = &qwArg; args[5] = &kwArg;
        args[6] = &tArg; args[7] = &hArg; args[8] = &hdArg; args[9] = &sArg; args[10] = &epsArg;
        if (!full) { args[11] = &psArg; args[12] = &qsArg; args[13] = &ksArg; args[14] = &vsArg; }
        uint grid = (uint)((long)tokens * heads);
        uint block = QkvNormBlockSize(headDim);
        uint sharedMem = 2u * block * sizeof(float);
        nint fn = full
            ? (f16 ? _ditQkvSplitNormHeadMajorF16 : _ditQkvSplitNormHeadMajorF32)
            : (f16 ? _ditQkvSplitNormHeadMajorSubsetF16 : _ditQkvSplitNormHeadMajorSubsetF32);
        CudaDriverApi.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>FourierEmbedder over device coords [count,3] → dst [count, 3·(2·bands+1)] (one thread per point·coord).</summary>
    public unsafe void LaunchFourierEmbed(ulong dst, ulong coords, int count, int bands, int dim, nint stream)
    {
        ulong dArg = dst, cArg = coords; uint countArg = (uint)count, bArg = (uint)bands, dimArg = (uint)dim;
        void** args = stackalloc void*[5];
        args[0] = &dArg; args[1] = &cArg; args[2] = &countArg; args[3] = &bArg; args[4] = &dimArg;
        uint grid = (uint)(((long)count * 3 + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_fourierEmbedF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Sparse submanifold-conv scatter/gather (one thread per voxel·channel). <paramref name="scatter"/>
    /// selects scatter (feats→grid) vs gather (grid→feats).</summary>
    public unsafe void LaunchSparseGridScatterGather(bool scatter, ulong grid, ulong feats, ulong coords, int n, int c, int r, nint stream)
    {
        ulong gArg = grid, fArg = feats, cArg = coords; int nA = n, cA = c, rA = r;
        void** args = stackalloc void*[6];
        // kernel signature: (grid/feats first arg, feats/grid second, coords, n, c, r)
        args[0] = scatter ? &gArg : &fArg; args[1] = scatter ? &fArg : &gArg; args[2] = &cArg;
        args[3] = &nA; args[4] = &cA; args[5] = &rA;
        uint grid1 = (uint)(((long)n * c + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(scatter ? _sparseScatterToGridF32 : _sparseGatherFromGridF32,
            grid1, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Row gather (<paramref name="gather"/>=true: out[j]=in[idx[j]]) or scatter-add (out[idx[j]]+=in[j]).</summary>
    public unsafe void LaunchRowGatherScatter(bool gather, ulong outp, ulong input, ulong indices, int m, int c, nint stream)
    {
        ulong oArg = outp, iArg = input, idxArg = indices; int mA = m, cA = c;
        void** args = stackalloc void*[5];
        args[0] = &oArg; args[1] = &iArg; args[2] = &idxArg; args[3] = &mA; args[4] = &cA;
        uint grid = (uint)(((long)m * c + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(gather ? _rowGatherF32 : _rowScatterAddF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Dense 3D convolution (gather form, one thread per output element). Weight [Cout,Cin,kD,kH,kW].</summary>
    public unsafe void LaunchConv3d(ulong outp, ulong input, ulong weight, ulong bias,
        int n, int cin, int cout, int iD, int iH, int iW, int oD, int oH, int oW,
        int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, nint stream)
    {
        ulong oArg = outp, iArg = input, wArg = weight, bArg = bias;
        int nA = n, cinA = cin, coutA = cout, iDA = iD, iHA = iH, iWA = iW, oDA = oD, oHA = oH, oWA = oW,
            kDA = kD, kHA = kH, kWA = kW, sDA = sD, sHA = sH, sWA = sW, pDA = pD, pHA = pH, pWA = pW;
        void** args = stackalloc void*[22];
        args[0] = &oArg; args[1] = &iArg; args[2] = &wArg; args[3] = &bArg;
        args[4] = &nA; args[5] = &cinA; args[6] = &coutA; args[7] = &iDA; args[8] = &iHA; args[9] = &iWA;
        args[10] = &oDA; args[11] = &oHA; args[12] = &oWA; args[13] = &kDA; args[14] = &kHA; args[15] = &kWA;
        args[16] = &sDA; args[17] = &sHA; args[18] = &sWA; args[19] = &pDA; args[20] = &pHA; args[21] = &pWA;
        long total = (long)n * cout * oD * oH * oW;
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_conv3dF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Add scalar (out = in + c) with F16 I/O (scalar stays F32).</summary>
    public void LaunchAddScalarF16(ulong output, ulong input, float c, int count, nint stream)
        => LaunchScaleImpl(_ditAddScalarF16, output, input, c, count, stream);

    /// <summary>Launches non-affine LayerNorm: per-row zero-mean unit-variance, no scale/bias.</summary>
    public void LaunchLayerNormNoAffine(ulong output, ulong input, int dim, int totalRows, float eps, nint stream)
        => LaunchLayerNormNoAffineImpl(_ditLayerNormNoAffineF32, output, input, dim, totalRows, eps, stream);

    /// <summary>Non-affine LayerNorm with F16 activation I/O (F32 accumulate) — the DiT F16 activation path.</summary>
    public void LaunchLayerNormNoAffineF16(ulong output, ulong input, int dim, int totalRows, float eps, nint stream)
        => LaunchLayerNormNoAffineImpl(_ditLayerNormNoAffineF16, output, input, dim, totalRows, eps, stream);

    /// <summary>MoE top-k gate weights: softmax over E + top-k renorm per token (one thread per token).</summary>
    public unsafe void LaunchMoeTopkGate(ulong weights, ulong logits, int numExperts, int topK, long totalTokens, nint stream)
    {
        ulong wArg = weights, lArg = logits;
        uint eArg = (uint)numExperts, kArg = (uint)topK;
        ulong totalArg = (ulong)totalTokens;
        void** args = stackalloc void*[5];
        args[0] = &wArg;
        args[1] = &lArg;
        args[2] = &eArg;
        args[3] = &kArg;
        args[4] = &totalArg;
        uint gridDim = (uint)((totalTokens + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ditMoeTopkGateF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-token gated accumulate: out[i] += gate[row(i), expertIdx] * val[i] (MoE combine).</summary>
    public unsafe void LaunchRowGatedAccum(ulong inout, ulong value, ulong gate, int numExperts, int expertIdx, int dim, long total, nint stream)
    {
        ulong oArg = inout, vArg = value, gArg = gate;
        uint eArg = (uint)numExperts, xArg = (uint)expertIdx, dArg = (uint)dim;
        ulong totalArg = (ulong)total;
        void** args = stackalloc void*[7];
        args[0] = &oArg;
        args[1] = &vArg;
        args[2] = &gArg;
        args[3] = &eArg;
        args[4] = &xArg;
        args[5] = &dArg;
        args[6] = &totalArg;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ditRowGatedAccumF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches index-add of embedding rows in-place: h[row,d] += table[indices[row], d].</summary>
    public void LaunchIndexAdd(ulong h, ulong table, ulong indices, int dim, long total, nint stream)
        => LaunchIndexAddImpl(_ditIndexAddF32, h, table, indices, dim, total, stream);

    /// <summary>Launches scatter-rows-after: output = [zeros(headRows), input] along the row axis.</summary>
    public void LaunchScatterRowsAfter(ulong output, ulong input, int headRows, int dim, long total, nint stream)
        => LaunchScatterRowsAfterImpl(_ditScatterRowsAfterF32, output, input, headRows, dim, total, stream);

    /// <summary>Launches contiguous row-block slice: output[i] = input[elemOffset + i].</summary>
    public void LaunchSliceRows(ulong output, ulong input, long elemOffset, long total, nint stream)
        => LaunchSliceRowsImpl(_ditSliceRowsF32, output, input, elemOffset, total, stream);

    /// <summary>Contiguous row-block slice with F16 I/O (pure copy). Same geometry as the F32 kernel.</summary>
    public void LaunchSliceRowsF16(ulong output, ulong input, long elemOffset, long total, nint stream)
        => LaunchSliceRowsImpl(_ditSliceRowsF16, output, input, elemOffset, total, stream);

    /// <summary>CHW F32 [-1,1] → HWC u8 [0,255] image conversion. One thread per pixel.</summary>
    public unsafe void LaunchChwToHwcU8(ulong output, ulong input, int height, int width, nint stream)
    {
        ulong oA = output, iA = input;
        uint hA = (uint)height, wA = (uint)width;
        void** args = stackalloc void*[4];
        args[0] = &oA; args[1] = &iA; args[2] = &hA; args[3] = &wA;
        uint gridDim = (uint)(((long)height * width + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ditChwToHwcU8, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Fused GroupNorm+SiLU Launches ───────────────────────────────────

    /// <summary>Launches fused GroupNorm+SiLU: normalize, affine, then SiLU in one kernel (F32).</summary>
    public void LaunchGroupNormSilu(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormSiluF32, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches fused GroupNorm+SiLU with F16 I/O, FP32 accumulation.</summary>
    public void LaunchGroupNormSiluF16(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormSiluF16, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    /// <summary>Launches fused GroupNorm+SiLU with BF16 I/O, FP32 accumulation.</summary>
    public void LaunchGroupNormSiluBf16(ulong output, ulong input, ulong weight, ulong bias,
        int batch, int channels, int spatial, int groups, float eps, nint stream)
        => LaunchGroupNormImpl(_groupnormSiluBf16, output, input, weight, bias, batch, channels, spatial, groups, eps, stream);

    // ── Spatial Launches ────────────────────────────────────────────────

    /// <summary>Launches UpsampleNearest2D (F32).</summary>
    public void LaunchUpsampleNearest2D(ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
        => LaunchUpsampleImpl(_upsampleNearest2dF32, output, input, batch, channels, inH, inW, outH, outW, scaleH, scaleW, stream);

    /// <summary>Launches UpsampleNearest2D (F16).</summary>
    public void LaunchUpsampleNearest2DF16(ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
        => LaunchUpsampleImpl(_upsampleNearest2dF16, output, input, batch, channels, inH, inW, outH, outW, scaleH, scaleW, stream);

    /// <summary>Launches UpsampleNearest2D (BF16).</summary>
    public void LaunchUpsampleNearest2DBf16(ulong output, ulong input,
        int batch, int channels, int inH, int inW, int outH, int outW, int scaleH, int scaleW, nint stream)
        => LaunchUpsampleImpl(_upsampleNearest2dBf16, output, input, batch, channels, inH, inW, outH, outW, scaleH, scaleW, stream);

    /// <summary>Launches Im2Col for one batch element (F32).</summary>
    public void LaunchIm2Col(ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
        => LaunchIm2ColImpl(_im2colF32, col, input, channels, inH, inW, kH, kW, padH, padW, strideH, strideW, outH, outW, batchOffset, stream);

    /// <summary>Launches Im2Col for one batch element (F16).</summary>
    public void LaunchIm2ColF16(ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
        => LaunchIm2ColImpl(_im2colF16, col, input, channels, inH, inW, kH, kW, padH, padW, strideH, strideW, outH, outW, batchOffset, stream);

    /// <summary>Launches Im2Col for one batch element (BF16).</summary>
    public void LaunchIm2ColBf16(ulong col, ulong input,
        int channels, int inH, int inW, int kH, int kW,
        int padH, int padW, int strideH, int strideW,
        int outH, int outW, int batchOffset, nint stream)
        => LaunchIm2ColImpl(_im2colBf16, col, input, channels, inH, inW, kH, kW, padH, padW, strideH, strideW, outH, outW, batchOffset, stream);

    /// <summary>Launches bias addition: output[i] += bias[channel_of(i)] (F32)</summary>
    public void LaunchBiasAdd(ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
        => LaunchBiasAddImpl(_col2biasAddF32, output, bias, outChannels, spatial, totalElements, stream);

    /// <summary>Launches bias addition: output[i] += bias[channel_of(i)] (F16)</summary>
    public void LaunchBiasAddF16(ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
        => LaunchBiasAddImpl(_col2biasAddF16, output, bias, outChannels, spatial, totalElements, stream);

    /// <summary>Launches bias addition: output[i] += bias[channel_of(i)] (BF16)</summary>
    public void LaunchBiasAddBf16(ulong output, ulong bias, int outChannels, int spatial, int totalElements, nint stream)
        => LaunchBiasAddImpl(_col2biasAddBf16, output, bias, outChannels, spatial, totalElements, stream);

    // ── Softmax Launches ────────────────────────────────────────────────

    /// <summary>Launches in-place per-row softmax (F32). One block per row.</summary>
    public void LaunchSoftmax(ulong data, int rowLen, int totalRows, nint stream)
        => LaunchSoftmaxImpl(_softmaxF32, data, rowLen, totalRows, stream);

    /// <summary>Launches in-place per-row softmax with F16 I/O, FP32 accumulation.</summary>
    public void LaunchSoftmaxF16(ulong data, int rowLen, int totalRows, nint stream)
        => LaunchSoftmaxImpl(_softmaxF16, data, rowLen, totalRows, stream);

    // ── Transpose/Permute Launches ──────────────────────────────────────

    /// <summary>Launches batched 2D transpose: [B, D1, D2] -> [B, D2, D1] (F32).</summary>
    public void LaunchTranspose2D(ulong output, ulong input, int d1, int d2, int totalElements, nint stream)
        => LaunchTranspose2DImpl(_transpose2dF32, output, input, d1, d2, totalElements, stream);

    /// <summary>Launches batched 2D transpose: [B, D1, D2] -> [B, D2, D1] (F16).</summary>
    public void LaunchTranspose2DF16(ulong output, ulong input, int d1, int d2, int totalElements, nint stream)
        => LaunchTranspose2DImpl(_transpose2dF16, output, input, d1, d2, totalElements, stream);

    /// <summary>Launches 4D permute(0,2,1,3): [B, S, H, D] -> [B, H, S, D] (F32).</summary>
    public void LaunchPermute0213(ulong output, ulong input, int s, int h, int d, int totalElements, nint stream)
        => LaunchPermute0213Impl(_permute0213F32, output, input, s, h, d, totalElements, stream);

    /// <summary>Launches 4D permute(0,2,1,3): [B, S, H, D] -> [B, H, S, D] (F16).</summary>
    public void LaunchPermute0213F16(ulong output, ulong input, int s, int h, int d, int totalElements, nint stream)
        => LaunchPermute0213Impl(_permute0213F16, output, input, s, h, d, totalElements, stream);

    /// <summary>Wan-Video interleaved RoPE, in-place on <paramref name="x"/> [S, heads·headDim]. One thread per
    /// (s, head, pair); cos/sin are [S, headDim] shared across heads (duplicated-pair layout).</summary>
    public unsafe void LaunchWanRopeInterleaved(ulong x, ulong cos, ulong sin, int S, int heads, int headDim, nint stream)
    {
        ulong xA = x, cA = cos, sA = sin;
        uint sArg = (uint)S, hArg = (uint)heads, dArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &xA; args[1] = &cA; args[2] = &sA; args[3] = &sArg; args[4] = &hArg; args[5] = &dArg;
        long total = (long)S * heads * (headDim / 2);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanRopeInterleaved, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-head interleaved RoPE (MG3 sigma_theta): cos/sin are [heads, S, headDim]. F32.</summary>
    public unsafe void LaunchWanRopeInterleavedPerHead(ulong x, ulong cos, ulong sin, int S, int heads, int headDim, nint stream)
    {
        ulong xA = x, cA = cos, sA = sin;
        uint sArg = (uint)S, hArg = (uint)heads, dArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &xA; args[1] = &cA; args[2] = &sA; args[3] = &sArg; args[4] = &hArg; args[5] = &dArg;
        long total = (long)S * heads * (headDim / 2);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanRopeInterleavedPerHead, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule: gather QKV slot into the batched temporal layout [sp, heads, tt, headDim].</summary>
    public unsafe void LaunchMg3SplitQkvTemporal(ulong qkv, ulong outp, int tt, int sp, int heads, int headDim, int part, int stride, nint stream)
    {
        ulong qkvArg = qkv, outpArg = outp;
        uint ttArg = (uint)tt, spArg = (uint)sp, headsArg = (uint)heads, headDimArg = (uint)headDim, partArg = (uint)part, strideArg = (uint)stride;
        void** args = stackalloc void*[8];
        args[0] = &qkvArg; args[1] = &outpArg; args[2] = &ttArg; args[3] = &spArg;
        args[4] = &headsArg; args[5] = &headDimArg; args[6] = &partArg; args[7] = &strideArg;
        long total = (long)sp * heads * tt * headDim; uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3SplitQkvTemporalF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule: [sp, heads, tt, headDim] back to token rows [tt·sp, streamDim].</summary>
    public unsafe void LaunchMg3MergeTemporal(ulong attn, ulong outp, int tt, int sp, int heads, int headDim, nint stream)
    {
        ulong attnArg = attn, outpArg = outp;
        uint ttArg = (uint)tt, spArg = (uint)sp, headsArg = (uint)heads, headDimArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &attnArg; args[1] = &outpArg; args[2] = &ttArg; args[3] = &spArg; args[4] = &headsArg; args[5] = &headDimArg;
        long total = (long)tt * sp * heads * headDim; uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3MergeTemporalF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule: interleaved rope on [sp, heads, tt, headDim]; cos/sin [gridRows, headDim].</summary>
    public unsafe void LaunchMg3RopeBatched(
        ulong x, ulong cos, ulong sin, int sp, int heads, int tt, int headDim, int gh, int gw,
        int broadcast, nint stream)
    {
        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint spArg = (uint)sp, headsArg = (uint)heads, ttArg = (uint)tt, headDimArg = (uint)headDim;
        uint ghArg = (uint)gh, gwArg = (uint)gw, broadcastArg = (uint)broadcast;
        void** args = stackalloc void*[10];
        args[0] = &xArg; args[1] = &cosArg; args[2] = &sinArg; args[3] = &spArg; args[4] = &headsArg;
        args[5] = &ttArg; args[6] = &headDimArg; args[7] = &ghArg; args[8] = &gwArg; args[9] = &broadcastArg;
        long total = (long)sp * heads * tt * (headDim / 2);
        uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3RopeBatchedF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule keyboard: expand K/V [tt, 2·streamDim] → k,v [sp, heads, tt, headDim].</summary>
    public unsafe void LaunchMg3KvExpand(ulong kv, ulong kOut, ulong vOut, int sp, int heads, int tt, int headDim, nint stream)
    {
        ulong kvArg = kv, kOutArg = kOut, vOutArg = vOut;
        uint spArg = (uint)sp, headsArg = (uint)heads, ttArg = (uint)tt, headDimArg = (uint)headDim;
        void** args = stackalloc void*[7];
        args[0] = &kvArg; args[1] = &kOutArg; args[2] = &vOutArg;
        args[3] = &spArg; args[4] = &headsArg; args[5] = &ttArg; args[6] = &headDimArg;
        long total = (long)sp * heads * tt * headDim; uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3KvExpandF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 mouse stream: build MLP input [tt·sp, imgDim+winFloats] = [hidden | broadcast mouse window].</summary>
    public unsafe void LaunchMg3MouseMlpConcat(ulong hidden, ulong mouseWin, ulong outp, int tt, int sp, int imgDim, int winFloats, nint stream)
    {
        ulong hiddenArg = hidden, mouseWinArg = mouseWin, outpArg = outp;
        uint ttArg = (uint)tt, spArg = (uint)sp, imgDimArg = (uint)imgDim, winFloatsArg = (uint)winFloats;
        void** args = stackalloc void*[7];
        args[0] = &hiddenArg; args[1] = &mouseWinArg; args[2] = &outpArg;
        args[3] = &ttArg; args[4] = &spArg; args[5] = &imgDimArg; args[6] = &winFloatsArg;
        long total = (long)tt * sp * (imgDim + winFloats); uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3MouseMlpConcatF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Interleaved RoPE with F16 activation I/O (cos/sin stay F32). Same geometry as the F32 kernel.</summary>
    public unsafe void LaunchWanRopeInterleavedF16(ulong x, ulong cos, ulong sin, int S, int heads, int headDim, nint stream)
    {
        ulong xA = x, cA = cos, sA = sin;
        uint sArg = (uint)S, hArg = (uint)heads, dArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &xA; args[1] = &cA; args[2] = &sA; args[3] = &sArg; args[4] = &hArg; args[5] = &dArg;
        long total = (long)S * heads * (headDim / 2);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ditRopeInterleavedF16, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>LTX-2 split (rotate-half, per-head cos) rotary. x [S, numHeads*headDim] in-place; cos/sin [S, dim/2].</summary>
    public unsafe void LaunchLtx2SplitRope(ulong x, ulong cos, ulong sin, int S, int numHeads, int headDim, nint stream, bool f16 = false)
    {
        ulong xA = x, cA = cos, sA = sin;
        uint sArg = (uint)S, hArg = (uint)numHeads, dArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &xA; args[1] = &cA; args[2] = &sA; args[3] = &sArg; args[4] = &hArg; args[5] = &dArg;
        long total = (long)S * numHeads * (headDim / 2);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(f16 ? _ltx2SplitRopeF16 : _ltx2SplitRopeF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }


    /// <summary>Fused LTX-2 QK path: full-row RMS norm -> optional split RoPE -> head-major emit, one pass.
    /// <paramref name="normW"/>/<paramref name="cos"/>/<paramref name="sin"/> may be 0. Shared memory is
    /// <c>(BlockSize + heads*headDim)</c> floats, so the caller must keep that under the 48 KB static limit.
    /// <paramref name="ropeF16"/> selects the F16-cos/sin variant (F32 activation has none — see the dispatch).</summary>
    public unsafe void LaunchLtx2QkNormRopeHeadMajor(ulong outp, ulong x, ulong normW, ulong cos, ulong sin,
        int seq, int heads, int headDim, float eps, nint stream, bool f16, bool ropeF16 = false)
    {
        ulong oA = outp, xA = x, nA = normW, cA = cos, sA = sin;
        uint sqA = (uint)seq, hA = (uint)heads, dA = (uint)headDim; float eA = eps;
        void** args = stackalloc void*[9];
        args[0] = &oA; args[1] = &xA; args[2] = &nA; args[3] = &cA; args[4] = &sA;
        args[5] = &sqA; args[6] = &hA; args[7] = &dA; args[8] = &eA;
        uint shared = (uint)((BlockSize + heads * headDim) * sizeof(float));
        nint fn = f16 ? (ropeF16 ? _ltx2QkNormRopeHeadMajorRopeF16 : _ltx2QkNormRopeHeadMajorF16) : _ltx2QkNormRopeHeadMajorF32;
        CudaDriverApi.cuLaunchKernel(fn, (uint)seq, 1, 1, BlockSize, 1, 1, shared, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Token-major twin of <see cref="LaunchLtx2QkNormRopeHeadMajor"/>: same math, output stays
    /// <c>[seq, heads*headDim]</c> so the row is written where it was read (no scattered head-plane store).</summary>
    public unsafe void LaunchLtx2QkNormRopeTokenMajor(ulong outp, ulong x, ulong normW, ulong cos, ulong sin,
        int seq, int heads, int headDim, float eps, nint stream, bool f16, bool ropeF16 = false)
    {
        ulong oA = outp, xA = x, nA = normW, cA = cos, sA = sin;
        uint sqA = (uint)seq, hA = (uint)heads, dA = (uint)headDim; float eA = eps;
        void** args = stackalloc void*[9];
        args[0] = &oA; args[1] = &xA; args[2] = &nA; args[3] = &cA; args[4] = &sA;
        args[5] = &sqA; args[6] = &hA; args[7] = &dA; args[8] = &eA;
        uint shared = (uint)((BlockSize + heads * headDim) * sizeof(float));
        nint fn = f16 ? (ropeF16 ? _ltx2QkNormRopeTokenMajorRopeF16 : _ltx2QkNormRopeTokenMajorF16) : _ltx2QkNormRopeTokenMajorF32;
        CudaDriverApi.cuLaunchKernel(fn, (uint)seq, 1, 1, BlockSize, 1, 1, shared, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused LTX-2 RMS-norm + AdaLN shift/scale: <c>out = norm(in)*(1+scale)+shift</c>, one pass.</summary>
    public unsafe void LaunchLtx2RmsModulate(ulong outp, ulong x, ulong scale, ulong shift,
        int dim, int rows, float eps, nint stream, bool f16)
    {
        ulong oA = outp, xA = x, scA = scale, shA = shift;
        uint dA = (uint)dim, rA = (uint)rows; float eA = eps;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &xA; args[2] = &scA; args[3] = &shA; args[4] = &dA; args[5] = &rA; args[6] = &eA;
        uint shared = BlockSize * sizeof(float);
        CudaDriverApi.cuLaunchKernel(f16 ? _ltx2RmsModulateF16 : _ltx2RmsModulateF32, (uint)rows, 1, 1, BlockSize, 1, 1, shared, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused LTX-2 per-head output gate, in place: <c>x[s, h*headDim+d] *= 2*sigmoid(logits[s,h])</c>.</summary>
    public unsafe void LaunchLtx2HeadGate(ulong x, ulong logits, int seq, int heads, int headDim, nint stream, bool f16)
    {
        ulong xA = x, lA = logits;
        uint sA = (uint)seq, hA = (uint)heads, dA = (uint)headDim;
        void** args = stackalloc void*[5];
        args[0] = &xA; args[1] = &lA; args[2] = &sA; args[3] = &hA; args[4] = &dA;
        long total = (long)seq * heads * headDim;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(f16 ? _ltx2HeadGateF16 : _ltx2HeadGateF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Extracts temporal frame <paramref name="ti"/> of a 5D <c>[B,C,Tsrc,H,W]</c> tensor into a 4D
    /// <c>[B,C,H,W]</c> frame, on-device.</summary>
    public unsafe void LaunchWanVaeExtractFrame(ulong outp, ulong src, int ti, int B, int C, int Tsrc, int frameHW, nint stream, bool bf16 = false)
    {
        ulong oA = outp, sA = src; uint tiA = (uint)ti, bA = (uint)B, cA = (uint)C, tsA = (uint)Tsrc, fA = (uint)frameHW;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &sA; args[2] = &tiA; args[3] = &bA; args[4] = &cA; args[5] = &tsA; args[6] = &fA;
        long total = (long)B * C * frameHW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeExtractFrameBf16 : _wanVaeExtractFrame, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Writes a 4D <c>[B,C,H,W]</c> frame (plus optional per-channel bias) into temporal slot
    /// <paramref name="to"/> of a 5D <c>[B,C,Tout,H,W]</c> tensor, on-device. <paramref name="bias"/>=0 for none.</summary>
    public unsafe void LaunchWanVaeWriteFrame(ulong outp, ulong acc, ulong bias, int to, int B, int C, int Tout, int frameHW, nint stream, bool bf16 = false)
    {
        ulong oA = outp, aA = acc, bsA = bias; uint toA = (uint)to, bA = (uint)B, cA = (uint)C, tA = (uint)Tout, fA = (uint)frameHW;
        void** args = stackalloc void*[8];
        args[0] = &oA; args[1] = &aA; args[2] = &bsA; args[3] = &toA; args[4] = &bA; args[5] = &cA; args[6] = &tA; args[7] = &fA;
        long total = (long)B * C * frameHW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeWriteFrameBf16 : _wanVaeWriteFrame, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Builds the frame-major padded input [paddedT, cIn, H+2·padH, W+2·padW] (transpose + temporal
    /// zero/replicate-first pad + cache + spatial edge-replicate pad) for batched CausalConv3d.
    /// <paramref name="cache"/>=0 for none.</summary>
    public unsafe void LaunchWanVaeBuildPadded(ulong padded, ulong input, ulong cache, int paddedT, int cIn, int Tin, int cacheLen, int zeroPad,
        int H, int W, int padH, int padW, bool replicateFirst, nint stream, bool bf16 = false, bool reflectSpatial = false)
    {
        ulong pA = padded, iA = input, cA = cache;
        uint ptA = (uint)paddedT, ciA = (uint)cIn, tiA = (uint)Tin, clA = (uint)cacheLen, zpA = (uint)zeroPad;
        uint hA = (uint)H, wA = (uint)W, phA = (uint)padH, pwA = (uint)padW, rfA = replicateFirst ? 1u : 0u;
        uint rsA = reflectSpatial ? 1u : 0u;
        void** args = stackalloc void*[14];
        args[0] = &pA; args[1] = &iA; args[2] = &cA; args[3] = &ptA; args[4] = &ciA; args[5] = &tiA; args[6] = &clA; args[7] = &zpA;
        args[8] = &hA; args[9] = &wA; args[10] = &phA; args[11] = &pwA; args[12] = &rfA; args[13] = &rsA;
        long total = (long)paddedT * cIn * (H + 2L * padH) * (W + 2L * padW);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeBuildPaddedBf16 : _wanVaeBuildPadded, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Per-channel RMS norm: out[c] = x[c] * (sqrtC / max(||x||_2, eps)) * gamma[c], over the C axis at each spatial position.</summary>
    public unsafe void LaunchWanVaeRmsNormChannel(
        ulong outp, ulong x, ulong gamma, int c, long spatial, float eps, float sqrtC,
        long numPos, nint stream, bool bf16 = false)
    {
        ulong oA = outp, xA = x, gA = gamma; int cA = c; long spA = spatial, npA = numPos; float eA = eps, scA = sqrtC;
        void** args = stackalloc void*[8];
        args[0] = &oA; args[1] = &xA; args[2] = &gA; args[3] = &cA; args[4] = &spA; args[5] = &eA; args[6] = &scA; args[7] = &npA;
        uint gridDim = (uint)((numPos + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeRmsNormChannelBf16 : _wanVaeRmsNormChannel, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Unpatchifies [b, c*p*p, t, h, w] into [b, c, t, h*p, w*p] via pixel-shuffle.</summary>
    public unsafe void LaunchWanVaeUnpatchify(ulong outp, ulong x, int b, int c, int t, int h, int w, int p, long numOut, nint stream, bool bf16 = false)
    {
        ulong oA = outp, xA = x; int bA = b, cA = c, tA = t, hA = h, wA = w, pA = p; long nA = numOut;
        void** args = stackalloc void*[9];
        args[0] = &oA; args[1] = &xA; args[2] = &bA; args[3] = &cA; args[4] = &tA; args[5] = &hA; args[6] = &wA; args[7] = &pA; args[8] = &nA;
        uint gridDim = (uint)((numOut + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeUnpatchifyBf16 : _wanVaeUnpatchify, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Splits fused attention input [bt, 3c, h, w] into q, k, v, each [bt, 1, hw, c].</summary>
    public unsafe void LaunchWanVaeSplitQkv(ulong q, ulong k, ulong v, ulong src, int bt, int c, int hw, long numEl, nint stream)
    {
        ulong qA = q, kA = k, vA = v, sA = src; int btA = bt, cA = c, hwA = hw; long nA = numEl;
        void** args = stackalloc void*[8];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &sA; args[4] = &btA; args[5] = &cA; args[6] = &hwA; args[7] = &nA;
        uint gridDim = (uint)((numEl + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeSplitQkv, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Inverse of the qkv split: reshapes attention output [bt, 1, hw, c] back into a frame [bt, c, h, w].</summary>
    public unsafe void LaunchWanVaeTokensToFrame(ulong outp, ulong a, int bt, int c, int hw, long numEl, nint stream)
    {
        ulong oA = outp, aA = a; int btA = bt, cA = c, hwA = hw; long nA = numEl;
        void** args = stackalloc void*[6];
        args[0] = &oA; args[1] = &aA; args[2] = &btA; args[3] = &cA; args[4] = &hwA; args[5] = &nA;
        uint gridDim = (uint)((numEl + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeTokensToFrame, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fills output [cOut, tout, HW] with per-channel bias (or 0 when <paramref name="bias"/>=0).</summary>
    public unsafe void LaunchWanVaeFillBias(ulong outp, ulong bias, int cOut, int tout, int HW, nint stream, bool bf16 = false)
    {
        ulong oA = outp, bA = bias; uint coA = (uint)cOut, toA = (uint)tout, hwA = (uint)HW;
        void** args = stackalloc void*[5];
        args[0] = &oA; args[1] = &bA; args[2] = &coA; args[3] = &toA; args[4] = &hwA;
        long total = (long)cOut * tout * HW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeFillBiasBf16 : _wanVaeFillBias, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Temporal gather-sum: out[co][to][hw] += convDt[to·strideT+dt][co][hw]. out=[cOut,tout,HW],
    /// convDt=[paddedT,cOut,HW].</summary>
    public unsafe void LaunchWanVaeAccumulateTap(ulong outp, ulong convDt, int dt, int strideT, int cOut, int tout, int HW, nint stream, bool bf16 = false)
    {
        ulong oA = outp, cA = convDt; uint dtA = (uint)dt, stA = (uint)strideT, coA = (uint)cOut, toA = (uint)tout, hwA = (uint)HW;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &cA; args[2] = &dtA; args[3] = &stA; args[4] = &coA; args[5] = &toA; args[6] = &hwA;
        long total = (long)cOut * tout * HW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _wanVaeAccumulateTapBf16 : _wanVaeAccumulateTap, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>SeedVR2 MAGViT channel-to-space shuffle (gather over output elements; temporal drops output frame 1).</summary>
    public unsafe void LaunchSeedVr2PixelShuffle(ulong outp, ulong inp, int c, int fFinal, int hOut, int wOut,
        int cIn, int f, int h, int w, int sr, int tr, bool dropDup, nint stream, bool bf16 = false)
    {
        ulong oA = outp, iA = inp;
        uint cA = (uint)c, ffA = (uint)fFinal, hoA = (uint)hOut, woA = (uint)wOut;
        uint ciA = (uint)cIn, fA = (uint)f, hA = (uint)h, wA = (uint)w;
        uint srA = (uint)sr, trA = (uint)tr, ddA = dropDup ? 1u : 0u;
        void** args = stackalloc void*[13];
        args[0] = &oA; args[1] = &iA; args[2] = &cA; args[3] = &ffA; args[4] = &hoA; args[5] = &woA;
        args[6] = &ciA; args[7] = &fA; args[8] = &hA; args[9] = &wA; args[10] = &srA; args[11] = &trA; args[12] = &ddA;
        long total = (long)c * fFinal * hOut * wOut;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _seedVr2PixelShuffleBf16 : _seedVr2PixelShuffle,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>SeedVR2 asymmetric zero pad (right/bottom only): [planes,h,w] → [planes,h+1,w+1].</summary>
    public unsafe void LaunchSeedVr2PadBottomRight(ulong outp, ulong inp, long planes, int h, int w, nint stream, bool bf16 = false)
    {
        ulong oA = outp, iA = inp, pA = (ulong)planes;
        uint hA = (uint)h, wA = (uint)w;
        void** args = stackalloc void*[5];
        args[0] = &oA; args[1] = &iA; args[2] = &pA; args[3] = &hA; args[4] = &wA;
        long total = planes * (h + 1L) * (w + 1L);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(bf16 ? _seedVr2PadBrBf16 : _seedVr2PadBr,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── GeGlu Launches ──────────────────────────────────────────────────

    /// <summary>Launches GEGLU: splits input along last dim, applies GELU gate (F32).</summary>
    public void LaunchGeGlu(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_gegluF32, output, input, innerDim, outputElements, stream);

    /// <summary>Launches GEGLU with F16 I/O, F32 internal compute.</summary>
    public void LaunchGeGluF16(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_gegluF16, output, input, innerDim, outputElements, stream);

    /// <summary>Launches GEGLU with BF16 I/O and F32 internal compute.</summary>
    public void LaunchGeGluBf16(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_ditGeGluBf16, output, input, innerDim, outputElements, stream);

    // ── BroadcastAdd Launches ───────────────────────────────────────────

    /// <summary>Launches broadcast add: hidden[b,c,s] += bias[b,c] in-place (F32).</summary>
    public void LaunchBroadcastAdd(ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
        => LaunchBroadcastAddImpl(_broadcastAddF32, hidden, bias, channels, spatial, totalElements, stream);

    /// <summary>Launches broadcast add: hidden[b,c,s] += bias[b,c] in-place (F16).</summary>
    public void LaunchBroadcastAddF16(ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
        => LaunchBroadcastAddImpl(_broadcastAddF16, hidden, bias, channels, spatial, totalElements, stream);

    /// <summary>Launches broadcast add: hidden[b,c,s] += bias[b,c] in-place (BF16).</summary>
    public void LaunchBroadcastAddBf16(ulong hidden, ulong bias, int channels, int spatial, int totalElements, nint stream)
        => LaunchBroadcastAddImpl(_broadcastAddBf16, hidden, bias, channels, spatial, totalElements, stream);

    // ── Cast Launches ───────────────────────────────────────────────────

    /// <summary>Launches FP32 to FP16 cast.</summary>
    public void LaunchCastF32ToF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF32ToF16, output, input, count, stream);

    /// <summary>Launches FP16 to FP32 cast.</summary>
    public void LaunchCastF16ToF32(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF16ToF32, output, input, count, stream);

    /// <summary>Launches FP8 E4M3 to FP16 cast. Input is 1 byte/element, output is 2 bytes/element.</summary>
    public void LaunchCastF8E4M3ToF16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF8E4M3ToF16, output, input, count, stream);

    /// <summary>Launches FP16 to FP8 E4M3 cast with saturation. Input is 2 bytes/element, output is 1 byte/element.</summary>
    public void LaunchCastF16ToF8E4M3(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF16ToF8E4M3, output, input, count, stream);

    /// <summary>Block count the absmax pass-1 launches for <paramref name="count"/> elements; size the blockMax scratch buffer with this.</summary>
    public static int Fp8AbsMaxBlockCount(int count) => Math.Min(1024, (count + 255) / 256);

    /// <summary>Two-pass per-tensor absmax → writes the e4m3 DEQUANT scale (<c>amax/448</c>, or 1.0 for an all-zero
    /// tensor) to <paramref name="scaleOut"/>[0] (device F32). <paramref name="blockMax"/> is scratch of
    /// ≥ <see cref="Fp8AbsMaxBlockCount"/> floats. Fully async — the scale stays on-device for
    /// CUBLASLT_MATMUL_DESC_B_SCALE_POINTER.</summary>
    public unsafe void LaunchFp8AbsMaxScale(ulong x, ulong blockMax, ulong scaleOut, int count, nint stream)
    {
        uint blocks = (uint)Fp8AbsMaxBlockCount(count);
        ulong xA = x, bmA = blockMax; uint nA = (uint)count;
        void** args = stackalloc void*[3];
        args[0] = &xA; args[1] = &bmA; args[2] = &nA;
        CudaDriverApi.cuLaunchKernel(_fp8AbsMax, blocks, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
        ulong scA = scaleOut; uint nbA = blocks;
        void** args2 = stackalloc void*[3];
        args2[0] = &bmA; args2[1] = &nbA; args2[2] = &scA;
        CudaDriverApi.cuLaunchKernel(_fp8AbsMaxFinalizeScale, 1, 1, 1, 256, 1, 1, 0, stream, (nint)args2, 0).ThrowOnError();
    }

    /// <summary>Quantizes F32 → e4m3fn: <c>out[i] = e4m3(x[i] · 448/amax)</c>, reading the dequant scale produced by
    /// <see cref="LaunchFp8AbsMaxScale"/> from device memory (no host sync).</summary>
    public unsafe void LaunchFp8QuantF32ToE4M3(ulong output, ulong input, ulong scale, int count, nint stream)
    {
        ulong iA = input, oA = output, sA = scale; uint nA = (uint)count;
        void** args = stackalloc void*[4];
        args[0] = &iA; args[1] = &oA; args[2] = &sA; args[3] = &nA;
        uint gridDim = (uint)((count + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_fp8QuantF32ToE4M3, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>F16-activation twin of <see cref="LaunchFp8AbsMaxScale"/>: two-pass |max| over __half input →
    /// scale[0] = amax/448 on device. Keeps the native fp8 GEMM available to F16 activations.</summary>
    public unsafe void LaunchFp8AbsMaxScaleF16(ulong x, ulong blockMax, ulong scaleOut, int count, nint stream)
    {
        uint blocks = (uint)Fp8AbsMaxBlockCount(count);
        ulong xA = x, bmA = blockMax; uint nA = (uint)count;
        void** args = stackalloc void*[3];
        args[0] = &xA; args[1] = &bmA; args[2] = &nA;
        CudaDriverApi.cuLaunchKernel(_fp8AbsMaxF16, blocks, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
        ulong scA = scaleOut; uint nbA = blocks;
        void** args2 = stackalloc void*[3];
        args2[0] = &bmA; args2[1] = &nbA; args2[2] = &scA;
        CudaDriverApi.cuLaunchKernel(_fp8AbsMaxFinalizeScale, 1, 1, 1, 256, 1, 1, 0, stream, (nint)args2, 0).ThrowOnError();
    }

    /// <summary>F16-activation twin of <see cref="LaunchFp8QuantF32ToE4M3"/>: out[i] = e4m3(h2f(x[i]) · 448/amax).</summary>
    public unsafe void LaunchFp8QuantF16ToE4M3(ulong output, ulong input, ulong scale, int count, nint stream)
    {
        ulong iA = input, oA = output, sA = scale; uint nA = (uint)count;
        void** args = stackalloc void*[4];
        args[0] = &iA; args[1] = &oA; args[2] = &sA; args[3] = &nA;
        uint gridDim = (uint)((count + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_fp8QuantF16ToE4M3, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches BF16 to F32 cast (lossless — BF16 is the upper 16 bits of F32). 2 bytes/element in, 4 out.</summary>
    public void LaunchCastBf16ToF32(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castBf16ToF32, output, input, count, stream);

    /// <summary>Launches F32 to BF16 cast with round-to-nearest-even. Input is 4 bytes/element, output is 2 bytes/element.</summary>
    public void LaunchCastF32ToBf16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF32ToBf16, output, input, count, stream);

    // ── GGUF Dequant Launches ────────────────────────────────────────────

    /// <summary>Launches Q8_0 → F16 dequant, one CUDA block (32 threads) per 32-element Q8_0 quant block.</summary>
    /// <param name="elementCount">Total element count; must be a multiple of 32.</param>
    public unsafe void LaunchDequantQ8_0ToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 32 != 0)
            throw new ArgumentException($"Q8_0 element count must be a multiple of 32, got {elementCount}.");
        int superBlockCount = elementCount / 32;
        LaunchDequantImpl(_dequantQ8_0ToF16, output, input, superBlockCount, threadsPerBlock: 32, stream);
    }

    /// <summary>Launches Q4_0 → F16 dequant. Legacy 32-element block (18 bytes: fp16 scale + 16 nibble bytes).</summary>
    public unsafe void LaunchDequantQ4_0ToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 32 != 0)
            throw new ArgumentException($"Q4_0 element count must be a multiple of 32, got {elementCount}.");
        int blockCount = elementCount / 32;
        LaunchDequantImpl(_dequantQ4_0ToF16, output, input, blockCount, threadsPerBlock: 32, stream);
    }

    /// <summary>Launches Q5_0 → F16 dequant. Legacy 32-element block (22 bytes: fp16 scale + uint32 high-bits + 16 nibble bytes).</summary>
    public unsafe void LaunchDequantQ5_0ToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 32 != 0)
            throw new ArgumentException($"Q5_0 element count must be a multiple of 32, got {elementCount}.");
        int blockCount = elementCount / 32;
        LaunchDequantImpl(_dequantQ5_0ToF16, output, input, blockCount, threadsPerBlock: 32, stream);
    }

    /// <summary>Launches Q4_K → F16 dequant. Element count must be a multiple of 256 (super-block size).</summary>
    public unsafe void LaunchDequantQ4_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q4_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
        LaunchDequantImpl(_dequantQ4_KToF16, output, input, superBlockCount, threadsPerBlock: 256, stream);
    }

    /// <summary>Launches Q5_K → F16 dequant. Element count must be a multiple of 256.</summary>
    public unsafe void LaunchDequantQ5_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q5_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
        LaunchDequantImpl(_dequantQ5_KToF16, output, input, superBlockCount, threadsPerBlock: 256, stream);
    }

    /// <summary>Launches Q6_K → F16 dequant. Element count must be a multiple of 256.</summary>
    public unsafe void LaunchDequantQ6_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q6_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
        // 64 threads per CUDA block, each emitting 4 elements at strides {0, +32, +64, +96}
        // (2 halves × 32 l-values = 64 threads cover all 256 elements of the super-block).
        LaunchDequantImpl(_dequantQ6_KToF16, output, input, superBlockCount, threadsPerBlock: 64, stream);
    }

    /// <summary>Fused Q4_K × F32 matrix-vector product for decode (M small). Computes
    /// output[M,N] = input[M,K] × dequant(weight[N,K])^T (+ bias), reading the Q4_K bytes once and
    /// dequantizing inline — no F16 weight materialization. K must be a multiple of 256 (guaranteed
    /// for Q4_K). One CUDA block (256 threads) per output element; grid = (N, M).</summary>
    public void LaunchMulMatVecQ4KF32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecQ4KF32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Fused Q6_K × F32 matrix-vector product for decode (M small). Same geometry as the Q4_K GEMV;
    /// used for the lm_head / output projection (commonly Q6_K).</summary>
    public void LaunchMulMatVecQ6KF32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecQ6KF32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Fused Q8_0 × F32 matrix-vector product for decode (M small). Same geometry as the Q4_K GEMV.</summary>
    public void LaunchMulMatVecQ8_0F32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecQ8_0F32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Dense BF16-weight × F32-activation GEMV (F32 accumulate) for small-M decode.</summary>
    public void LaunchMulMatVecBf16F32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecFloatImpl(_mulMatVecBf16F32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Dense F16-weight × F32-activation GEMV (F32 accumulate) for small-M decode.</summary>
    public void LaunchMulMatVecF16F32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecFloatImpl(_mulMatVecF16F32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Fused Q5_0 × F32 matrix-vector product for decode (M small). Same geometry as the Q4_K GEMV.
    /// Q5_0 is llama.cpp's fallback quant (in Q4_K_M-style mixed schemes) for any tensor whose K isn't a
    /// multiple of 256, so without this kernel those tensors silently miss every fused GEMV path.</summary>
    public void LaunchMulMatVecQ5_0F32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecQ5_0F32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Fused Q4_0 × F32 matrix-vector product for decode (M small). Same geometry as the Q5_0 GEMV.
    /// Q4_0 is a common legacy/baseline GGUF quant that previously missed every fused GEMV path.</summary>
    public void LaunchMulMatVecQ4_0F32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecQ4_0F32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Fused Q5_K × F32 matrix-vector product for decode (M small). Same geometry as the Q4_K GEMV,
    /// extended with Q5_K's extra high-bit plane.</summary>
    public void LaunchMulMatVecQ5KF32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecQ5KF32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Quantizes an F32 activation [M,K] to int8 (Q8_1): xq int8 + per-32-block scale xd + int-sum xs.
    /// One warp per 32-block.</summary>
    public unsafe void LaunchQuantizeActivationQ8_1(ulong xq, ulong xd, ulong xs, ulong x, int M, int K, nint stream)
    {
        int nblocks = M * (K / 32);
        ulong xqA = xq, xdA = xd, xsA = xs, xA = x; int nbA = nblocks;
        void** args = stackalloc void*[5];
        args[0] = &xqA; args[1] = &xdA; args[2] = &xsA; args[3] = &xA; args[4] = &nbA;
        const uint WARPS_PER_BLOCK = 8;
        uint grid = ((uint)nblocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(_quantActQ8_1F32, grid, 1, 1, 32, WARPS_PER_BLOCK, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused Q4_K × Q8_1 dp4a matrix-vector product for decode (M small). Consumes the pre-quantized
    /// int8 activation (xq/xd/xs from <see cref="LaunchQuantizeActivationQ8_1"/>).</summary>
    public unsafe void LaunchMulMatVecQ4KQ8_1(ulong output, ulong xq, ulong xd, ulong xs, ulong weight, ulong bias, int N, int K, int M, nint stream)
    {
        ulong outA = output, xqA = xq, xdA = xd, xsA = xs, wA = weight, bA = bias;
        int nA = N, kA = K, mA = M;
        void** args = stackalloc void*[9];
        args[0] = &outA; args[1] = &xqA; args[2] = &xdA; args[3] = &xsA; args[4] = &wA; args[5] = &bA;
        args[6] = &nA; args[7] = &kA; args[8] = &mA;
        uint ksplit = GemvKsplitWarps(N, K);
        if (ksplit > 1)
        {
            CudaDriverApi.cuLaunchKernel(_mulMatVecQ4KQ8_1Ksplit, (uint)N, (uint)M, 1, 32, ksplit, 1, 0, stream, (nint)args, 0).ThrowOnError();
            return;
        }
        uint WARPS_PER_BLOCK = (uint)_wpbOverride;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(_mulMatVecQ4KQ8_1, gridX, (uint)M, 1, 32, WARPS_PER_BLOCK, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    // Block-per-row K-split policy for the dp4a GEMV kernels: warp-per-row leaves long-K/small-N shapes
    // (the ffn_down class — DeepSeek-1.5B's 8960×1536 measured 51% of DRAM peak, ~1.1 waves of warps)
    // under-occupied; splitting each row across 4 warps multiplies resident parallelism with a
    // deterministic shared-memory combine. Gated tightly: at N ≥ ~2560 warp-per-row measured equal or
    // better (2026-07-22 sweep). HARTSY_GEMV_KSPLIT=0 disables, =W forces W warps/row everywhere.
    private static readonly int _ksplitOverride =
        int.TryParse(Environment.GetEnvironmentVariable("HARTSY_GEMV_KSPLIT"), out int v) ? v : -1;

    // Rows-per-block for the warp-per-row GEMV kernels (sweep knob HARTSY_GEMV_WPB). Default 4: the
    // 2026-07-23 sweep measured WPB=4 equal-or-faster than the original 8 at every production shape class
    // (Q4_K K=4096: N=256 11.4 vs 13.0 µs, N=4096 37.8 vs 38.4, N=27392 197.1 vs 198.1) — finer blocks
    // (128 threads) smooth the launch/drain tail; resident-warp occupancy is unchanged (12 blocks/SM).
    private static readonly int _wpbOverride =
        int.TryParse(Environment.GetEnvironmentVariable("HARTSY_GEMV_WPB"), out int w) ? Math.Clamp(w, 1, 16) : 4;

    private static uint GemvKsplitWarps(int rows, int k)
    {
        if (_ksplitOverride == 0) return 1;
        if (_ksplitOverride > 1) return (uint)Math.Min(_ksplitOverride, 16);
        // Measured sweet spots (RTX 3060, 2026-07-23): DeepSeek-1.5B ffn_down 8960×1536 42.2→38.3 µs,
        // GLM-4-9B ffn_down 13696×4096 Q5_0 168.8→134.3 / Q8_0 205→191.9 µs at W=4. Shorter-K downs
        // (qwen2.5-0.5b 4864, gemma3-1b 6912) measured ambiguous-to-worse under a split (too little
        // work per warp), and the square 4096×4096 attention shape clearly regresses — hence the high
        // K floor.
        return k >= 8192 && rows <= 4096 ? 4u : 1u;
    }

    /// <summary>Fused Q8_0 × Q8_1 dp4a matrix-vector product for decode (M small). Q8_0 is symmetric, so only
    /// the int8 activation and per-block scale are consumed (no int-sum term).</summary>
    public unsafe void LaunchMulMatVecQ8_0Q8_1(ulong output, ulong xq, ulong xd, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecQ8_1Impl(_mulMatVecQ8_0Q8_1, _mulMatVecQ8_0Q8_1Ksplit, output, xq, xd, weight, bias, N, K, M, stream);

    /// <summary>Fused Q6_K × Q8_1 dp4a matrix-vector product for decode (M small). Q6_K scales are signed
    /// (symmetric), so only the int8 activation and per-block scale are consumed (no int-sum term).</summary>
    public unsafe void LaunchMulMatVecQ6KQ8_1(ulong output, ulong xq, ulong xd, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecQ8_1Impl(_mulMatVecQ6KQ8_1, _mulMatVecQ6KQ8_1Ksplit, output, xq, xd, weight, bias, N, K, M, stream);

    /// <summary>Fused Q4_0 × Q8_1 dp4a matrix-vector product for decode (M small). The fixed −8 offset is
    /// folded into the packed weights via <c>__vsub4</c>, so no int-sum term is consumed.</summary>
    public unsafe void LaunchMulMatVecQ4_0Q8_1(ulong output, ulong xq, ulong xd, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecQ8_1Impl(_mulMatVecQ4_0Q8_1, 0, output, xq, xd, weight, bias, N, K, M, stream);

    /// <summary>Fused Q5_0 × Q8_1 dp4a matrix-vector product for decode (M small). The fixed −16 offset is
    /// folded into the packed weights via <c>__vsub4</c>, so no int-sum term is consumed.</summary>
    public unsafe void LaunchMulMatVecQ5_0Q8_1(ulong output, ulong xq, ulong xd, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecQ8_1Impl(_mulMatVecQ5_0Q8_1, _mulMatVecQ5_0Q8_1Ksplit, output, xq, xd, weight, bias, N, K, M, stream);

    /// <summary>Fused Q5_K × Q8_1 dp4a matrix-vector product for decode (M small). Like Q4_K, the per-sub-block
    /// min term consumes the Q8_1 int-sum xs.</summary>
    public unsafe void LaunchMulMatVecQ5KQ8_1(ulong output, ulong xq, ulong xd, ulong xs, ulong weight, ulong bias, int N, int K, int M, nint stream)
    {
        ulong outA = output, xqA = xq, xdA = xd, xsA = xs, wA = weight, bA = bias;
        int nA = N, kA = K, mA = M;
        void** args = stackalloc void*[9];
        args[0] = &outA; args[1] = &xqA; args[2] = &xdA; args[3] = &xsA; args[4] = &wA; args[5] = &bA;
        args[6] = &nA; args[7] = &kA; args[8] = &mA;
        uint WARPS_PER_BLOCK = (uint)_wpbOverride;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(_mulMatVecQ5KQ8_1, gridX, (uint)M, 1, 32, WARPS_PER_BLOCK, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchMulMatVecQ8_1Impl(
        nint func, nint ksplitFunc, ulong output, ulong xq, ulong xd, ulong weight, ulong bias,
        int N, int K, int M, nint stream)
    {
        ulong outA = output, xqA = xq, xdA = xd, wA = weight, bA = bias;
        int nA = N, kA = K, mA = M;
        void** args = stackalloc void*[8];
        args[0] = &outA; args[1] = &xqA; args[2] = &xdA; args[3] = &wA; args[4] = &bA;
        args[5] = &nA; args[6] = &kA; args[7] = &mA;
        uint ksplit = ksplitFunc != 0 ? GemvKsplitWarps(N, K) : 1;
        if (ksplit > 1)
        {
            CudaDriverApi.cuLaunchKernel(ksplitFunc, (uint)N, (uint)M, 1, 32, ksplit, 1, 0, stream, (nint)args, 0).ThrowOnError();
            return;
        }
        uint WARPS_PER_BLOCK = (uint)_wpbOverride;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(func, gridX, (uint)M, 1, 32, WARPS_PER_BLOCK, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchMulMatVecImpl(nint func, ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
    {
        ulong outA = output, inA = input, wA = weight, bA = bias;
        int nA = N, kA = K, mA = M;
        void** args = stackalloc void*[7];
        args[0] = &outA; args[1] = &inA; args[2] = &wA; args[3] = &bA;
        args[4] = &nA; args[5] = &kA; args[6] = &mA;
        // One warp per output row, WARPS_PER_BLOCK rows per block, one grid.y slice per batch row.
        // NOTE: this impl is shared by ALL the dequant/float GEMV kernels — only the bf16/f16 kernels
        // have the in-block multi-M path, so the small-M gridY collapse lives in
        // LaunchMulMatVecFloatImpl, NOT here (an earlier version collapsed gridY for every caller and
        // silently dropped batch rows 1..M-1 from the quant kernels — caught by FusedGemvGroundTruthTests).
        uint WARPS_PER_BLOCK = (uint)_wpbOverride;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(
            func,
            gridX, (uint)M, 1,
            32, WARPS_PER_BLOCK, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launch impl for the bf16/f16 float GEMV kernels only: for M ≤ 4 they run their in-block
    /// multi-M row-reuse path under a single grid.y slice (each weight element read once for all M rows).</summary>
    private unsafe void LaunchMulMatVecFloatImpl(nint func, ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
    {
        ulong outA = output, inA = input, wA = weight, bA = bias;
        int nA = N, kA = K, mA = M;
        void** args = stackalloc void*[7];
        args[0] = &outA; args[1] = &inA; args[2] = &wA; args[3] = &bA;
        args[4] = &nA; args[5] = &kA; args[6] = &mA;
        uint WARPS_PER_BLOCK = (uint)_wpbOverride;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        uint gridY = M >= 2 && M <= 4 ? 1u : (uint)M;
        CudaDriverApi.cuLaunchKernel(
            func,
            gridX, gridY, 1,
            32, WARPS_PER_BLOCK, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchDequantImpl(nint func, ulong output, ulong input, int superBlockCount, int threadsPerBlock, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint countArg = (uint)superBlockCount;
        void** args = stackalloc void*[3];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &countArg;
        CudaDriverApi.cuLaunchKernel(
            func,
            (uint)superBlockCount, 1, 1,
            (uint)threadsPerBlock, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── Dispose ─────────────────────────────────────────────────────────

    private void ReleaseOwnedModules(bool throwOnError)
    {
        List<Exception>? failures = null;
        foreach (CudaModule module in _ownedModules)
        {
            if (!throwOnError)
            {
                module.DisposeNoThrow();
                continue;
            }

            try { module.Dispose(); }
            catch (Exception error) { (failures ??= []).Add(error); }
        }
        _ownedModules.Clear();
        if (failures is not null)
            throw new AggregateException("One or more CUDA modules failed to unload.", failures);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        try
        {
            ReleaseOwnedModules(throwOnError: true);
        }
        finally
        {
            GC.SuppressFinalize(this);
        }
    }

    ~CudaKernels()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        try { ReleaseOwnedModules(throwOnError: false); }
        catch { /* Finalizers must never surface managed or native teardown failures. */ }
    }
}
