using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Loads PTX modules from disk and provides typed kernel launch methods. Function handles stored as nint fields for zero-alloc dispatch.</summary>
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
    private readonly nint _lmGatherRowsF32;
    private readonly nint _lmScatterAddWeightedRowsF32;
    private readonly nint _lmArgMaxLastDimF32;
    private readonly nint _lmRopeInterleavedF32;
    private readonly nint _lmRopeDecodeSplitHalf;
    private readonly nint _lmRopeDecodeInterleaved;
    private readonly nint _lmEmbedGatherDecodeF32;
    private readonly nint _lmHistoryAppend;
    private readonly nint _lmRepetitionPenaltyF32;
    private readonly nint _lmKvSliceTimeF32;
    private readonly CudaModule _flashAttnF32Module;
    private readonly nint _flashAttnF32;
    private readonly CudaModule _flashAttnF32SplitModule;
    private readonly nint _flashAttnF32Split;
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
    private readonly nint _wanVaeRmsNormChannel;
    private readonly nint _wanVaeUnpatchify;
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
    private readonly nint _ditUnpatchifyF32;
    private readonly nint _ditTanhF32;
    private readonly nint _ditRopeF32;
    private readonly nint _ltx2SplitRopeF32;
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
    private readonly nint _fourierEmbedF32;
    private readonly nint _conv3dF32;
    private readonly nint _sparseScatterToGridF32;
    private readonly nint _sparseGatherFromGridF32;
    private readonly nint _rowGatherF32;
    private readonly nint _rowScatterAddF32;

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
    private readonly nint _ditRepeatKvF16;
    private readonly nint _ditSliceRowsF16;
    private readonly nint _ditChwToHwcU8;

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
    private CudaModule? _mulMatVecF16Bf16Module;
    private nint _mulMatVecBf16F32;
    private nint _mulMatVecF16F32;
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

    private const uint BlockSize = 256;

    /// <summary>Loads all PTX kernels from the specified directory.</summary>
    public CudaKernels(string ptxDir)
    {
        if (!Directory.Exists(ptxDir))
            throw new DirectoryNotFoundException($"PTX directory not found: {ptxDir}");

        // ── F32 modules ──────────────────────────────────────────────────
        _elementwiseModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "elementwise_f32.ptx"));
        _addF32 = _elementwiseModule.GetFunction("elementwise_add_f32");
        _mulF32 = _elementwiseModule.GetFunction("elementwise_mul_f32");
        _scaleF32 = _elementwiseModule.GetFunction("elementwise_scale_f32");
        _siluF32 = _elementwiseModule.GetFunction("elementwise_silu_f32");
        _geluF32 = _elementwiseModule.GetFunction("elementwise_gelu_f32");
        _clampF32 = _elementwiseModule.GetFunction("elementwise_clamp_f32");

        _groupnormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_f32.ptx"));
        _groupnormF32 = _groupnormModule.GetFunction("groupnorm_f32");

        _layernormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "layernorm_f32.ptx"));
        _layernormF32 = _layernormModule.GetFunction("layernorm_f32");

        _spatialModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "spatial_f32.ptx"));
        _upsampleNearest2dF32 = _spatialModule.GetFunction("upsample_nearest2d_f32");
        _im2colF32 = _spatialModule.GetFunction("im2col_f32");
        _col2biasAddF32 = _spatialModule.GetFunction("col2bias_add_f32");

        _im2colBandedModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "im2col_banded.ptx"));
        _im2colBandedF32 = _im2colBandedModule.GetFunction("im2col_banded_f32");
        _im2colBandedF16 = _im2colBandedModule.GetFunction("im2col_banded_f16");
        _im2colBandedBf16 = _im2colBandedModule.GetFunction("im2col_banded_bf16");

        _maxpool2dModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "maxpool2d.ptx"));
        _maxpool2dF32 = _maxpool2dModule.GetFunction("maxpool2d_f32");
        _maxpool2dF16 = _maxpool2dModule.GetFunction("maxpool2d_f16");

        _depthwiseConv2dModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "depthwise_conv2d.ptx"));
        _depthwiseConv2dF32 = _depthwiseConv2dModule.GetFunction("depthwise_conv2d_f32");
        _depthwiseConv2dF16 = _depthwiseConv2dModule.GetFunction("depthwise_conv2d_f16");

        _msdaModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "msda.ptx"));
        _msdaForwardF32 = _msdaModule.GetFunction("msda_forward_f32");

        _softmaxModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "softmax_f32.ptx"));
        _softmaxF32 = _softmaxModule.GetFunction("softmax_f32");

        _transposeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "transpose_f32.ptx"));
        _transpose2dF32 = _transposeModule.GetFunction("transpose_2d_f32");
        _permute0213F32 = _transposeModule.GetFunction("permute_0213_f32");

        _wanRopeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "wan_rope.ptx"));
        _wanRopeInterleaved = _wanRopeModule.GetFunction("wan_rope_interleaved");
        _wanRopeInterleavedPerHead = _wanRopeModule.GetFunction("wan_rope_interleaved_perhead");

        _wanVaeFramesModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "wan_vae_frames.ptx"));
        _wanVaeExtractFrame = _wanVaeFramesModule.GetFunction("wan_vae_extract_frame");
        _wanVaeWriteFrame = _wanVaeFramesModule.GetFunction("wan_vae_write_frame");

        _wanVaeConv3dModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "wan_vae_conv3d.ptx"));
        _wanVaeBuildPadded = _wanVaeConv3dModule.GetFunction("wan_vae_build_padded");
        _wanVaeFillBias = _wanVaeConv3dModule.GetFunction("wan_vae_fill_bias");
        _wanVaeAccumulateTap = _wanVaeConv3dModule.GetFunction("wan_vae_accumulate_tap");

        _wanVaeNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "wan_vae_norm.ptx"));
        _wanVaeRmsNormChannel = _wanVaeNormModule.GetFunction("wan_vae_rms_norm_channel");
        _wanVaeUnpatchify = _wanVaeNormModule.GetFunction("wan_vae_unpatchify");
        _wanVaeSplitQkv = _wanVaeNormModule.GetFunction("wan_vae_split_qkv");
        _wanVaeTokensToFrame = _wanVaeNormModule.GetFunction("wan_vae_tokens_to_frame");

        _gegluModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "geglu_f32.ptx"));
        _gegluF32 = _gegluModule.GetFunction("geglu_f32");

        _broadcastAddModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "broadcast_add_f32.ptx"));
        _broadcastAddF32 = _broadcastAddModule.GetFunction("broadcast_add_f32");

        // ── F16 modules ──────────────────────────────────────────────────
        _elementwiseF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "elementwise_f16.ptx"));
        _addF16 = _elementwiseF16Module.GetFunction("elementwise_add_f16");
        _mulF16 = _elementwiseF16Module.GetFunction("elementwise_mul_f16");
        _scaleF16 = _elementwiseF16Module.GetFunction("elementwise_scale_f16");
        _siluF16 = _elementwiseF16Module.GetFunction("elementwise_silu_f16");
        _geluF16 = _elementwiseF16Module.GetFunction("elementwise_gelu_f16");
        _clampF16 = _elementwiseF16Module.GetFunction("elementwise_clamp_f16");

        _groupnormF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_f16.ptx"));
        _groupnormF16 = _groupnormF16Module.GetFunction("groupnorm_f16");

        _layernormF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "layernorm_f16.ptx"));
        _layernormF16 = _layernormF16Module.GetFunction("layernorm_f16");

        _spatialF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "spatial_f16.ptx"));
        _upsampleNearest2dF16 = _spatialF16Module.GetFunction("upsample_nearest2d_f16");
        _im2colF16 = _spatialF16Module.GetFunction("im2col_f16");
        _col2biasAddF16 = _spatialF16Module.GetFunction("col2bias_add_f16");

        _softmaxF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "softmax_f16.ptx"));
        _softmaxF16 = _softmaxF16Module.GetFunction("softmax_f16");

        _transposeF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "transpose_f16.ptx"));
        _transpose2dF16 = _transposeF16Module.GetFunction("transpose_2d_f16");
        _permute0213F16 = _transposeF16Module.GetFunction("permute_0213_f16");

        _gegluF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "geglu_f16.ptx"));
        _gegluF16 = _gegluF16Module.GetFunction("geglu_f16");

        _broadcastAddF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "broadcast_add_f16.ptx"));
        _broadcastAddF16 = _broadcastAddF16Module.GetFunction("broadcast_add_f16");

        // ── BF16 modules (subset VAE needs; SDXL VAE F16 overflows) ──────
        _elementwiseBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "elementwise_bf16.ptx"));
        _addBf16 = _elementwiseBf16Module.GetFunction("elementwise_add_bf16");
        _mulBf16 = _elementwiseBf16Module.GetFunction("elementwise_mul_bf16");
        _scaleBf16 = _elementwiseBf16Module.GetFunction("elementwise_scale_bf16");
        _siluBf16 = _elementwiseBf16Module.GetFunction("elementwise_silu_bf16");
        _geluBf16 = _elementwiseBf16Module.GetFunction("elementwise_gelu_bf16");
        _clampBf16 = _elementwiseBf16Module.GetFunction("elementwise_clamp_bf16");

        _groupnormBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_bf16.ptx"));
        _groupnormBf16 = _groupnormBf16Module.GetFunction("groupnorm_bf16");

        _layernormBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "layernorm_bf16.ptx"));
        _layernormBf16 = _layernormBf16Module.GetFunction("layernorm_bf16");

        _spatialBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "spatial_bf16.ptx"));
        _upsampleNearest2dBf16 = _spatialBf16Module.GetFunction("upsample_nearest2d_bf16");
        _im2colBf16 = _spatialBf16Module.GetFunction("im2col_bf16");
        _col2biasAddBf16 = _spatialBf16Module.GetFunction("col2bias_add_bf16");

        _broadcastAddBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "broadcast_add_bf16.ptx"));
        _broadcastAddBf16 = _broadcastAddBf16Module.GetFunction("broadcast_add_bf16");

        _groupnormSiluBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_silu_bf16.ptx"));
        _groupnormSiluBf16 = _groupnormSiluBf16Module.GetFunction("groupnorm_silu_bf16");

        // ── Fused GroupNorm+SiLU ─────────────────────────────────────────
        _groupnormSiluModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_silu_f32.ptx"));
        _groupnormSiluF32 = _groupnormSiluModule.GetFunction("groupnorm_silu_f32");

        _groupnormSiluF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "groupnorm_silu_f16.ptx"));
        _groupnormSiluF16 = _groupnormSiluF16Module.GetFunction("groupnorm_silu_f16");

        // ── Cast ─────────────────────────────────────────────────────────
        _castModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "cast_f32_f16.ptx"));
        _castF32ToF16 = _castModule.GetFunction("cast_f32_to_f16");
        _castF16ToF32 = _castModule.GetFunction("cast_f16_to_f32");

        // ── FP8 Cast ─────────────────────────────────────────────────────
        _castF8Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "cast_f8e4m3_f16.ptx"));
        _castF8E4M3ToF16 = _castF8Module.GetFunction("cast_f8e4m3_to_f16");
        _castF16ToF8E4M3 = _castF8Module.GetFunction("cast_f16_to_f8e4m3");

        // ── FP8 Activation Quantization ──────────────────────────────────
        _fp8QuantModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "fp8_quant.ptx"));
        _fp8AbsMax = _fp8QuantModule.GetFunction("absmax_f32");
        _fp8AbsMaxFinalizeScale = _fp8QuantModule.GetFunction("absmax_finalize_scale");
        _fp8QuantF32ToE4M3 = _fp8QuantModule.GetFunction("quant_f32_e4m3");
        _fp8AbsMaxF16 = _fp8QuantModule.GetFunction("absmax_f16");
        _fp8QuantF16ToE4M3 = _fp8QuantModule.GetFunction("quant_f16_e4m3");

        // ── BF16 <-> F32 Cast ───────────────────────────────────────────
        _castBf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "cast_bf16_f32.ptx"));
        _castBf16ToF32 = _castBf16Module.GetFunction("cast_bf16_to_f32");
        _castF32ToBf16 = _castBf16Module.GetFunction("cast_f32_to_bf16");

        // ── DiT glue (F32) ───────────────────────────────────────────────
        _ditF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dit_f32.ptx"));
        _ditRmsNormF32 = _ditF32Module.GetFunction("dit_rmsnorm_f32");
        _ditAffineBroadcastF32 = _ditF32Module.GetFunction("dit_affine_broadcast_lastdim_f32");
        _ditGatedResidualF32 = _ditF32Module.GetFunction("dit_gated_residual_lastdim_f32");
        _ditModulation4F32 = _ditF32Module.GetFunction("dit_modulation4_f32");
        _ditCfgEulerF32 = _ditF32Module.GetFunction("dit_cfg_euler_f32");
        _ditUnpatchifyF32 = _ditF32Module.GetFunction("dit_unpatchify_f32");
        _ditTanhF32 = _ditF32Module.GetFunction("dit_tanh_f32");
        _ditRopeF32 = _ditF32Module.GetFunction("dit_rope_f32");
        _ltx2SplitRopeF32 = _ditF32Module.GetFunction("ltx2_split_rope_f32");
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
        _fourierEmbedF32 = _ditF32Module.GetFunction("fourier_embed_f32");
        _conv3dF32 = _ditF32Module.GetFunction("conv3d_f32");
        _sparseScatterToGridF32 = _ditF32Module.GetFunction("sparse_scatter_to_grid_f32");
        _sparseGatherFromGridF32 = _ditF32Module.GetFunction("sparse_gather_from_grid_f32");
        _rowGatherF32 = _ditF32Module.GetFunction("row_gather_f32");
        _rowScatterAddF32 = _ditF32Module.GetFunction("row_scatter_add_f32");

        _mg3ActionModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mg3_action.ptx"));
        _mg3SplitQkvTemporalF32 = _mg3ActionModule.GetFunction("mg3_split_qkv_temporal_f32");
        _mg3MergeTemporalF32 = _mg3ActionModule.GetFunction("mg3_merge_temporal_f32");
        _mg3RopeBatchedF32 = _mg3ActionModule.GetFunction("mg3_rope_batched_f32");
        _mg3KvExpandF32 = _mg3ActionModule.GetFunction("mg3_kv_expand_f32");
        _mg3MouseMlpConcatF32 = _mg3ActionModule.GetFunction("mg3_mouse_mlp_concat_f32");

        // ── DiT glue (F16 I/O, F32 accumulate) — DiT F16 activation path ─
        _ditF16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dit_f16.ptx"));
        _ditRmsNormF16 = _ditF16Module.GetFunction("dit_rmsnorm_f16");
        _ditLayerNormNoAffineF16 = _ditF16Module.GetFunction("dit_layernorm_noaffine_f16");
        _ditAffineBroadcastF16 = _ditF16Module.GetFunction("dit_affine_broadcast_lastdim_f16");
        _ditGatedResidualF16 = _ditF16Module.GetFunction("dit_gated_residual_lastdim_f16");
        _ditAddScalarF16 = _ditF16Module.GetFunction("dit_add_scalar_f16");
        _ditSigmoidF16 = _ditF16Module.GetFunction("dit_sigmoid_f16");
        _ditRopeInterleavedF16 = _ditF16Module.GetFunction("dit_rope_interleaved_f16");
        _ditRopeF16 = _ditF16Module.GetFunction("dit_rope_f16");
        _ditSliceLastDimF16 = _ditF16Module.GetFunction("dit_slice_lastdim_f16");
        _ditConcat2F16 = _ditF16Module.GetFunction("dit_concat2_f16");
        _ditLayerNormModulateF16 = _ditF16Module.GetFunction("dit_layernorm_modulate_f16");
        _ditQkvSplitNormF16 = _ditF16Module.GetFunction("dit_qkv_split_norm_f16");
        _ditRepeatKvF16 = _ditF16Module.GetFunction("dit_repeat_kv_f16");
        _ditSliceRowsF16 = _ditF16Module.GetFunction("dit_slice_rows_f16");
        _ditChwToHwcU8 = _ditF16Module.GetFunction("dit_chw_f32_to_hwc_u8");

        // ── Audio conv (codec/TTS Conv1d + ConvTranspose1d, F32) ─────────
        _audioConvF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "conv1d_f32.ptx"));
        _conv1dF32 = _audioConvF32Module.GetFunction("conv1d_f32");
        _convTranspose1dF32 = _audioConvF32Module.GetFunction("conv_transpose1d_f32");

        _audioActF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "audio_activations_f32.ptx"));
        _audioSigmoidF32 = _audioActF32Module.GetFunction("audio_sigmoid_f32");
        _audioMishF32 = _audioActF32Module.GetFunction("audio_mish_f32");
        _audioEluF32 = _audioActF32Module.GetFunction("audio_elu_f32");
        _audioLeakyReluF32 = _audioActF32Module.GetFunction("audio_leaky_relu_f32");
        _audioSnakeF32 = _audioActF32Module.GetFunction("audio_snake_f32");
        _audioSnakeBetaF32 = _audioActF32Module.GetFunction("audio_snake_beta_f32");
        _audioPreluF32 = _audioActF32Module.GetFunction("audio_prelu_f32");
        _audioRepeatTimeF32 = _audioActF32Module.GetFunction("audio_repeat_time_f32");

        _audioAdain1dF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "adain1d_f32.ptx"));
        _audioAdain1dF32 = _audioAdain1dF32Module.GetFunction("audio_adain1d_f32");

        // ── Language-model glue (F32) ────────────────────────────────────
        _lmF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "lm_f32.ptx"));
        _lmRepeatKvF32 = _lmF32Module.GetFunction("lm_repeat_kv_f32");
        _lmKvAppendF32 = _lmF32Module.GetFunction("lm_kv_append_f32");
        _lmGatherRowsF32 = _lmF32Module.GetFunction("lm_gather_rows_f32");
        _lmScatterAddWeightedRowsF32 = _lmF32Module.GetFunction("lm_scatter_add_weighted_rows_f32");
        _lmArgMaxLastDimF32 = _lmF32Module.GetFunction("lm_argmax_lastdim_f32");
        _lmRopeInterleavedF32 = _lmF32Module.GetFunction("lm_rope_interleaved_f32");
        _lmRopeDecodeSplitHalf = _lmF32Module.GetFunction("lm_rope_decode_splithalf");
        _lmRopeDecodeInterleaved = _lmF32Module.GetFunction("lm_rope_decode_interleaved");
        _lmEmbedGatherDecodeF32 = _lmF32Module.GetFunction("lm_embed_gather_decode_f32");
        _lmHistoryAppend = _lmF32Module.GetFunction("lm_history_append");
        _lmRepetitionPenaltyF32 = _lmF32Module.GetFunction("lm_repetition_penalty_f32");
        _lmKvSliceTimeF32 = _lmF32Module.GetFunction("lm_kv_slice_time_f32");
        _flashAttnF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "flash_attn_f32.ptx"));
        _flashAttnF32 = _flashAttnF32Module.GetFunction("lm_flash_attn_f32");
        _flashAttnF32SplitModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "flash_attn_f32_split.ptx"));
        _flashAttnF32Split = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f32_split");
        _flashAttnF32Combine = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f32_combine");
        _flashV2Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "flash_attn_v2_tf32.ptx"));
        _flashV2Tf32 = _flashV2Module.GetFunction("lm_flash_attn_v2_tf32");
        // Opt the fused flash kernel into >48 KB dynamic shared memory (K/V/S/O tiles ≈ 72 KB for D=128).
        // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8. Ignore failure (kernel launch will surface it).
        CudaDriverApi.cuFuncSetAttribute(_flashV2Tf32, 8, 96 * 1024);

        // ── GGUF Dequant ─────────────────────────────────────────────────
        _dequantQ8_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q8_0_to_f16.ptx"));
        _dequantQ8_0ToF16 = _dequantQ8_0Module.GetFunction("dequant_q8_0_to_f16");
        _dequantQ4_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q4_0_to_f16.ptx"));
        _dequantQ4_0ToF16 = _dequantQ4_0Module.GetFunction("dequant_q4_0_to_f16");
        _dequantQ5_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q5_0_to_f16.ptx"));
        _dequantQ5_0ToF16 = _dequantQ5_0Module.GetFunction("dequant_q5_0_to_f16");
        _dequantQ4_KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q4_k_to_f16.ptx"));
        _dequantQ4_KToF16 = _dequantQ4_KModule.GetFunction("dequant_q4_k_to_f16");
        _dequantQ5_KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q5_k_to_f16.ptx"));
        _dequantQ5_KToF16 = _dequantQ5_KModule.GetFunction("dequant_q5_k_to_f16");
        _dequantQ6_KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "dequant_q6_k_to_f16.ptx"));
        _dequantQ6_KToF16 = _dequantQ6_KModule.GetFunction("dequant_q6_k_to_f16");

        _mulMatVecQ4KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q4k_f32.ptx"));
        _mulMatVecQ4KF32 = _mulMatVecQ4KModule.GetFunction("mul_mat_vec_q4k_f32");
        _mulMatVecQ6KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q6k_f32.ptx"));
        _mulMatVecQ6KF32 = _mulMatVecQ6KModule.GetFunction("mul_mat_vec_q6k_f32");
        _mulMatVecQ8_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q8_0_f32.ptx"));
        _mulMatVecQ8_0F32 = _mulMatVecQ8_0Module.GetFunction("mul_mat_vec_q8_0_f32");
        // On by default (HARTSY_BF16_GEMV=0 disables). The PTX targets sm_80 like the engine's other lm/world
        // kernels, so it JITs on every GPU this engine already runs on; a genuine load failure is caught below
        // and falls back to cuBLAS. This is a large decode win — see the lm_head GEMV note in GenericTransformer.
        if (Environment.GetEnvironmentVariable("HARTSY_BF16_GEMV") != "0")
        {
            try
            {
                _mulMatVecF16Bf16Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_f16_bf16_f32.ptx"));
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
        _mulMatVecQ5_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q5_0_f32.ptx"));
        _mulMatVecQ5_0F32 = _mulMatVecQ5_0Module.GetFunction("mul_mat_vec_q5_0_f32");
        _mulMatVecQ4_0Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q4_0_f32.ptx"));
        _mulMatVecQ4_0F32 = _mulMatVecQ4_0Module.GetFunction("mul_mat_vec_q4_0_f32");
        _mulMatVecQ5KModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q5k_f32.ptx"));
        _mulMatVecQ5KF32 = _mulMatVecQ5KModule.GetFunction("mul_mat_vec_q5k_f32");
        _quantActQ8_1Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "quantize_activation_q8_1_f32.ptx"));
        _quantActQ8_1F32 = _quantActQ8_1Module.GetFunction("quantize_activation_q8_1_f32");
        _mulMatVecQ4KQ8_1Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "mul_mat_vec_q4k_q8_1.ptx"));
        _mulMatVecQ4KQ8_1 = _mulMatVecQ4KQ8_1Module.GetFunction("mul_mat_vec_q4k_q8_1");
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

    /// <summary>Launches Conv1d (F32, channels-first [B,C,T]). One thread per output element.
    /// Pass <paramref name="bias"/>=0 / <paramref name="hasBias"/>=0 when there is no bias.</summary>
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

    /// <summary>Launches ConvTranspose1d (F32, channels-first [B,C,T], weight [C_in,C_out/groups,K]).
    /// Grouped like <see cref="LaunchConv1d"/> — groups==channels is depthwise (BigVGAN anti-aliased upsampling).</summary>
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
    {
        ulong outArg = output, inArg = input;
        int countArg = count;
        void** args = stackalloc void*[3];
        args[0] = &outArg; args[1] = &inArg; args[2] = &countArg;
        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_audioSigmoidF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchAudioMish(ulong output, ulong input, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        int countArg = count;
        void** args = stackalloc void*[3];
        args[0] = &outArg; args[1] = &inArg; args[2] = &countArg;
        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_audioMishF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches the audio Elu activation over <paramref name="count"/> F32 elements.</summary>
    public unsafe void LaunchAudioElu(ulong output, ulong input, float alpha, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        float alphaArg = alpha;
        int countArg = count;
        void** args = stackalloc void*[4];
        args[0] = &outArg; args[1] = &inArg; args[2] = &alphaArg; args[3] = &countArg;
        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_audioEluF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches Leaky ReLU (x if x&gt;=0 else slope*x) over <paramref name="count"/> F32 elements.
    /// StyleTTS 2 / Kokoro / HiFi-GAN / VITS use slope=0.2.</summary>
    public unsafe void LaunchAudioLeakyRelu(ulong output, ulong input, float slope, int count, nint stream)
    {
        ulong outArg = output, inArg = input;
        float slopeArg = slope;
        int countArg = count;
        void** args = stackalloc void*[4];
        args[0] = &outArg; args[1] = &inArg; args[2] = &slopeArg; args[3] = &countArg;
        uint gridDim = ((uint)count + BlockSize - 1) / BlockSize;
        CudaDriverApi.cuLaunchKernel(_audioLeakyReluF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches the Snake activation x + sin²(αx)/α over [B,C,T] F32, α per-channel.
    /// When <paramref name="beta"/>≠0, uses the β-divisor variant x + sin²(αx)/(β+ε).</summary>
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

    /// <summary>Launches parametric ReLU over [B,C,T] F32: x if x&gt;=0 else α·x, α per-channel
    /// (<paramref name="alphaPerChannel"/>=1) or a single shared α (0).</summary>
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

    /// <summary>Launches the frame-repeat over [B,C,inT] F32: each time-step duplicated
    /// <paramref name="numSamples"/> times → [B,C,inT*numSamples].</summary>
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

    /// <summary>Launches Adaptive InstanceNorm 1D over [B,C,T] F32: per-(batch,channel) row,
    /// normalize across T then affine by (1+gamma)/beta. <paramref name="perBatch"/>=1 when
    /// gamma/beta are [B,C], else 0 ([C], broadcast over batch). One block per row.</summary>
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

    /// <summary>Launches banded Im2Col for one batch element: fills the col buffer for output rows
    /// [ohBase, ohBase + bandRows) only. Dtype dispatch mirrors the full-image kernel.</summary>
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

    /// <summary>Multi-scale deformable attention forward. One thread per (query, head, channel); softmax over
    /// <c>levels·points</c> is folded in. Pointer args are all F32 device buffers except <paramref name="shapes"/>
    /// (<c>int[levels*2]</c> h,w) and <paramref name="levelStart"/> (<c>int[levels]</c>).</summary>
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

    /// <summary>Depthwise 2D conv (groups==channels) over an NCHW tensor; weight is <c>[C,1,kH,kW]</c>, bias
    /// optional. One thread per output element. Dtype selects the F32 or F16 kernel.</summary>
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

    /// <summary>Launches GroupNorm with BF16 I/O, FP32 accumulation. Used for SDXL VAE
    /// where F16 activations overflow (resnet activations exceed Â±65504).</summary>
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

    /// <summary>Launches RMSNorm: one block per row, reduces over <paramref name="normDim"/>, applies weight. Also serves per-head QK-RMSNorm (rows = B*L*heads, normDim = headDim).</summary>
    public void LaunchRmsNorm(ulong output, ulong input, ulong weight, int normDim, int totalRows, float eps, nint stream)
        => LaunchRmsNormImpl(_ditRmsNormF32, output, input, weight, normDim, totalRows, eps, stream);

    /// <summary>RMSNorm with F16 I/O (F32 accumulate, F32 weight). Same launch geometry as the F32 kernel.</summary>
    public void LaunchRmsNormF16(ulong output, ulong input, ulong weight, int normDim, int totalRows, float eps, nint stream)
        => LaunchRmsNormImpl(_ditRmsNormF16, output, input, weight, normDim, totalRows, eps, stream);

    /// <summary>Launches broadcast affine over the last dim: out[b,s,d] = in[b,s,d]*scale[b,d] + shift[b,d] (shift optional, pass 0 to skip).</summary>
    public void LaunchAffineBroadcastLastDim(ulong output, ulong input, ulong scale, ulong shift, int seqLen, int dim, long total, nint stream)
        => LaunchAffineBroadcastImpl(_ditAffineBroadcastF32, output, input, scale, shift, seqLen, dim, total, stream);

    /// <summary>Broadcast affine over the last dim, F16 activation I/O (scale/shift stay F32).</summary>
    public void LaunchAffineBroadcastLastDimF16(ulong output, ulong input, ulong scale, ulong shift, int seqLen, int dim, long total, nint stream)
        => LaunchAffineBroadcastImpl(_ditAffineBroadcastF16, output, input, scale, shift, seqLen, dim, total, stream);

    /// <summary>Launches GQA K/V head repeat (block pattern): [B,Hkv,L,D] → [B,Hkv*group,L,D].</summary>
    public void LaunchRepeatKv(ulong output, ulong input, int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
        => LaunchRepeatKvImpl(_lmRepeatKvF32, output, input, kvHeads, group, seqLen, headDim, total, stream);

    /// <summary>GQA K/V head repeat with F16 I/O (pure copy — no accumulation).</summary>
    public void LaunchRepeatKvF16(ulong output, ulong input, int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
        => LaunchRepeatKvImpl(_ditRepeatKvF16, output, input, kvHeads, group, seqLen, headDim, total, stream);

    /// <summary>Launches FlashAttention (one block per (b, q-head, q-row); blockDim = headDim; shared mem =
    /// headDim floats). <paramref name="lk"/> is the K/V buffer seq stride; <paramref name="kvLen"/> the valid
    /// key count.</summary>
    public unsafe void LaunchFlashAttention(ulong outPtr, ulong q, ulong k, ulong v,
        int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap, ulong sink, int slidingWindow, ulong alibiSlopes, nint stream, ulong dPos = 0)
    {
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

        // Block threads = next power of two >= headDim, so the kernel's tree reduction is always power-of-two
        // and non-pow2 head dims (e.g. Phi-3's 96) work; padding threads contribute 0.
        uint blockThreads = 1;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq);
        uint sharedBytes = blockThreads * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _flashAttnF32, gridDim, 1, 1, blockThreads, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fused FlashAttention-2, TF32 tensor cores, F32 accumulate — no-mask MHA, D∈{64,128}, Hkv==Hq.
    /// Q/out [B,Hq,Sq,D], K/V [B,Hq,Skv,D]. Grid (ceil(Sq/64), Hq, B); block 128 (4 warps); BR=64/BC=32 tiles.</summary>
    public unsafe void LaunchFlashAttentionV2Tf32(ulong outPtr, ulong q, ulong k, ulong v,
        int b, int hq, int sq, int skv, int d, float scale, nint stream)
    {
        ulong oArg = outPtr, qArg = q, kArg = k, vArg = v;
        uint bA = (uint)b, hA = (uint)hq, sqA = (uint)sq, skvA = (uint)skv, dA = (uint)d;
        float scA = scale;
        void** args = stackalloc void*[10];
        args[0] = &oArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bA; args[5] = &hA; args[6] = &sqA; args[7] = &skvA; args[8] = &dA; args[9] = &scA;
        const int BR = 32, BC = 16;   // must match flash_attn_v2_tf32.cu #defines (M2: 34KB smem → 2 blocks/SM)
        uint grid = (uint)((sq + BR - 1) / BR);
        uint smem = (uint)((BC * d + BC * d + BR * BC + BR * d) * sizeof(float));
        CudaDriverApi.cuLaunchKernel(_flashV2Tf32, grid, (uint)hq, (uint)b, 64, 1, 1,
            smem, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Flash-decoding split phase (plain path: no sink/alibi/softcap/window). Launches
    /// <c>batch·hq·tq·splits</c> blocks; each computes the partial online-softmax state (m, l, Σp·V) for
    /// its key chunk into the scratch buffers. <paramref name="chunk"/> = ceil(kvLen / splits).</summary>
    public unsafe void LaunchFlashAttentionSplit(ulong partialM, ulong partialL, ulong partialAcc,
        ulong q, ulong k, ulong v, int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen,
        int kvGroup, bool causal, int qOffset, float scale, int splits, int chunk, nint stream, ulong dPos = 0)
    {
        ulong pmArg = partialM, plArg = partialL, paArg = partialAcc, qArg = q, kArg = k, vArg = v, dPosArg = dPos;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim;
        uint hkvArg = (uint)hkv, lkArg = (uint)lk, kvLenArg = (uint)kvLen, grpArg = (uint)kvGroup;
        int causalArg = causal ? 1 : 0, offArg = qOffset;
        float scaleArg = scale;
        uint gArg = (uint)splits, chunkArg = (uint)chunk;

        void** args = stackalloc void*[20];
        args[0] = &pmArg; args[1] = &plArg; args[2] = &paArg; args[3] = &qArg; args[4] = &kArg; args[5] = &vArg;
        args[6] = &bArg; args[7] = &hqArg; args[8] = &tqArg; args[9] = &dArg;
        args[10] = &hkvArg; args[11] = &lkArg; args[12] = &kvLenArg; args[13] = &grpArg;
        args[14] = &causalArg; args[15] = &offArg; args[16] = &scaleArg;
        args[17] = &gArg; args[18] = &chunkArg; args[19] = &dPosArg;

        uint blockThreads = 1;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq * splits);
        uint sharedBytes = blockThreads * sizeof(float);
        CudaDriverApi.cuLaunchKernel(
            _flashAttnF32Split, gridDim, 1, 1, blockThreads, 1, 1,
            sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Flash-decoding combine phase: merges the <paramref name="splits"/> chunk partials per query
    /// into the final output via online softmax. Launches <c>batch·hq·tq</c> blocks.</summary>
    public unsafe void LaunchFlashAttentionCombine(ulong outPtr, ulong partialM, ulong partialL,
        ulong partialAcc, int batch, int hq, int tq, int headDim, int splits, nint stream)
    {
        ulong outArg = outPtr, pmArg = partialM, plArg = partialL, paArg = partialAcc;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim, gArg = (uint)splits;

        void** args = stackalloc void*[9];
        args[0] = &outArg; args[1] = &pmArg; args[2] = &plArg; args[3] = &paArg;
        args[4] = &bArg; args[5] = &hqArg; args[6] = &tqArg; args[7] = &dArg; args[8] = &gArg;

        uint blockThreads = 1;
        while (blockThreads < (uint)headDim) blockThreads <<= 1;
        uint gridDim = (uint)((long)batch * hq * tq);
        CudaDriverApi.cuLaunchKernel(
            _flashAttnF32Combine, gridDim, 1, 1, blockThreads, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches in-place KV-cache append: copies newKv [1,H,tNew,D] into buffer [1,H,maxSeq,D] at offset.</summary>
    public unsafe void LaunchKvAppend(ulong buffer, ulong newKv, int heads, int maxSeq, int tNew, int headDim, int offset, nint stream, ulong dPos = 0)
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

    /// <summary>Launches per-row argmax over the last dim: indices[r] = argmax_c input[r,c]. One block per row,
    /// <paramref name="blockThreads"/> threads (power-of-two), shared mem for the (value,index) reduction.</summary>
    public unsafe void LaunchArgMaxLastDim(ulong idxPtr, ulong inPtr, int rows, int c, nint stream)
    {
        ulong outArg = idxPtr, inArg = inPtr;
        uint cArg = (uint)c;
        void** args = stackalloc void*[3];
        args[0] = &outArg; args[1] = &inArg; args[2] = &cArg;
        uint blockThreads = BlockSize;   // 256, power of two; threads beyond C just hold -FLT_MAX sentinels
        uint sharedBytes = blockThreads * (uint)(sizeof(float) + sizeof(int));
        CudaDriverApi.cuLaunchKernel(_lmArgMaxLastDimF32, (uint)rows, 1, 1, blockThreads, 1, 1, sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Launches gated residual over the last dim: out[b,s,d] = residual[b,s,d] + gate[b,d]*value[b,s,d].</summary>
    public void LaunchGatedResidualLastDim(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total, nint stream)
        => LaunchGatedResidualImpl(_ditGatedResidualF32, output, residual, value, gate, seqLen, dim, total, stream);

    /// <summary>Gated residual over the last dim, F16 activation I/O (gate stays F32).</summary>
    public void LaunchGatedResidualLastDimF16(ulong output, ulong residual, ulong value, ulong gate, int seqLen, int dim, long total, nint stream)
        => LaunchGatedResidualImpl(_ditGatedResidualF16, output, residual, value, gate, seqLen, dim, total, stream);

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

    /// <summary>Launches unpatchify: token grid [hP·wP, C·p²] → pixel latent [C, H, W] (B=1, F32).
    /// <paramref name="innerChannelFastest"/>: token inner order (ph, pw, c) when true (Z-Image), (c, ph, pw) when false (Krea2).</summary>
    public unsafe void LaunchDitUnpatchify(ulong output, ulong input, int channels, int hPacked, int wPacked,
        int patch, bool innerChannelFastest, nint stream)
    {
        ulong outArg = output, inArg = input;
        uint chArg = (uint)channels, hpArg = (uint)hPacked, wpArg = (uint)wPacked, pArg = (uint)patch;
        uint innerArg = innerChannelFastest ? 1u : 0u;
        ulong totalArg = (ulong)channels * (ulong)hPacked * (ulong)wPacked * (ulong)(patch * patch);
        void** args = stackalloc void*[8];
        args[0] = &outArg;
        args[1] = &inArg;
        args[2] = &chArg;
        args[3] = &hpArg;
        args[4] = &wpArg;
        args[5] = &pArg;
        args[6] = &innerArg;
        args[7] = &totalArg;
        uint gridDim = (uint)((totalArg + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            _ditUnpatchifyF32, gridDim, 1, 1, BlockSize, 1, 1,
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

    /// <summary>Launches in-place interleaved (GPT-J) rotary embedding on x [B,L,numHeads,headDim];
    /// cos/sin [B,L,headDim]. Rotates adjacent pairs (2i,2i+1) by frequency i (not the split-half convention).</summary>
    public unsafe void LaunchRopeInterleaved(ulong x, ulong cos, ulong sin, int numHeads, int headDim, long totalVecs, nint stream)
    {
        ulong xArg = x, cosArg = cos, sinArg = sin;
        uint headsArg = (uint)numHeads, headDimArg = (uint)headDim;
        ulong vecsArg = (ulong)totalVecs;
        void** args = stackalloc void*[6];
        args[0] = &xArg;
        args[1] = &cosArg;
        args[2] = &sinArg;
        args[3] = &headsArg;
        args[4] = &headDimArg;
        args[5] = &vecsArg;
        long threads = totalVecs * (headDim / 2);
        uint gridDim = (uint)((threads + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(
            _lmRopeInterleavedF32, gridDim, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Graph-capture decode: split-half RoPE on a single decode-step Q/K tensor [1,numHeads,1,headDim],
    /// reading the rotation row from a precomputed full table via devicePos[1] (qOffset). Grid is fixed
    /// (position-independent) — same devicePos convention as LaunchKvAppend/LaunchFlashAttention.</summary>
    public unsafe void LaunchRopeDecodeSplitHalf(ulong x, ulong cosTable, ulong sinTable, int numHeads, int headDim, int rotaryDim, ulong devicePos, nint stream)
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
    public unsafe void LaunchOasisSplitHeads(ulong output, ulong qkv, int frames, int sp, int heads, int headDim, int part, bool temporal, nint stream)
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
    public unsafe void LaunchOasisAdaLn(ulong output, ulong input, ulong mod, int dim, int sp, int totalRows, int modStride, int shiftOff, int scaleOff, float eps, nint stream)
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

    /// <summary>Fused QKV split + per-head QK-RMSNorm (one block per (token, head); blockDim = headDim).</summary>
    public unsafe void LaunchQkvSplitNorm(bool f16, ulong q, ulong k, ulong v, ulong qkv, ulong qW, ulong kW,
        int tokens, int heads, int headDim, float eps, nint stream)
    {
        ulong qArg = q, kArg = k, vArg = v, qkvArg = qkv, qwArg = qW, kwArg = kW;
        uint tArg = (uint)tokens, hArg = (uint)heads, hdArg = (uint)headDim; float epsArg = eps;
        void** args = stackalloc void*[10];
        args[0] = &qArg; args[1] = &kArg; args[2] = &vArg; args[3] = &qkvArg; args[4] = &qwArg; args[5] = &kwArg;
        args[6] = &tArg; args[7] = &hArg; args[8] = &hdArg; args[9] = &epsArg;
        uint grid = (uint)((long)tokens * heads);
        uint sharedMem = (uint)headDim * sizeof(float);
        CudaDriverApi.cuLaunchKernel(f16 ? _ditQkvSplitNormF16 : _ditQkvSplitNormF32,
            grid, 1, 1, (uint)headDim, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
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
        ulong a0 = qkv, a1 = outp; uint u2 = (uint)tt, u3 = (uint)sp, u4 = (uint)heads, u5 = (uint)headDim, u6 = (uint)part, u7 = (uint)stride;
        void** args = stackalloc void*[8];
        args[0] = &a0; args[1] = &a1; args[2] = &u2; args[3] = &u3; args[4] = &u4; args[5] = &u5; args[6] = &u6; args[7] = &u7;
        long total = (long)sp * heads * tt * headDim; uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3SplitQkvTemporalF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule: [sp, heads, tt, headDim] back to token rows [tt·sp, streamDim].</summary>
    public unsafe void LaunchMg3MergeTemporal(ulong attn, ulong outp, int tt, int sp, int heads, int headDim, nint stream)
    {
        ulong a0 = attn, a1 = outp; uint u2 = (uint)tt, u3 = (uint)sp, u4 = (uint)heads, u5 = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &a0; args[1] = &a1; args[2] = &u2; args[3] = &u3; args[4] = &u4; args[5] = &u5;
        long total = (long)tt * sp * heads * headDim; uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3MergeTemporalF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule: interleaved rope on [sp, heads, tt, headDim]; cos/sin [gridRows, headDim].</summary>
    public unsafe void LaunchMg3RopeBatched(ulong x, ulong cos, ulong sin, int sp, int heads, int tt, int headDim, int gh, int gw, int broadcast, nint stream)
    {
        ulong a0 = x, a1 = cos, a2 = sin; uint u3 = (uint)sp, u4 = (uint)heads, u5 = (uint)tt, u6 = (uint)headDim, u7 = (uint)gh, u8 = (uint)gw, u9 = (uint)broadcast;
        void** args = stackalloc void*[10];
        args[0] = &a0; args[1] = &a1; args[2] = &a2; args[3] = &u3; args[4] = &u4; args[5] = &u5; args[6] = &u6; args[7] = &u7; args[8] = &u8; args[9] = &u9;
        long total = (long)sp * heads * tt * (headDim / 2); uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3RopeBatchedF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 ActionModule keyboard: expand K/V [tt, 2·streamDim] → k,v [sp, heads, tt, headDim].</summary>
    public unsafe void LaunchMg3KvExpand(ulong kv, ulong kOut, ulong vOut, int sp, int heads, int tt, int headDim, nint stream)
    {
        ulong a0 = kv, a1 = kOut, a2 = vOut; uint u3 = (uint)sp, u4 = (uint)heads, u5 = (uint)tt, u6 = (uint)headDim;
        void** args = stackalloc void*[7];
        args[0] = &a0; args[1] = &a1; args[2] = &a2; args[3] = &u3; args[4] = &u4; args[5] = &u5; args[6] = &u6;
        long total = (long)sp * heads * tt * headDim; uint grid = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mg3KvExpandF32, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>MG3 mouse stream: build MLP input [tt·sp, imgDim+winFloats] = [hidden | broadcast mouse window].</summary>
    public unsafe void LaunchMg3MouseMlpConcat(ulong hidden, ulong mouseWin, ulong outp, int tt, int sp, int imgDim, int winFloats, nint stream)
    {
        ulong a0 = hidden, a1 = mouseWin, a2 = outp; uint u3 = (uint)tt, u4 = (uint)sp, u5 = (uint)imgDim, u6 = (uint)winFloats;
        void** args = stackalloc void*[7];
        args[0] = &a0; args[1] = &a1; args[2] = &a2; args[3] = &u3; args[4] = &u4; args[5] = &u5; args[6] = &u6;
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
    public unsafe void LaunchLtx2SplitRope(ulong x, ulong cos, ulong sin, int S, int numHeads, int headDim, nint stream)
    {
        ulong xA = x, cA = cos, sA = sin;
        uint sArg = (uint)S, hArg = (uint)numHeads, dArg = (uint)headDim;
        void** args = stackalloc void*[6];
        args[0] = &xA; args[1] = &cA; args[2] = &sA; args[3] = &sArg; args[4] = &hArg; args[5] = &dArg;
        long total = (long)S * numHeads * (headDim / 2);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_ltx2SplitRopeF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Extracts temporal frame <paramref name="ti"/> of a 5D <c>[B,C,Tsrc,H,W]</c> tensor into a 4D
    /// <c>[B,C,H,W]</c> frame, on-device.</summary>
    public unsafe void LaunchWanVaeExtractFrame(ulong outp, ulong src, int ti, int B, int C, int Tsrc, int frameHW, nint stream)
    {
        ulong oA = outp, sA = src; uint tiA = (uint)ti, bA = (uint)B, cA = (uint)C, tsA = (uint)Tsrc, fA = (uint)frameHW;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &sA; args[2] = &tiA; args[3] = &bA; args[4] = &cA; args[5] = &tsA; args[6] = &fA;
        long total = (long)B * C * frameHW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeExtractFrame, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Writes a 4D <c>[B,C,H,W]</c> frame (plus optional per-channel bias) into temporal slot
    /// <paramref name="to"/> of a 5D <c>[B,C,Tout,H,W]</c> tensor, on-device. <paramref name="bias"/>=0 for none.</summary>
    public unsafe void LaunchWanVaeWriteFrame(ulong outp, ulong acc, ulong bias, int to, int B, int C, int Tout, int frameHW, nint stream)
    {
        ulong oA = outp, aA = acc, bsA = bias; uint toA = (uint)to, bA = (uint)B, cA = (uint)C, tA = (uint)Tout, fA = (uint)frameHW;
        void** args = stackalloc void*[8];
        args[0] = &oA; args[1] = &aA; args[2] = &bsA; args[3] = &toA; args[4] = &bA; args[5] = &cA; args[6] = &tA; args[7] = &fA;
        long total = (long)B * C * frameHW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeWriteFrame, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Builds the frame-major padded input [paddedT, cIn, H+2·padH, W+2·padW] (transpose + temporal
    /// zero/replicate-first pad + cache + spatial edge-replicate pad) for batched CausalConv3d.
    /// <paramref name="cache"/>=0 for none.</summary>
    public unsafe void LaunchWanVaeBuildPadded(ulong padded, ulong input, ulong cache, int paddedT, int cIn, int Tin, int cacheLen, int zeroPad,
        int H, int W, int padH, int padW, bool replicateFirst, nint stream)
    {
        ulong pA = padded, iA = input, cA = cache;
        uint ptA = (uint)paddedT, ciA = (uint)cIn, tiA = (uint)Tin, clA = (uint)cacheLen, zpA = (uint)zeroPad;
        uint hA = (uint)H, wA = (uint)W, phA = (uint)padH, pwA = (uint)padW, rfA = replicateFirst ? 1u : 0u;
        void** args = stackalloc void*[13];
        args[0] = &pA; args[1] = &iA; args[2] = &cA; args[3] = &ptA; args[4] = &ciA; args[5] = &tiA; args[6] = &clA; args[7] = &zpA;
        args[8] = &hA; args[9] = &wA; args[10] = &phA; args[11] = &pwA; args[12] = &rfA;
        long total = (long)paddedT * cIn * (H + 2L * padH) * (W + 2L * padW);
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeBuildPadded, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Fills output [cOut, tout, HW] with per-channel bias (or 0 when <paramref name="bias"/>=0).</summary>
    public unsafe void LaunchWanVaeRmsNormChannel(ulong outp, ulong x, ulong gamma, int c, long spatial, float eps, float sqrtC, long numPos, nint stream)
    {
        ulong oA = outp, xA = x, gA = gamma; int cA = c; long spA = spatial, npA = numPos; float eA = eps, scA = sqrtC;
        void** args = stackalloc void*[8];
        args[0] = &oA; args[1] = &xA; args[2] = &gA; args[3] = &cA; args[4] = &spA; args[5] = &eA; args[6] = &scA; args[7] = &npA;
        uint gridDim = (uint)((numPos + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeRmsNormChannel, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchWanVaeUnpatchify(ulong outp, ulong x, int b, int c, int t, int h, int w, int p, long numOut, nint stream)
    {
        ulong oA = outp, xA = x; int bA = b, cA = c, tA = t, hA = h, wA = w, pA = p; long nA = numOut;
        void** args = stackalloc void*[9];
        args[0] = &oA; args[1] = &xA; args[2] = &bA; args[3] = &cA; args[4] = &tA; args[5] = &hA; args[6] = &wA; args[7] = &pA; args[8] = &nA;
        uint gridDim = (uint)((numOut + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeUnpatchify, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchWanVaeSplitQkv(ulong q, ulong k, ulong v, ulong src, int bt, int c, int hw, long numEl, nint stream)
    {
        ulong qA = q, kA = k, vA = v, sA = src; int btA = bt, cA = c, hwA = hw; long nA = numEl;
        void** args = stackalloc void*[8];
        args[0] = &qA; args[1] = &kA; args[2] = &vA; args[3] = &sA; args[4] = &btA; args[5] = &cA; args[6] = &hwA; args[7] = &nA;
        uint gridDim = (uint)((numEl + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeSplitQkv, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchWanVaeTokensToFrame(ulong outp, ulong a, int bt, int c, int hw, long numEl, nint stream)
    {
        ulong oA = outp, aA = a; int btA = bt, cA = c, hwA = hw; long nA = numEl;
        void** args = stackalloc void*[6];
        args[0] = &oA; args[1] = &aA; args[2] = &btA; args[3] = &cA; args[4] = &hwA; args[5] = &nA;
        uint gridDim = (uint)((numEl + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeTokensToFrame, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public unsafe void LaunchWanVaeFillBias(ulong outp, ulong bias, int cOut, int tout, int HW, nint stream)
    {
        ulong oA = outp, bA = bias; uint coA = (uint)cOut, toA = (uint)tout, hwA = (uint)HW;
        void** args = stackalloc void*[5];
        args[0] = &oA; args[1] = &bA; args[2] = &coA; args[3] = &toA; args[4] = &hwA;
        long total = (long)cOut * tout * HW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeFillBias, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Temporal gather-sum: out[co][to][hw] += convDt[to·strideT+dt][co][hw]. out=[cOut,tout,HW],
    /// convDt=[paddedT,cOut,HW].</summary>
    public unsafe void LaunchWanVaeAccumulateTap(ulong outp, ulong convDt, int dt, int strideT, int cOut, int tout, int HW, nint stream)
    {
        ulong oA = outp, cA = convDt; uint dtA = (uint)dt, stA = (uint)strideT, coA = (uint)cOut, toA = (uint)tout, hwA = (uint)HW;
        void** args = stackalloc void*[7];
        args[0] = &oA; args[1] = &cA; args[2] = &dtA; args[3] = &stA; args[4] = &coA; args[5] = &toA; args[6] = &hwA;
        long total = (long)cOut * tout * HW;
        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_wanVaeAccumulateTap, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    // ── GeGlu Launches ──────────────────────────────────────────────────

    /// <summary>Launches GEGLU: splits input along last dim, applies GELU gate (F32).</summary>
    public void LaunchGeGlu(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_gegluF32, output, input, innerDim, outputElements, stream);

    /// <summary>Launches GEGLU with F16 I/O, F32 internal compute.</summary>
    public void LaunchGeGluF16(ulong output, ulong input, int innerDim, int outputElements, nint stream)
        => LaunchGeGluImpl(_gegluF16, output, input, innerDim, outputElements, stream);

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

    /// <summary>Launches BF16 to F32 cast (lossless — BF16 is the upper 16 bits of F32). Input is 2 bytes/element, output is 4 bytes/element.</summary>
    public void LaunchCastBf16ToF32(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castBf16ToF32, output, input, count, stream);

    /// <summary>Launches F32 to BF16 cast with round-to-nearest-even. Input is 4 bytes/element, output is 2 bytes/element.</summary>
    public void LaunchCastF32ToBf16(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_castF32ToBf16, output, input, count, stream);

    // ── GGUF Dequant Launches ────────────────────────────────────────────

    /// <summary>Launches Q8_0 → F16 dequant. <paramref name="elementCount"/> is the total element count (must be a multiple of 32). Internally launches one CUDA block per Q8_0 quant block (32 elements), 32 threads per CUDA block.</summary>
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

    /// <summary>Launches Q6_K → F16 dequant. Element count must be a multiple of 256. The Q6_K kernel uses 64 threads per CUDA block — each thread emits 4 elements at strides {0, +32, +64, +96} (2 halves × 32 l-values = 64 threads cover all 256 elements).</summary>
    public unsafe void LaunchDequantQ6_KToF16(ulong output, ulong input, int elementCount, nint stream)
    {
        if (elementCount % 256 != 0)
            throw new ArgumentException($"Q6_K element count must be a multiple of 256, got {elementCount}.");
        int superBlockCount = elementCount / 256;
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
        => LaunchMulMatVecImpl(_mulMatVecBf16F32, output, input, weight, bias, N, K, M, stream);

    /// <summary>Dense F16-weight × F32-activation GEMV (F32 accumulate) for small-M decode.</summary>
    public void LaunchMulMatVecF16F32(ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
        => LaunchMulMatVecImpl(_mulMatVecF16F32, output, input, weight, bias, N, K, M, stream);

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
        CudaDriverApi.cuLaunchKernel(_quantActQ8_1F32, (uint)nblocks, 1, 1, 32, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
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
        const uint WARPS_PER_BLOCK = 8;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(_mulMatVecQ4KQ8_1, gridX, (uint)M, 1, 32, WARPS_PER_BLOCK, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    private unsafe void LaunchMulMatVecImpl(nint func, ulong output, ulong input, ulong weight, ulong bias, int N, int K, int M, nint stream)
    {
        ulong outA = output, inA = input, wA = weight, bA = bias;
        int nA = N, kA = K, mA = M;
        void** args = stackalloc void*[7];
        args[0] = &outA; args[1] = &inA; args[2] = &wA; args[3] = &bA;
        args[4] = &nA; args[5] = &kA; args[6] = &mA;
        // One warp per output row, WARPS_PER_BLOCK rows per block.
        const uint WARPS_PER_BLOCK = 8;
        uint gridX = ((uint)N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        CudaDriverApi.cuLaunchKernel(
            func,
            gridX, (uint)M, 1,
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

    private void DisposeModules()
    {
        // Null-safe: if the constructor threw partway through, some modules will not
        // have been assigned. Finalizers must not raise â a null-ref here would crash
        // the process during GC after a CUDA backend init failure / Vulkan fallback.
        _elementwiseModule?.Dispose();
        _groupnormModule?.Dispose();
        _layernormModule?.Dispose();
        _spatialModule?.Dispose();
        _maxpool2dModule?.Dispose();
        _depthwiseConv2dModule?.Dispose();
        _msdaModule?.Dispose();
        _softmaxModule?.Dispose();
        _transposeModule?.Dispose();
        _gegluModule?.Dispose();
        _broadcastAddModule?.Dispose();
        _elementwiseF16Module?.Dispose();
        _groupnormF16Module?.Dispose();
        _layernormF16Module?.Dispose();
        _spatialF16Module?.Dispose();
        _softmaxF16Module?.Dispose();
        _transposeF16Module?.Dispose();
        _gegluF16Module?.Dispose();
        _broadcastAddF16Module?.Dispose();
        _elementwiseBf16Module?.Dispose();
        _groupnormBf16Module?.Dispose();
        _layernormBf16Module?.Dispose();
        _spatialBf16Module?.Dispose();
        _broadcastAddBf16Module?.Dispose();
        _groupnormSiluBf16Module?.Dispose();
        _groupnormSiluModule?.Dispose();
        _groupnormSiluF16Module?.Dispose();
        _castModule?.Dispose();
        _ditF32Module?.Dispose();
        _audioConvF32Module?.Dispose();
        _audioActF32Module?.Dispose();
        _audioAdain1dF32Module?.Dispose();
        _lmF32Module?.Dispose();
        _flashAttnF32Module?.Dispose();
        _castF8Module?.Dispose();
        _fp8QuantModule?.Dispose();
        _castBf16Module?.Dispose();
        _dequantQ8_0Module?.Dispose();
        _dequantQ4_KModule?.Dispose();
        _dequantQ5_KModule?.Dispose();
        _dequantQ6_KModule?.Dispose();
        _mulMatVecQ4KModule?.Dispose();
        _mulMatVecQ6KModule?.Dispose();
        _mulMatVecQ8_0Module?.Dispose();
        _mulMatVecF16Bf16Module?.Dispose();
        _mulMatVecQ5_0Module?.Dispose();
        _mulMatVecQ4_0Module?.Dispose();
        _mulMatVecQ5KModule?.Dispose();
        _quantActQ8_1Module?.Dispose();
        _mulMatVecQ4KQ8_1Module?.Dispose();
    }

    public void Dispose()
    {
        DisposeModules();
        GC.SuppressFinalize(this);
    }

    ~CudaKernels()
    {
        DisposeModules();
    }
}
