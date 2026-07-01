namespace HartsyInference.Cuda;

/// <summary>Loads PTX modules from disk and provides typed kernel launch methods. Function handles stored as nint fields for zero-alloc dispatch.</summary>
public sealed class CudaKernels : IDisposable
{
    // ── F32 Modules ──────────────────────────────────────────────────────
    private readonly CudaModule _elementwiseModule;
    private readonly CudaModule _groupnormModule;
    private readonly CudaModule _layernormModule;
    private readonly CudaModule _spatialModule;
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

    // ── Audio conv Module + handles (codec/TTS Conv1d + ConvTranspose1d, F32) ─
    private readonly CudaModule _audioConvF32Module;
    private readonly nint _conv1dF32;
    private readonly nint _convTranspose1dF32;

    // ── Audio activation Module + handles (Sigmoid / Elu / Snake, F32) ───
    private readonly CudaModule _audioActF32Module;
    private readonly nint _audioSigmoidF32;
    private readonly nint _audioEluF32;
    private readonly nint _audioLeakyReluF32;
    private readonly nint _audioSnakeF32;
    private readonly nint _audioSnakeBetaF32;

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
    private readonly CudaModule _flashAttnF32Module;
    private readonly nint _flashAttnF32;
    private readonly CudaModule _flashAttnF32SplitModule;
    private readonly nint _flashAttnF32Split;
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

    // ── Softmax function handles ─────────────────────────────────────────
    private readonly nint _softmaxF32;
    private readonly nint _softmaxF16;

    // ── Transpose/Permute function handles ───────────────────────────────
    private readonly nint _transpose2dF32;
    private readonly nint _permute0213F32;
    private readonly nint _transpose2dF16;
    private readonly nint _permute0213F16;
    private readonly nint _wanRopeInterleaved;
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
    private readonly nint _ditTanhF32;
    private readonly nint _ditRopeF32;
    private readonly nint _ditSliceLastDimF32;
    private readonly nint _ditRowScaleF32;
    private readonly nint _ditAddScalarF32;
    private readonly nint _ditLayerNormNoAffineF32;
    private readonly nint _ditIndexAddF32;
    private readonly nint _ditScatterRowsAfterF32;
    private readonly nint _ditSliceRowsF32;

    // ── FP8 Cast Modules + Handles ────────────────────────────────────────
    private readonly CudaModule _castF8Module;
    private readonly nint _castF8E4M3ToF16;

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

        _softmaxModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "softmax_f32.ptx"));
        _softmaxF32 = _softmaxModule.GetFunction("softmax_f32");

        _transposeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "transpose_f32.ptx"));
        _transpose2dF32 = _transposeModule.GetFunction("transpose_2d_f32");
        _permute0213F32 = _transposeModule.GetFunction("permute_0213_f32");

        _wanRopeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "wan_rope.ptx"));
        _wanRopeInterleaved = _wanRopeModule.GetFunction("wan_rope_interleaved");

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
        _ditTanhF32 = _ditF32Module.GetFunction("dit_tanh_f32");
        _ditRopeF32 = _ditF32Module.GetFunction("dit_rope_f32");
        _ditSliceLastDimF32 = _ditF32Module.GetFunction("dit_slice_lastdim_f32");
        _ditRowScaleF32 = _ditF32Module.GetFunction("dit_row_scale_f32");
        _ditAddScalarF32 = _ditF32Module.GetFunction("dit_add_scalar_f32");
        _ditLayerNormNoAffineF32 = _ditF32Module.GetFunction("dit_layernorm_noaffine_f32");
        _ditIndexAddF32 = _ditF32Module.GetFunction("dit_index_add_f32");
        _ditScatterRowsAfterF32 = _ditF32Module.GetFunction("dit_scatter_rows_after_f32");
        _ditSliceRowsF32 = _ditF32Module.GetFunction("dit_slice_rows_f32");

        // ── Audio conv (codec/TTS Conv1d + ConvTranspose1d, F32) ─────────
        _audioConvF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "conv1d_f32.ptx"));
        _conv1dF32 = _audioConvF32Module.GetFunction("conv1d_f32");
        _convTranspose1dF32 = _audioConvF32Module.GetFunction("conv_transpose1d_f32");

        _audioActF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "audio_activations_f32.ptx"));
        _audioSigmoidF32 = _audioActF32Module.GetFunction("audio_sigmoid_f32");
        _audioEluF32 = _audioActF32Module.GetFunction("audio_elu_f32");
        _audioLeakyReluF32 = _audioActF32Module.GetFunction("audio_leaky_relu_f32");
        _audioSnakeF32 = _audioActF32Module.GetFunction("audio_snake_f32");
        _audioSnakeBetaF32 = _audioActF32Module.GetFunction("audio_snake_beta_f32");

        _audioAdain1dF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "adain1d_f32.ptx"));
        _audioAdain1dF32 = _audioAdain1dF32Module.GetFunction("audio_adain1d_f32");

        // ── Language-model glue (F32) ────────────────────────────────────
        _lmF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "lm_f32.ptx"));
        _lmRepeatKvF32 = _lmF32Module.GetFunction("lm_repeat_kv_f32");
        _lmKvAppendF32 = _lmF32Module.GetFunction("lm_kv_append_f32");
        _lmGatherRowsF32 = _lmF32Module.GetFunction("lm_gather_rows_f32");
        _lmScatterAddWeightedRowsF32 = _lmF32Module.GetFunction("lm_scatter_add_weighted_rows_f32");
        _lmArgMaxLastDimF32 = _lmF32Module.GetFunction("lm_argmax_lastdim_f32");
        _flashAttnF32Module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "flash_attn_f32.ptx"));
        _flashAttnF32 = _flashAttnF32Module.GetFunction("lm_flash_attn_f32");
        _flashAttnF32SplitModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "flash_attn_f32_split.ptx"));
        _flashAttnF32Split = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f32_split");
        _flashAttnF32Combine = _flashAttnF32SplitModule.GetFunction("lm_flash_attn_f32_combine");

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

    /// <summary>Launches broadcast affine over the last dim: out[b,s,d] = in[b,s,d]*scale[b,d] + shift[b,d] (shift optional, pass 0 to skip).</summary>
    public void LaunchAffineBroadcastLastDim(ulong output, ulong input, ulong scale, ulong shift, int seqLen, int dim, long total, nint stream)
        => LaunchAffineBroadcastImpl(_ditAffineBroadcastF32, output, input, scale, shift, seqLen, dim, total, stream);

    /// <summary>Launches GQA K/V head repeat (block pattern): [B,Hkv,L,D] → [B,Hkv*group,L,D].</summary>
    public void LaunchRepeatKv(ulong output, ulong input, int kvHeads, int group, int seqLen, int headDim, long total, nint stream)
        => LaunchRepeatKvImpl(_lmRepeatKvF32, output, input, kvHeads, group, seqLen, headDim, total, stream);

    /// <summary>Launches FlashAttention (one block per (b, q-head, q-row); blockDim = headDim; shared mem =
    /// headDim floats). <paramref name="lk"/> is the K/V buffer seq stride; <paramref name="kvLen"/> the valid
    /// key count.</summary>
    public unsafe void LaunchFlashAttention(ulong outPtr, ulong q, ulong k, ulong v,
        int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap, ulong sink, int slidingWindow, ulong alibiSlopes, nint stream)
    {
        ulong outArg = outPtr, qArg = q, kArg = k, vArg = v, sinkArg = sink, alibiArg = alibiSlopes;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim;
        uint hkvArg = (uint)hkv, lkArg = (uint)lk, kvLenArg = (uint)kvLen, grpArg = (uint)kvGroup;
        int causalArg = causal ? 1 : 0, offArg = qOffset, swArg = slidingWindow;
        float scaleArg = scale, softcapArg = softcap;

        void** args = stackalloc void*[19];
        args[0] = &outArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bArg; args[5] = &hqArg; args[6] = &tqArg; args[7] = &dArg;
        args[8] = &hkvArg; args[9] = &lkArg; args[10] = &kvLenArg; args[11] = &grpArg;
        args[12] = &causalArg; args[13] = &offArg; args[14] = &scaleArg; args[15] = &softcapArg;
        args[16] = &sinkArg; args[17] = &swArg; args[18] = &alibiArg;

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

    /// <summary>Flash-decoding split phase (plain path: no sink/alibi/softcap/window). Launches
    /// <c>batch·hq·tq·splits</c> blocks; each computes the partial online-softmax state (m, l, Σp·V) for
    /// its key chunk into the scratch buffers. <paramref name="chunk"/> = ceil(kvLen / splits).</summary>
    public unsafe void LaunchFlashAttentionSplit(ulong partialM, ulong partialL, ulong partialAcc,
        ulong q, ulong k, ulong v, int batch, int hq, int tq, int headDim, int hkv, int lk, int kvLen,
        int kvGroup, bool causal, int qOffset, float scale, int splits, int chunk, nint stream)
    {
        ulong pmArg = partialM, plArg = partialL, paArg = partialAcc, qArg = q, kArg = k, vArg = v;
        uint bArg = (uint)batch, hqArg = (uint)hq, tqArg = (uint)tq, dArg = (uint)headDim;
        uint hkvArg = (uint)hkv, lkArg = (uint)lk, kvLenArg = (uint)kvLen, grpArg = (uint)kvGroup;
        int causalArg = causal ? 1 : 0, offArg = qOffset;
        float scaleArg = scale;
        uint gArg = (uint)splits, chunkArg = (uint)chunk;

        void** args = stackalloc void*[19];
        args[0] = &pmArg; args[1] = &plArg; args[2] = &paArg; args[3] = &qArg; args[4] = &kArg; args[5] = &vArg;
        args[6] = &bArg; args[7] = &hqArg; args[8] = &tqArg; args[9] = &dArg;
        args[10] = &hkvArg; args[11] = &lkArg; args[12] = &kvLenArg; args[13] = &grpArg;
        args[14] = &causalArg; args[15] = &offArg; args[16] = &scaleArg;
        args[17] = &gArg; args[18] = &chunkArg;

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
    public unsafe void LaunchKvAppend(ulong buffer, ulong newKv, int heads, int maxSeq, int tNew, int headDim, int offset, nint stream)
    {
        ulong bufArg = buffer, newArg = newKv;
        uint hArg = (uint)heads, maxArg = (uint)maxSeq, tArg = (uint)tNew, dArg = (uint)headDim, offArg = (uint)offset;
        ulong total = (ulong)heads * (ulong)tNew * (ulong)headDim;
        ulong totalArg = total;

        void** args = stackalloc void*[8];
        args[0] = &bufArg; args[1] = &newArg; args[2] = &hArg; args[3] = &maxArg;
        args[4] = &tArg; args[5] = &dArg; args[6] = &offArg; args[7] = &totalArg;

        uint gridDim = (uint)((total + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_lmKvAppendF32, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
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

    /// <summary>Launches AdaLN modulation split: proj[B,4D] → (1+scale_msa, tanh(gate_msa), 1+scale_mlp, tanh(gate_mlp)), each [B,D].</summary>
    public void LaunchModulation4(ulong scaleMsa, ulong gateMsa, ulong scaleMlp, ulong gateMlp, ulong proj, int dim, int batch, nint stream)
        => LaunchModulation4Impl(_ditModulation4F32, scaleMsa, gateMsa, scaleMlp, gateMlp, proj, dim, batch, stream);

    /// <summary>Launches CFG combine + Euler step in-place on z: z[i] += delta*(guidance*pos[i] + (1-guidance)*neg[i]).</summary>
    public void LaunchCfgEuler(ulong z, ulong pos, ulong neg, float guidance, float delta, int count, nint stream)
        => LaunchCfgEulerImpl(_ditCfgEulerF32, z, pos, neg, guidance, delta, count, stream);

    /// <summary>Launches tanh: output[i] = tanh(input[i]) (F32).</summary>
    public void LaunchTanh(ulong output, ulong input, int count, nint stream)
        => LaunchUnaryImpl(_ditTanhF32, output, input, count, stream);

    /// <summary>Launches in-place rotary embedding on x [B,L,numHeads,headDim]; cos/sin [B,L,headDim].</summary>
    public void LaunchRope(ulong x, ulong cos, ulong sin, int numHeads, int headDim, long totalVecs, nint stream, int rotaryDim = 0)
        => LaunchRopeImpl(_ditRopeF32, x, cos, sin, numHeads, headDim, totalVecs, rotaryDim, stream);

    /// <summary>Launches last-dim slice: out[row,d] = in[row, offset+d], in row stride = inDim.</summary>
    public void LaunchSliceLastDim(ulong output, ulong input, int outDim, int inDim, int offset, long total, nint stream)
        => LaunchSliceLastDimImpl(_ditSliceLastDimF32, output, input, outDim, inDim, offset, total, stream);

    /// <summary>Launches per-row scalar multiply: out[row,c] = in[row,c] * rowScale[row] (token masking).</summary>
    public void LaunchRowScale(ulong output, ulong input, ulong rowScale, int channels, long total, nint stream)
        => LaunchRowScaleImpl(_ditRowScaleF32, output, input, rowScale, channels, total, stream);

    /// <summary>Launches add-scalar: out[i] = in[i] + c (F32).</summary>
    public void LaunchAddScalar(ulong output, ulong input, float c, int count, nint stream)
        => LaunchScaleImpl(_ditAddScalarF32, output, input, c, count, stream);

    /// <summary>Launches non-affine LayerNorm: per-row zero-mean unit-variance, no scale/bias.</summary>
    public void LaunchLayerNormNoAffine(ulong output, ulong input, int dim, int totalRows, float eps, nint stream)
        => LaunchLayerNormNoAffineImpl(_ditLayerNormNoAffineF32, output, input, dim, totalRows, eps, stream);

    /// <summary>Launches index-add of embedding rows in-place: h[row,d] += table[indices[row], d].</summary>
    public void LaunchIndexAdd(ulong h, ulong table, ulong indices, int dim, long total, nint stream)
        => LaunchIndexAddImpl(_ditIndexAddF32, h, table, indices, dim, total, stream);

    /// <summary>Launches scatter-rows-after: output = [zeros(headRows), input] along the row axis.</summary>
    public void LaunchScatterRowsAfter(ulong output, ulong input, int headRows, int dim, long total, nint stream)
        => LaunchScatterRowsAfterImpl(_ditScatterRowsAfterF32, output, input, headRows, dim, total, stream);

    /// <summary>Launches contiguous row-block slice: output[i] = input[elemOffset + i].</summary>
    public void LaunchSliceRows(ulong output, ulong input, long elemOffset, long total, nint stream)
        => LaunchSliceRowsImpl(_ditSliceRowsF32, output, input, elemOffset, total, stream);

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

    /// <summary>Builds the frame-major padded input [paddedT, cIn, H, W] (transpose + causal zero-pad + cache) for
    /// batched CausalConv3d. <paramref name="cache"/>=0 for none.</summary>
    public unsafe void LaunchWanVaeBuildPadded(ulong padded, ulong input, ulong cache, int paddedT, int cIn, int Tin, int cacheLen, int zeroPad, int HW, nint stream)
    {
        ulong pA = padded, iA = input, cA = cache; uint ptA = (uint)paddedT, ciA = (uint)cIn, tiA = (uint)Tin, clA = (uint)cacheLen, zpA = (uint)zeroPad, hwA = (uint)HW;
        void** args = stackalloc void*[9];
        args[0] = &pA; args[1] = &iA; args[2] = &cA; args[3] = &ptA; args[4] = &ciA; args[5] = &tiA; args[6] = &clA; args[7] = &zpA; args[8] = &hwA;
        long total = (long)paddedT * cIn * HW;
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
        _castBf16Module?.Dispose();
        _dequantQ8_0Module?.Dispose();
        _dequantQ4_KModule?.Dispose();
        _dequantQ5_KModule?.Dispose();
        _dequantQ6_KModule?.Dispose();
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
