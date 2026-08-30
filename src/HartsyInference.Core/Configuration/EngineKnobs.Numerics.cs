namespace HartsyInference.Core.Configuration;

/// <summary>Kernel selection, precision, fusion and solver math. These change what the engine computes.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Routes audio conv1d through cuDNN (1D-as-2D, TF32) instead of the direct kernel; 0 disables.</summary>
    public static readonly Knob<bool> AudioConvCudnn =
        Bool("numerics.audioConvCudnn", "HARTSY_AUDIO_CONV_CUDNN", true, KnobScope.Runtime, KnobDomain.Numerics, "Routes audio conv1d through cuDNN (1D-as-2D, TF32) instead of the direct kernel; 0 disables.");

    /// <summary>Kill-switch for AuraFlow's device-resident packed-token denoise loop; 0 reverts to the host reference loop.</summary>
    public static readonly Knob<bool> AuraflowPacked =
        Bool("numerics.auraflowPacked", "HARTSY_AURAFLOW_PACKED", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for AuraFlow's device-resident packed-token denoise loop; 0 reverts to the host reference loop.");

    /// <summary>Uses the fused BF16/F16 decode GEMV kernel for small-m F32-activation matmuls instead of cuBLAS GemmEx.</summary>
    public static readonly Knob<bool> Bf16Gemv =
        Bool("numerics.bf16Gemv", "HARTSY_BF16_GEMV", true, KnobScope.Runtime, KnobDomain.Numerics, "Uses the fused BF16/F16 decode GEMV kernel for small-m F32-activation matmuls instead of cuBLAS GemmEx.");

    /// <summary>Keeps Chroma's fused BFL qkv weights at conversion so blocks run one qkv GEMM plus a fused split-norm.</summary>
    public static readonly Knob<bool> ChromaFusedQkv =
        Bool("numerics.chromaFusedQkv", "HARTSY_CHROMA_FUSED_QKV", false, KnobScope.Construction, KnobDomain.Numerics, "Keeps Chroma's fused BFL qkv weights at conversion so blocks run one qkv GEMM plus a fused split-norm.");

    /// <summary>Selects the f16-staged wide fused ConvRot+quant kernel over the bit-identical rotate-then-quant pair.</summary>
    public static readonly Knob<bool> ConvrotWide =
        Bool("numerics.convrotWide", "HARTSY_CONVROT_WIDE", false, KnobScope.Runtime, KnobDomain.Numerics, "Selects the f16-staged wide fused ConvRot+quant kernel over the bit-identical rotate-then-quant pair.");

    /// <summary>Kill-switch for cuDNN convolution forward; 0 routes every 2D conv back through im2col+cuBLAS.</summary>
    public static readonly Knob<bool> ConvCudnn =
        Bool("numerics.convCudnn", "HARTSY_CONV_CUDNN", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for cuDNN convolution forward; 0 routes every 2D conv back through im2col+cuBLAS.");

    /// <summary>Runs CSM's CFG cond+uncond as one B=2 batched backbone step; 0 restores the two separate stream steps.</summary>
    public static readonly Knob<bool> CsmCfgBatch =
        Bool("numerics.csmCfgBatch", "HARTSY_CSM_CFG_BATCH", true, KnobScope.Runtime, KnobDomain.Numerics, "Runs CSM's CFG cond+uncond as one B=2 batched backbone step; 0 restores the two separate stream steps.");

    /// <summary>Replays the CSM batched B=2 CFG backbone step as one captured CUDA graph instead of the eager batched step.</summary>
    public static readonly Knob<bool> CsmCfgGraph =
        Bool("numerics.csmCfgGraph", "HARTSY_CSM_CFG_GRAPH", true, KnobScope.Runtime, KnobDomain.Numerics, "Replays the CSM batched B=2 CFG backbone step as one captured CUDA graph instead of the eager batched step.");

    /// <summary>Kill-switch for CUDA-graph replay of CSM's steady-state one-row backbone decode step; 0 keeps it eager.</summary>
    public static readonly Knob<bool> CsmGraph =
        Bool("numerics.csmGraph", "HARTSY_CSM_GRAPH", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for CUDA-graph replay of CSM's steady-state one-row backbone decode step; 0 keeps it eager.");

    /// <summary>Activation dtype for opted-in DiT block/attention hot paths; 0 forces F32 instead of the default F16.</summary>
    public static readonly Knob<bool> DitF16 =
        Bool("numerics.ditF16", "HARTSY_DIT_F16", true, KnobScope.Runtime, KnobDomain.Numerics, "Activation dtype for opted-in DiT block/attention hot paths; 0 forces F32 instead of the default F16.");

    /// <summary>CUDA-graph capture of a DiT denoise step: 0 kills both tiers, 1 forces both, unset keeps per-arch defaults.</summary>
    public static readonly Knob<bool> DitGraph =
        Bool("numerics.ditGraph", "HARTSY_DIT_GRAPH", false, KnobScope.Runtime, KnobDomain.Numerics, "CUDA-graph capture of a DiT denoise step: 0 kills both tiers, 1 forces both, unset keeps per-arch defaults.");

    /// <summary>Kill-switch for the dp4a int8-activation decode GEMV kernels over Q4_K/Q6_K/Q8_0 LLM weights.</summary>
    public static readonly Knob<bool> Dp4aOn =
        Bool("numerics.dp4aOn", "HARTSY_DP4A_ON", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for the dp4a int8-activation decode GEMV kernels over Q4_K/Q6_K/Q8_0 LLM weights.");

    /// <summary>Uses cuBLASLt's fused bias epilogue for biased Linear instead of a separate BiasAdd kernel; 0 disables.</summary>
    public static readonly Knob<bool> EpilogueFusion =
        Bool("numerics.epilogueFusion", "HARTSY_EPILOGUE_FUSION", true, KnobScope.Runtime, KnobDomain.Numerics, "Uses cuBLASLt's fused bias epilogue for biased Linear instead of a separate BiasAdd kernel; 0 disables.");

    /// <summary>Enables the experimental F5-TTS DiT CUDA-graph capture; replay currently throws ILLEGAL_ADDRESS, keep off.</summary>
    public static readonly Knob<bool> F5Graph =
        Bool("numerics.f5Graph", "HARTSY_F5_GRAPH", false, KnobScope.Runtime, KnobDomain.Numerics, "Enables the experimental F5-TTS DiT CUDA-graph capture; replay currently throws ILLEGAL_ADDRESS, keep off.");

    /// <summary>Quantizes fp8 activations with the checkpoint's .input_scale rather than a per-call absmax; changes numerics.</summary>
    public static readonly Knob<bool> Fp8StaticInputScale =
        Bool("numerics.fp8StaticInputScale", "HARTSY_FP8_STATIC_INPUT_SCALE", true, KnobScope.Runtime, KnobDomain.Numerics, "Quantizes fp8 activations with the checkpoint's .input_scale rather than a per-call absmax; changes numerics.");

    /// <summary>At Ideogram-4 checkpoint conversion, fuses each block's SwiGLU w1/w3 into one w13 weight (fp8 pairs requantized).</summary>
    public static readonly Knob<bool> FusedFfn =
        Bool("numerics.fusedFfn", "HARTSY_FUSED_FFN", false, KnobScope.Construction, KnobDomain.Numerics, "At Ideogram-4 checkpoint conversion, fuses each block's SwiGLU w1/w3 into one w13 weight (fp8 pairs requantized).");

    /// <summary>Opts LLM decode into the captured CUDA-graph step (greedy, non-JSON, dense GQA only) over the eager loop.</summary>
    public static readonly Knob<bool> GraphDecode =
        Bool("numerics.graphDecode", "HARTSY_GRAPH_DECODE", false, KnobScope.Runtime, KnobDomain.Numerics, "Opts LLM decode into the captured CUDA-graph step (greedy, non-JSON, dense GQA only) over the eager loop.");

    /// <summary>Kill-switch for grouped resident-int8 Linears sharing one activation rotate+quant pass across projections.</summary>
    public static readonly Knob<bool> GroupedLinear =
        Bool("numerics.groupedLinear", "HARTSY_GROUPED_LINEAR", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for grouped resident-int8 Linears sharing one activation rotate+quant pass across projections.");

    /// <summary>Kill-switch for the fused int8 GEMM+dequant mma kernel on the shapes it is narrowed to (m >= 1024).</summary>
    public static readonly Knob<bool> Int8FusedMma =
        Bool("numerics.int8FusedMma", "HARTSY_INT8_FUSED_MMA", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for the fused int8 GEMM+dequant mma kernel on the shapes it is narrowed to (m >= 1024).");

    /// <summary>A/B layout control for the fused int8 mma GEMM: 0 picks the padded kernel (not the feature kill switch).</summary>
    public static readonly Knob<bool> Int8MmaSwizzle =
        Bool("numerics.int8MmaSwizzle", "HARTSY_INT8_MMA_SWIZZLE", true, KnobScope.Runtime, KnobDomain.Numerics, "A/B layout control for the fused int8 mma GEMM: 0 picks the padded kernel (not the feature kill switch).");

    /// <summary>Widens the fused INT8 MMA GEMM admission bound from n<=2k to n<=4k, admitting LTX-2.5's ffn_up shape.</summary>
    public static readonly Knob<bool> Int8MmaWideGate =
        Bool("numerics.int8MmaWideGate", "HARTSY_INT8_MMA_WIDE_GATE", false, KnobScope.Runtime, KnobDomain.Numerics, "Widens the fused INT8 MMA GEMM admission bound from n<=2k to n<=4k, admitting LTX-2.5's ffn_up shape.");

    /// <summary>Uses the query-tiled LTX-2.5 na3d kernel (online softmax over the tile's union window) over the per-query one.</summary>
    public static readonly Knob<bool> Ltx25Na3dTiled =
        Bool("numerics.ltx25Na3dTiled", "HARTSY_LTX25_NA3D_TILED", true, KnobScope.Runtime, KnobDomain.Numerics, "Uses the query-tiled LTX-2.5 na3d kernel (online softmax over the tile's union window) over the per-query one.");

    /// <summary>Builds LTX-2.5's diffusion video decoder instead of the ~40x faster conv decoder when the checkpoint has both.</summary>
    public static readonly Knob<bool> Ltx2DiffusionVae =
        Bool("numerics.ltx2DiffusionVae", "HARTSY_LTX2_DIFFUSION_VAE", false, KnobScope.Construction, KnobDomain.Numerics, "Builds LTX-2.5's diffusion video decoder instead of the ~40x faster conv decoder when the checkpoint has both.");

    /// <summary>Kill switch for folding LTX-2's per-head gate into the int8 activation rotate+quant pass (bit-identical).</summary>
    public static readonly Knob<bool> Ltx2Gatefuse =
        Bool("numerics.ltx2Gatefuse", "HARTSY_LTX2_GATEFUSE", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill switch for folding LTX-2's per-head gate into the int8 activation rotate+quant pass (bit-identical).");

    /// <summary>Stores the LTX-2 rope cos/sin tables as F16 instead of F32; faster fused QK kernel, SSIM ~0.9956 change.</summary>
    public static readonly Knob<bool> Ltx2Ropef16 =
        Bool("numerics.ltx2Ropef16", "HARTSY_LTX2_ROPEF16", false, KnobScope.Runtime, KnobDomain.Numerics, "Stores the LTX-2 rope cos/sin tables as F16 instead of F32; faster fused QK kernel, SSIM ~0.9956 change.");

    /// <summary>Lets a token-major SDPA call permute into head-major so it can take the SageAttention kernel.</summary>
    public static readonly Knob<bool> Ltx2SageTokenmajor =
        Bool("numerics.ltx2SageTokenmajor", "HARTSY_LTX2_SAGE_TOKENMAJOR", false, KnobScope.Runtime, KnobDomain.Numerics, "Lets a token-major SDPA call permute into head-major so it can take the SageAttention kernel.");

    /// <summary>Kill switch for LTX-2's token-major attention route, falling back to the head-major layout.</summary>
    public static readonly Knob<bool> Ltx2Tokenmajor =
        Bool("numerics.ltx2Tokenmajor", "HARTSY_LTX2_TOKENMAJOR", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill switch for LTX-2's token-major attention route, falling back to the head-major layout.");

    /// <summary>Runs MiniMax-Music3's AR CFG cond+uncond rows as one batched step; 0 restores separate passes.</summary>
    public static readonly Knob<bool> Mm3CfgBatch =
        Bool("numerics.mm3CfgBatch", "HARTSY_MM3_CFG_BATCH", true, KnobScope.Runtime, KnobDomain.Numerics, "Runs MiniMax-Music3's AR CFG cond+uncond rows as one batched step; 0 restores separate passes.");

    /// <summary>Also quantizes the MiniMax-Music3 depth decoder; off leaves it at checkpoint precision.</summary>
    public static readonly Knob<bool> Mm3DepthQuant =
        Bool("numerics.mm3DepthQuant", "HARTSY_MM3_DEPTH_QUANT", true, KnobScope.Construction, KnobDomain.Numerics, "Also quantizes the MiniMax-Music3 depth decoder; off leaves it at checkpoint precision.");

    /// <summary>Kill-switch for the modulate/affine-broadcast kernel emitting fp8 e4m3 activations straight into the next GEMM.</summary>
    public static readonly Knob<bool> ModulateEmitFp8 =
        Bool("numerics.modulateEmitFp8", "HARTSY_MODULATE_EMIT_FP8", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for the modulate/affine-broadcast kernel emitting fp8 e4m3 activations straight into the next GEMM.");

    /// <summary>Presence-only: setting it to ANY value (even 0) disables CUDA-graph decode in the MusicGen AR loop.</summary>
    public static readonly Knob<string?> MusicgenGraphOff =
        Str("numerics.musicgenGraphOff", "HARTSY_MUSICGEN_GRAPH_OFF", null, KnobScope.Runtime, KnobDomain.Numerics, "Presence-only: setting it to ANY value (even 0) disables CUDA-graph decode in the MusicGen AR loop.");

    /// <summary>Kill-switch for the fused per-head QK-norm + rope + KV-scatter decode epilogue (Qwen3/Gemma-3 shapes).</summary>
    public static readonly Knob<bool> QknormScatter =
        Bool("numerics.qknormScatter", "HARTSY_QKNORM_SCATTER", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for the fused per-head QK-norm + rope + KV-scatter decode epilogue (Qwen3/Gemma-3 shapes).");

    /// <summary>0 disables the partial Q+K weight concat used when V's dtype differs; the all-match QKV fuse is unaffected.</summary>
    public static readonly Knob<bool> QkFusion =
        Bool("numerics.qkFusion", "HARTSY_QK_FUSION", true, KnobScope.Construction, KnobDomain.Numerics, "0 disables the partial Q+K weight concat used when V's dtype differs; the all-match QKV fuse is unaffected.");

    /// <summary>Kill switch for the fused [q|k] GEMV plus rope-scatter-V decode kernel; off restores the composed ops.</summary>
    public static readonly Knob<bool> QkScatter =
        Bool("numerics.qkScatter", "HARTSY_QK_SCATTER", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill switch for the fused [q|k] GEMV plus rope-scatter-V decode kernel; off restores the composed ops.");

    /// <summary>Kill-switch for norm kernels emitting a Q8_1 sidecar so the dp4a GEMV skips a separate quantize pass.</summary>
    public static readonly Knob<bool> QuantAtProducer =
        Bool("numerics.quantAtProducer", "HARTSY_QUANT_AT_PRODUCER", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for norm kernels emitting a Q8_1 sidecar so the dp4a GEMV skips a separate quantize pass.");

    /// <summary>Requantizes Chroma Radiance DiT blocks to fp8 during checkpoint conversion; 0 keeps the wide BF16 weights.</summary>
    public static readonly Knob<bool> RadianceFp8 =
        Bool("numerics.radianceFp8", "HARTSY_RADIANCE_FP8", true, KnobScope.Construction, KnobDomain.Numerics, "Requantizes Chroma Radiance DiT blocks to fp8 during checkpoint conversion; 0 keeps the wide BF16 weights.");

    /// <summary>Uses the division-free head-major RoPE kernel instead of the older one; bit-identical.</summary>
    public static readonly Knob<bool> RopeV2 =
        Bool("numerics.ropeV2", "HARTSY_ROPE_V2", true, KnobScope.Runtime, KnobDomain.Numerics, "Uses the division-free head-major RoPE kernel instead of the older one; bit-identical.");

    /// <summary>Feeds the conditional audio to Wan-S2V's uncond branch so CFG steers text adherence only; changes output.</summary>
    public static readonly Knob<bool> S2vTextCfg =
        Bool("numerics.s2vTextCfg", "HARTSY_S2V_TEXT_CFG", false, KnobScope.Runtime, KnobDomain.Numerics, "Feeds the conditional audio to Wan-S2V's uncond branch so CFG steers text adherence only; changes output.");

    /// <summary>Selects the INT8 SageAttention kernel for attention; =0 disables it (any non-0 value leaves it on).</summary>
    public static readonly Knob<bool> SageAttn =
        Bool("numerics.sageAttn", "HARTSY_SAGE_ATTN", true, KnobScope.Runtime, KnobDomain.Numerics, "Selects the INT8 SageAttention kernel for attention; =0 disables it (any non-0 value leaves it on).");

    /// <summary>Literal f16acc picks SageAttention v1's F16-accumulate PV variant; any other value silently means off.</summary>
    public static readonly Knob<string?> SagePv =
        Str("numerics.sagePv", "HARTSY_SAGE_PV", null, KnobScope.Runtime, KnobDomain.Numerics, "Literal f16acc picks SageAttention v1's F16-accumulate PV variant; any other value silently means off.");

    /// <summary>Forces the older v0 SageAttention flash kernel instead of the register-resident v1 path.</summary>
    public static readonly Knob<bool> SageV0 =
        Bool("numerics.sageV0", "HARTSY_SAGE_V0", false, KnobScope.Runtime, KnobDomain.Numerics, "Forces the older v0 SageAttention flash kernel instead of the register-resident v1 path.");

    /// <summary>Kill-switch for collapsing post-attn norm + residual add + pre-FFN norm into one LLM decode kernel.</summary>
    public static readonly Knob<bool> SandwichFusion =
        Bool("numerics.sandwichFusion", "HARTSY_SANDWICH_FUSION", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for collapsing post-attn norm + residual add + pre-FFN norm into one LLM decode kernel.");

    /// <summary>Routes scaled-dot-product attention through cuDNN's fused flash engine; 0 falls back to materialized paths.</summary>
    public static readonly Knob<bool> SdpaCudnn =
        Bool("numerics.sdpaCudnn", "HARTSY_SDPA_CUDNN", true, KnobScope.Runtime, KnobDomain.Numerics, "Routes scaled-dot-product attention through cuDNN's fused flash engine; 0 falls back to materialized paths.");

    /// <summary>Builds the SeedVR2 VAE with F32 activations instead of the default CUDA BF16 (which halves the peak).</summary>
    public static readonly Knob<bool> Seedvr2VaeF32 =
        Bool("numerics.seedvr2VaeF32", "HARTSY_SEEDVR2_VAE_F32", false, KnobScope.Construction, KnobDomain.Numerics, "Builds the SeedVR2 VAE with F32 activations instead of the default CUDA BF16 (which halves the peak).");

    /// <summary>Default for prompt-lookup speculative decoding when the request omits it; greedy non-JSON requests only.</summary>
    public static readonly Knob<bool> SpecDecode =
        Bool("numerics.specDecode", "HARTSY_SPEC_DECODE", false, KnobScope.Runtime, KnobDomain.Numerics, "Default for prompt-lookup speculative decoding when the request omits it; greedy non-JSON requests only.");

    /// <summary>Uses the row-parallel SSM delta-rule recurrence kernel; 0 falls back to the legacy block-per-row kernel.</summary>
    public static readonly Knob<bool> SsmDeltaV2 =
        Bool("numerics.ssmDeltaV2", "HARTSY_SSM_DELTA_V2", true, KnobScope.Runtime, KnobDomain.Numerics, "Uses the row-parallel SSM delta-rule recurrence kernel; 0 falls back to the legacy block-per-row kernel.");

    /// <summary>Uses the row-parallel warp-per-row SSM delta-rule step kernel over the legacy block-per-head one.</summary>
    public static readonly Knob<bool> SsmDeltaWarprow =
        Bool("numerics.ssmDeltaWarprow", "HARTSY_SSM_DELTA_WARPROW", true, KnobScope.Runtime, KnobDomain.Numerics, "Uses the row-parallel warp-per-row SSM delta-rule step kernel over the legacy block-per-head one.");

    /// <summary>Kill-switch for Qwen3.5's on-device single-token step in both the attention and gated-DeltaNet layers.</summary>
    public static readonly Knob<bool> SsmDeviceStep =
        Bool("numerics.ssmDeviceStep", "HARTSY_SSM_DEVICE_STEP", true, KnobScope.Runtime, KnobDomain.Numerics, "Kill-switch for Qwen3.5's on-device single-token step in both the attention and gated-DeltaNet layers.");

    /// <summary>0 disables CUDA-graph capture/replay of the SSM (Qwen3.5-class) greedy decode step.</summary>
    public static readonly Knob<bool> SsmGraph =
        Bool("numerics.ssmGraph", "HARTSY_SSM_GRAPH", true, KnobScope.Runtime, KnobDomain.Numerics, "0 disables CUDA-graph capture/replay of the SSM (Qwen3.5-class) greedy decode step.");

    /// <summary>Casts VAE weights to F32 at recipe load instead of the default BF16 on BF16-capable backends.</summary>
    public static readonly Knob<bool> VaeF32 =
        Bool("numerics.vaeF32", "HARTSY_VAE_F32", false, KnobScope.Construction, KnobDomain.Numerics, "Casts VAE weights to F32 at recipe load instead of the default BF16 on BF16-capable backends.");

    /// <summary>Runs large-M float Linears as per-channel-int8 weight x per-row-int8 activation IMMA GEMM; lossy.</summary>
    public static readonly Knob<bool> W8a8 =
        Bool("numerics.w8a8", "HARTSY_W8A8", false, KnobScope.Runtime, KnobDomain.Numerics, "Runs large-M float Linears as per-channel-int8 weight x per-row-int8 activation IMMA GEMM; lossy.");

}
