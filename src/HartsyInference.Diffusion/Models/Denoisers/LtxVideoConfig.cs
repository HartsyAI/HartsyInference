namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for LTX-Video (Lightricks, Apache-2.0) — a fast DiT text-to-video model. Verified against diffusers <c>LTXVideoTransformer3DModel</c>. The DiT is single-stream: self-attention + 3D RoPE on VAE-latent tokens, cross-attention to a frozen T5-XXL, AdaLN-Single timestep conditioning. See <c>docs/Research/LTX_VIDEO_ARCHITECTURE.md</c>.</summary>
public sealed record LtxVideoConfig
{
    /// <summary>Transformer I/O channels (= VAE latent channels; patch_size 1 → no extra patchify).</summary>
    public int InChannels { get; init; } = 128;

    public int OutChannels { get; init; } = 128;

    public int NumHeads { get; init; } = 32;

    public int HeadDim { get; init; } = 64;

    public int InnerDim => NumHeads * HeadDim;

    /// <summary>Cross-attention dim (T5 features after caption projection).</summary>
    public int CrossAttentionDim { get; init; } = 2048;

    /// <summary>Number of DiT blocks.</summary>
    public int NumLayers { get; init; } = 28;

    /// <summary>T5-XXL feature width fed to the caption projection.</summary>
    public int CaptionChannels { get; init; } = 4096;

    /// <summary>RMSNorm epsilon for the no-affine pre-norms (norm1/norm2/norm_out).</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>QK-RMSNorm epsilon (across-heads, affine).</summary>
    public float QkNormEps { get; init; } = 1e-5f;

    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>RoPE base frame count (normalizer for the temporal axis).</summary>
    public int RopeBaseNumFrames { get; init; } = 20;

    /// <summary>RoPE base height (normalizer for the H axis).</summary>
    public int RopeBaseHeight { get; init; } = 2048;

    /// <summary>RoPE base width (normalizer for the W axis).</summary>
    public int RopeBaseWidth { get; init; } = 2048;

    // ── VAE / spatial-temporal compression (for the pipeline) ──
    public int VaeSpatialCompression { get; init; } = 32;

    public int VaeTemporalCompression { get; init; } = 8;

    /// <summary>Default flow-match timestep shift.</summary>
    public float TimestepShift { get; init; } = 1.0f;

    public int NumInferenceSteps { get; init; } = 50;

    /// <summary>Default CFG guidance scale.</summary>
    public float GuidanceScale { get; init; } = 3.0f;

    /// <summary>Whether the VAE decoder is timestep-conditioned (0.9.1+ and all 13B variants). When true the
    /// pipeline must decode at <see cref="DecodeTimestep"/> / <see cref="DecodeNoiseScale"/>; the 0.9.0 base VAE
    /// is not timestep-conditioned.</summary>
    public bool VaeTimestepConditioned { get; init; }

    /// <summary>VAE decode timestep (0.9.1+/13B). Reference default 0.05.</summary>
    public float DecodeTimestep { get; init; } = 0.05f;

    /// <summary>VAE decode noise scale (0.9.1+/13B). Reference default 0.025.</summary>
    public float DecodeNoiseScale { get; init; } = 0.025f;

    /// <summary>The base LTX-Video 0.9.0 preset (2B): 28 layers, head_dim 64, cross-attn 2048, non-timestep VAE.</summary>
    public static LtxVideoConfig V09 => new();

    /// <summary>LTX-Video 0.9.5 preset (2B): same 28-layer transformer as 0.9, but the VAE is timestep-conditioned
    /// (decode at 0.05 / 0.025) with the newer residual-upsampler decoder. The pipeline decode path differs; the VAE
    /// itself is built with the 0.9.5 block config (decoder_block_out_channels (256,512,1024), 5 layers/block,
    /// upsample_factor 2, residual upsamplers) — see the generation test.</summary>
    public static LtxVideoConfig V095 => new()
    {
        VaeTimestepConditioned = true,
    };

    /// <summary>LTX-Video 0.9.7 / 0.9.8 preset (13B). Verified against
    /// <c>Lightricks/LTX-Video-0.9.7-distilled/transformer/config.json</c>: <c>attention_head_dim=128</c>
    /// (inner 4096), <c>num_layers=48</c>, <c>cross_attention_dim=4096</c>. The 13B VAE is timestep-conditioned
    /// (decode at 0.05 / 0.025). 0.9.8 shares this transformer shape.</summary>
    public static LtxVideoConfig V097 => new()
    {
        HeadDim = 128,
        NumLayers = 48,
        CrossAttentionDim = 4096,
        VaeTimestepConditioned = true,
    };
}
