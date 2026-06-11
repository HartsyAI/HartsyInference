namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Wan-Video (Wan-AI, Apache-2.0). Defaults target **Wan2.2 TI2V-5B** — the variant whose VAE is the already-built <c>Wan22VaeDecoder</c> (z=48, 16× spatial / 4× temporal). The DiT is single-stream: self-attention + per-head 3D RoPE, cross-attention to umT5, AdaLN (6-param) timestep modulation, FP32 LayerNorms. See <c>docs/Research/WAN_VIDEO_ARCHITECTURE.md</c>.</summary>
public sealed record WanVideoConfig
{
    /// <summary>Patch size (t, h, w) for the Conv3d patch embedding.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Attention heads.</summary>
    public int NumHeads { get; init; } = 24;

    /// <summary>Per-head dim.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Inner model dim (<c>NumHeads × HeadDim</c>).</summary>
    public int InnerDim => NumHeads * HeadDim;

    /// <summary>VAE latent channels in/out.</summary>
    public int InChannels { get; init; } = 48;

    /// <summary>Output channels.</summary>
    public int OutChannels { get; init; } = 48;

    /// <summary>umT5 feature width.</summary>
    public int TextDim { get; init; } = 4096;

    /// <summary>Timestep sinusoidal frequency dim.</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>FFN inner dim.</summary>
    public int FfnDim { get; init; } = 14336;

    /// <summary>Number of DiT blocks.</summary>
    public int NumLayers { get; init; } = 30;

    /// <summary>LayerNorm/RMSNorm epsilon.</summary>
    public float Eps { get; init; } = 1e-6f;

    /// <summary>RoPE base θ.</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>RoPE max precomputed sequence length per axis.</summary>
    public int RopeMaxSeqLen { get; init; } = 1024;

    /// <summary>VAE spatial compression.</summary>
    public int VaeSpatialCompression { get; init; } = 16;

    /// <summary>VAE temporal compression.</summary>
    public int VaeTemporalCompression { get; init; } = 4;

    /// <summary>Default flow-match shift (5.0 for 720p, 3.0 for 480p).</summary>
    public float FlowShift { get; init; } = 5.0f;

    /// <summary>Default sampling steps.</summary>
    public int NumInferenceSteps { get; init; } = 50;

    /// <summary>Default CFG guidance scale.</summary>
    public float GuidanceScale { get; init; } = 5.0f;

    /// <summary>The Wan2.2 TI2V-5B preset.</summary>
    public static WanVideoConfig Ti2V5B => new();
}
