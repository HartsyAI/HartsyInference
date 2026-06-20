namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Wan-Video (Wan-AI, Apache-2.0), shared by every variant of the <c>WanTransformer3DModel</c>
/// DiT: single-stream self-attention + per-head 3D RoPE, cross-attention to umT5, optional image cross-attention
/// (I2V), 6-param AdaLN timestep modulation, FP32 LayerNorms. Defaults target **Wan2.2 TI2V-5B** (z=48 VAE, 16×
/// spatial / 4× temporal). Use the static presets for the other variants. See <c>docs/Research/WAN_VIDEO_ARCHITECTURE.md</c>.</summary>
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

    /// <summary>Transformer input channels (T2V = VAE z; I2V = z + cond-latent + mask, e.g. 36).</summary>
    public int InChannels { get; init; } = 48;

    /// <summary>Output channels (the predicted-velocity VAE latent width).</summary>
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

    /// <summary>VAE spatial compression (Wan2.1 16-ch VAE = 8; Wan2.2 48-ch TI2V VAE = 16).</summary>
    public int VaeSpatialCompression { get; init; } = 16;

    /// <summary>VAE temporal compression.</summary>
    public int VaeTemporalCompression { get; init; } = 4;

    /// <summary>VAE latent channels (16 for Wan2.1 / A14B, 48 for TI2V-5B). Equals <see cref="OutChannels"/>.</summary>
    public int VaeLatentChannels { get; init; } = 48;

    /// <summary>CLIP image-encoder feature width fed to the I2V image cross-attention (0 = no CLIP image conditioning;
    /// 1280 for Wan2.1 I2V-14B's CLIP-ViT-H). Drives <c>condition_embedder.image_embedder</c> + the per-block
    /// <c>add_k_proj</c>/<c>add_v_proj</c> image KV projections.</summary>
    public int ImageDim { get; init; }

    /// <summary>Added KV projection dim for the I2V image cross-attention (0 = none; = <see cref="ImageDim"/> when set).</summary>
    public int AddedKvProjDim { get; init; }

    /// <summary>Learnable position-embedding sequence length in the image embedder (0 = none; 257 for CLIP-ViT-H).</summary>
    public int PosEmbedSeqLen { get; init; }

    /// <summary>Mixture-of-experts boundary ratio for Wan2.2 A14B (0 = single expert). When &gt; 0, the high-noise
    /// expert runs while <c>timestep ≥ BoundaryRatio · 1000</c> and the low-noise expert below it.</summary>
    public float BoundaryRatio { get; init; }

    /// <summary>Default flow-match shift (5.0 for 720p, 3.0 for 480p).</summary>
    public float FlowShift { get; init; } = 5.0f;

    /// <summary>VACE control-branch layer indices (the main-block indices where the VACE hints are added back).</summary>
    public int[] VaceLayers { get; init; } = [];

    /// <summary>VACE control patch-embed input channels (Wan VACE: 96 = inactive-latent 16 + reactive-latent 16 + space-to-depth mask 64).</summary>
    public int VaceInChannels { get; init; } = 96;

    /// <summary>Default sampling steps.</summary>
    public int NumInferenceSteps { get; init; } = 50;

    /// <summary>Default CFG guidance scale.</summary>
    public float GuidanceScale { get; init; } = 5.0f;

    /// <summary>True when this variant carries CLIP image conditioning (Wan2.1 I2V).</summary>
    public bool HasImageConditioning => ImageDim > 0;

    /// <summary>True when this variant is a two-expert MoE (Wan2.2 A14B).</summary>
    public bool IsMixtureOfExperts => BoundaryRatio > 0f;

    // ── Presets ──

    /// <summary>Wan2.2 TI2V-5B (z=48 VAE, text+first-frame-latent I2V via expand_timesteps).</summary>
    public static WanVideoConfig Ti2V5B => new();

    /// <summary>Wan2.1 T2V-1.3B (480p): inner 1536, 30 layers, ffn 8960, z=16 VAE (8×/4×).</summary>
    public static WanVideoConfig T2V_1_3B => new()
    {
        NumHeads = 12, HeadDim = 128, InChannels = 16, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 8960, NumLayers = 30, VaeSpatialCompression = 8, FlowShift = 3.0f,
    };

    /// <summary>Wan2.1 T2V-14B: inner 5120, 40 layers, ffn 13824, z=16 VAE (8×/4×).</summary>
    public static WanVideoConfig T2V_14B => new()
    {
        NumHeads = 40, HeadDim = 128, InChannels = 16, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 13824, NumLayers = 40, VaeSpatialCompression = 8, FlowShift = 5.0f,
    };

    /// <summary>Wan2.1 I2V-14B (480p): T2V-14B + 36-ch concat conditioning (z 16 + cond-latent 16 + mask 4) and
    /// CLIP-ViT-H image cross-attention (image_dim 1280, 257 pos-embed tokens).</summary>
    public static WanVideoConfig I2V_14B_480p => new()
    {
        NumHeads = 40, HeadDim = 128, InChannels = 36, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 13824, NumLayers = 40, VaeSpatialCompression = 8, FlowShift = 3.0f,
        ImageDim = 1280, AddedKvProjDim = 1280, PosEmbedSeqLen = 257,
    };

    /// <summary>Wan2.1 I2V-14B (720p): like <see cref="I2V_14B_480p"/> with the 720p flow shift (5.0).</summary>
    public static WanVideoConfig I2V_14B_720p => I2V_14B_480p with { FlowShift = 5.0f };

    /// <summary>Wan2.2 T2V-A14B: MoE dual-expert (boundary 0.875), inner 5120, 40 layers, z=16 VAE (8×/4×).</summary>
    public static WanVideoConfig T2V_A14B => new()
    {
        NumHeads = 40, HeadDim = 128, InChannels = 16, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 13824, NumLayers = 40, VaeSpatialCompression = 8, FlowShift = 5.0f, BoundaryRatio = 0.875f,
    };

    /// <summary>Wan2.2 I2V-A14B: MoE dual-expert (boundary 0.9) with 36-ch concat conditioning. No CLIP image
    /// encoder (image is conditioned purely through the concatenated cond-latent + mask channels).</summary>
    public static WanVideoConfig I2V_A14B => new()
    {
        NumHeads = 40, HeadDim = 128, InChannels = 36, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 13824, NumLayers = 40, VaeSpatialCompression = 8, FlowShift = 5.0f, BoundaryRatio = 0.9f,
    };

    /// <summary>Wan2.1 VACE-14B: T2V-14B + the VACE control branch (8 hint layers, 96-ch control patch embed).</summary>
    public static WanVideoConfig Vace_14B => new()
    {
        NumHeads = 40, HeadDim = 128, InChannels = 16, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 13824, NumLayers = 40, VaeSpatialCompression = 8, FlowShift = 5.0f,
        VaceLayers = [0, 5, 10, 15, 20, 25, 30, 35], VaceInChannels = 96,
    };

    /// <summary>Wan2.1 VACE-1.3B: T2V-1.3B + the VACE control branch (6 hint layers within the 30-layer DiT).</summary>
    public static WanVideoConfig Vace_1_3B => new()
    {
        NumHeads = 12, HeadDim = 128, InChannels = 16, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 8960, NumLayers = 30, VaeSpatialCompression = 8, FlowShift = 3.0f,
        VaceLayers = [0, 5, 10, 15, 20, 25], VaceInChannels = 96,
    };

    /// <summary>Wan2.2-S2V-14B (speech-to-video): the Wan2.1-14B backbone with the audio injector. Config values are
    /// provisional (reconstructed from the original Wan repo, not diffusers — see <c>WAN_VIDEO_ARCHITECTURE.md</c>).
    /// <c>InChannels</c> stays at the VAE z here (audio+text); the pipeline may bump it to concat a reference latent.</summary>
    public static WanVideoConfig S2V_14B => new()
    {
        NumHeads = 40, HeadDim = 128, InChannels = 16, OutChannels = 16, VaeLatentChannels = 16,
        FfnDim = 13824, NumLayers = 40, VaeSpatialCompression = 8, FlowShift = 5.0f,
    };
}
