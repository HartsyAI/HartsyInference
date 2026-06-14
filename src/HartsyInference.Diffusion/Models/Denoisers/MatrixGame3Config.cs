namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Matrix-Game 3.0 (Skywork, Apache-2.0) — a memory-augmented interactive Wan2.2 finetune.
/// DiT shape defaults follow the repo's <c>wan/configs/config.py</c> (dim 5120 / 40 layers / 40 heads); note the
/// Wan-AI diffusers TI2V-5B config disagrees (3072/24/30) and the 12.9 GB checkpoint size favors the smaller shape —
/// <c>MatrixGame3CheckpointConverter.InferShape</c> resolves the real shape from the weights at load time, so these
/// defaults are placeholders until first download. See <c>docs/Research/MATRIX_GAME_3_ARCHITECTURE.md</c>.</summary>
public sealed record MatrixGame3Config
{
    /// <summary>Patch size (t, h, w) for the Conv3d patch embedding.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Attention heads.</summary>
    public int NumHeads { get; init; } = 40;

    /// <summary>Per-head dim (fixed 128 across Wan2.2 variants).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Inner model dim.</summary>
    public int InnerDim => NumHeads * HeadDim;

    /// <summary>VAE latent channels (Wan2.2 z=48).</summary>
    public int InChannels { get; init; } = 48;

    /// <summary>Output channels.</summary>
    public int OutChannels { get; init; } = 48;

    /// <summary>umT5-XXL feature width.</summary>
    public int TextDim { get; init; } = 4096;

    /// <summary>Timestep sinusoidal frequency dim.</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>FFN inner dim.</summary>
    public int FfnDim { get; init; } = 13824;

    /// <summary>Number of DiT blocks.</summary>
    public int NumLayers { get; init; } = 40;

    /// <summary>Norm epsilon.</summary>
    public float Eps { get; init; } = 1e-6f;

    /// <summary>Main DiT RoPE θ (Wan2.2 default; the ActionModule uses its own θ=256).</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>RoPE max precomputed sequence length per axis.</summary>
    public int RopeMaxSeqLen { get; init; } = 1024;

    /// <summary>Block indices carrying an ActionModule; null = every block (Open Question 5 — confirmed at key-dump time).</summary>
    public int[]? ActionBlocks { get; init; }

    /// <summary>Action-stream attention width (mouse/keyboard hidden, 1024).</summary>
    public int ActionStreamDim { get; init; } = 1024;

    /// <summary>Keyboard embed width.</summary>
    public int ActionHiddenSize { get; init; } = 128;

    /// <summary>ActionModule heads.</summary>
    public int ActionHeads { get; init; } = 16;

    /// <summary>ActionModule RoPE θ.</summary>
    public float ActionRopeTheta { get; init; } = 256f;

    /// <summary>Temporal window (latent frames) the ActionModule attends over.</summary>
    public int ActionWindowSize { get; init; } = 3;

    /// <summary>Plücker token input width (<c>patch_embedding_wancamctrl</c> in-features; ray channels × pixels per
    /// token — validation-gated against the checkpoint, the loader accepts whatever the weight says).</summary>
    public int PluckerPatchDim { get; init; } = 6144;

    /// <summary>FOV-retrieved memory slots prepended to the sequence.</summary>
    public int MemorySlots { get; init; } = 5;

    /// <summary>Clean past-latent overlap frames carried into each new segment.</summary>
    public int PastFrames { get; init; } = 4;

    /// <summary>Latent frames in the bootstrap segment (57 RGB frames).</summary>
    public int FirstSegmentLatents { get; init; } = 15;

    /// <summary>Latent frames per subsequent segment (40 RGB frames).</summary>
    public int SegmentLatents { get; init; } = 10;

    /// <summary>VAE spatial compression.</summary>
    public int VaeSpatialCompression { get; init; } = 16;

    /// <summary>VAE temporal compression.</summary>
    public int VaeTemporalCompression { get; init; } = 4;

    /// <summary>Flow-matching timestep shift (<c>sample_shift</c>).</summary>
    public float SampleShift { get; init; } = 5.0f;

    /// <summary>CFG scale (<c>sample_guide_scale</c>).</summary>
    public float GuidanceScale { get; init; } = 5.0f;

    /// <summary>FlowUniPC steps for the base checkpoint.</summary>
    public int StepsBase { get; init; } = 50;

    /// <summary>FlowUniPC steps for the DMD-distilled checkpoint.</summary>
    public int StepsDistilled { get; init; } = 3;

    /// <summary>The 5B preset per the repo config.</summary>
    public static MatrixGame3Config Base5B => new();

    /// <summary>The Wan2.2 TI2V-5B diffusers shape (3072 / 24 heads / 30 layers / ffn 14336) — the alternative reading
    /// of the checkpoint size; selected automatically when shape inference says so.</summary>
    public static MatrixGame3Config Ti2V5BShape => new() { NumHeads = 24, NumLayers = 30, FfnDim = 14336 };

    /// <summary>The matching WanVideoConfig for constructing the reused Wan blocks.</summary>
    public WanVideoConfig ToWanConfig() => new()
    {
        PatchSize = PatchSize,
        NumHeads = NumHeads,
        HeadDim = HeadDim,
        InChannels = InChannels,
        OutChannels = OutChannels,
        TextDim = TextDim,
        FreqDim = FreqDim,
        FfnDim = FfnDim,
        NumLayers = NumLayers,
        Eps = Eps,
        RopeTheta = RopeTheta,
        RopeMaxSeqLen = RopeMaxSeqLen,
        VaeSpatialCompression = VaeSpatialCompression,
        VaeTemporalCompression = VaeTemporalCompression,
        FlowShift = SampleShift,
        GuidanceScale = GuidanceScale,
        NumInferenceSteps = StepsBase,
    };
}
