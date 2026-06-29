namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Matrix-Game 3.0 (Skywork, Apache-2.0) — a memory-augmented interactive Wan2.2 finetune.
/// DiT shape defaults match the shipped checkpoint's <c>base_model/config.json</c> (dim 3072 / 24 heads / 30 layers /
/// ffn 14336); the repo's stale <c>wan/configs/config.py</c> lists 5120/40/40 but no published checkpoint uses it.
/// <c>MatrixGame3CheckpointConverter.InferShape</c> still resolves the real shape from the weights at load time.
/// See <c>docs/Research/MATRIX_GAME_3_ARCHITECTURE.md</c>.</summary>
public sealed record MatrixGame3Config
{
    /// <summary>Patch size (t, h, w) for the Conv3d patch embedding.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Attention heads (<c>config.json num_heads</c>).</summary>
    public int NumHeads { get; init; } = 24;

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

    /// <summary>FFN inner dim (<c>config.json ffn_dim</c>).</summary>
    public int FfnDim { get; init; } = 14336;

    /// <summary>Number of DiT blocks (<c>config.json num_layers</c>).</summary>
    public int NumLayers { get; init; } = 30;

    /// <summary>Norm epsilon.</summary>
    public float Eps { get; init; } = 1e-6f;

    /// <summary>Main DiT RoPE θ (Wan2.2 default; the ActionModule uses its own θ=256).</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>Per-head RoPE θ spread (<c>config.json sigma_theta</c>): when &gt; 0 the main DiT rope uses a distinct
    /// θ per head, <c>θ_h = RopeTheta·(1 + SigmaTheta·ε_h)</c> with <c>ε_h = linspace(-1, 1, NumHeads)</c>. This is the
    /// FOV-memory rope modulation; 0 falls back to a single shared θ (plain Wan2.2 behavior).</summary>
    public float SigmaTheta { get; init; } = 0.8f;

    /// <summary>RoPE max precomputed sequence length per axis.</summary>
    public int RopeMaxSeqLen { get; init; } = 1024;

    /// <summary>Max text-context length: umT5 features are zero-padded / truncated to this (<c>config.json text_len</c>).</summary>
    public int TextLen { get; init; } = 512;

    /// <summary>Block indices carrying an ActionModule. Default = the shipped checkpoint's blocks 0..14 (15 of 30);
    /// the converter overrides this from the actual <c>action_model.*</c> keys at load.</summary>
    public int[]? ActionBlocks { get; init; } = [.. Enumerable.Range(0, 15)];

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

    /// <summary>The shipped 5B preset (dim 3072 / 24 heads / 30 layers / ffn 14336), matching
    /// <c>base_model/config.json</c>.</summary>
    public static MatrixGame3Config Base5B => new();

    /// <summary>Back-compat alias for <see cref="Base5B"/> — the shipped checkpoint <i>is</i> the TI2V-5B shape (the
    /// repo's 5120/40/40 <c>config.py</c> is stale and unused).</summary>
    public static MatrixGame3Config Ti2V5BShape => Base5B;

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
