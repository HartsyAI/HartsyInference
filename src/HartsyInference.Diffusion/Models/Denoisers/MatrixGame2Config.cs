namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Matrix-Game 2.0 (Skywork, MIT) — the 1.8B real-time interactive world model:
/// SkyReels-V2 / Wan2.1 1.3B I2V DiT (text branch removed, CLIP-ViT-H visual context) with per-block ActionModules
/// and block-causal sliding-window attention, DMD-distilled to 3-4 steps. Three distilled variants differ only in
/// keyboard vocab / mouse presence / window / step list. See <c>docs/Research/MATRIX_GAME_2_ARCHITECTURE.md</c>.</summary>
public sealed record MatrixGame2Config
{
    /// <summary>Patch size (t, h, w) for the Conv3d patch embedding.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Attention heads.</summary>
    public int NumHeads { get; init; } = 12;

    /// <summary>Per-head dim.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Inner model dim.</summary>
    public int InnerDim => NumHeads * HeadDim;

    /// <summary>Patch-embed input channels: 16 noisy latent + 4 mask + 16 image-condition latent.</summary>
    public int InChannels { get; init; } = 36;

    /// <summary>Output channels (Wan2.1 VAE latent).</summary>
    public int OutChannels { get; init; } = 16;

    /// <summary>CLIP-ViT-H visual-context feature width.</summary>
    public int ClipContextDim { get; init; } = 1280;

    /// <summary>CLIP visual-context tokens (CLS + 16×16).</summary>
    public int ClipContextTokens { get; init; } = 257;

    /// <summary>Timestep sinusoidal frequency dim.</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>FFN inner dim.</summary>
    public int FfnDim { get; init; } = 8960;

    /// <summary>Number of DiT blocks.</summary>
    public int NumLayers { get; init; } = 30;

    /// <summary>Norm epsilon.</summary>
    public float Eps { get; init; } = 1e-6f;

    /// <summary>Main DiT RoPE θ.</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>RoPE max precomputed sequence length per axis.</summary>
    public int RopeMaxSeqLen { get; init; } = 1024;

    /// <summary>Block indices carrying an ActionModule (distilled: 0–14; foundation: all). Null = every block.</summary>
    public int[]? ActionBlocks { get; init; } = [.. Enumerable.Range(0, 15)];

    /// <summary>Keyboard one-hot width (universal 4, GTA 2, TempleRun 7).</summary>
    public int KeyboardDim { get; init; } = 4;

    /// <summary>Whether the mouse stream exists (TempleRun: false).</summary>
    public bool EnableMouse { get; init; } = true;

    /// <summary>Action-stream attention width.</summary>
    public int ActionStreamDim { get; init; } = 1024;

    /// <summary>Keyboard embed width.</summary>
    public int ActionHiddenSize { get; init; } = 128;

    /// <summary>ActionModule heads.</summary>
    public int ActionHeads { get; init; } = 16;

    /// <summary>ActionModule RoPE θ.</summary>
    public float ActionRopeTheta { get; init; } = 256f;

    /// <summary>Temporal window (latent frames) for action grouping.</summary>
    public int ActionWindowSize { get; init; } = 3;

    /// <summary>Sliding causal-attention window in latent frames (≤ 0 = full causal).</summary>
    public int LocalAttnSize { get; init; } = 6;

    /// <summary>Latent frames generated per autoregressive block.</summary>
    public int NumFramePerBlock { get; init; } = 3;

    /// <summary>Distilled denoising index list (descending, into the 1000-step schedule).</summary>
    public int[] DenoisingStepList { get; init; } = [1000, 666, 333];

    /// <summary>Flow-match timestep shift the student was distilled against.</summary>
    public float TimestepShift { get; init; } = 5.0f;

    /// <summary>Whether <see cref="DenoisingStepList"/> entries are warped through the shifted sigma grid before use
    /// (upstream <c>warp_denoising_step</c>; true for all shipped variants). False treats them as raw timesteps.</summary>
    public bool WarpDenoisingStep { get; init; } = true;

    /// <summary>Leading frames pinned in the rolling KV cache (upstream <c>sink_size</c>; 0 in all shipped variants).</summary>
    public int SinkSize { get; init; } = 0;

    /// <summary>Noise added to the cached context frames (upstream <c>context_noise</c>; 0 in all shipped variants).</summary>
    public float ContextNoise { get; init; } = 0f;

    /// <summary>Wan2.1 VAE spatial compression.</summary>
    public int VaeSpatialCompression { get; init; } = 8;

    /// <summary>Wan2.1 VAE temporal compression.</summary>
    public int VaeTemporalCompression { get; init; } = 4;

    /// <summary>Universal distilled variant (4-key WASD + mouse, 3 steps, window 6).</summary>
    public static MatrixGame2Config Universal => new();

    /// <summary>GTA-driving distilled variant (2-key + mouse, 3 steps, window 4).</summary>
    public static MatrixGame2Config GtaDrive => new() { KeyboardDim = 2, LocalAttnSize = 4 };

    /// <summary>TempleRun distilled variant (7-key, no mouse, 4 steps).</summary>
    public static MatrixGame2Config TempleRun => new()
    {
        KeyboardDim = 7,
        EnableMouse = false,
        DenoisingStepList = [1000, 750, 500, 250],
    };

    /// <summary>The undistilled foundation model (ActionModules in all 30 blocks, full causal attention).</summary>
    public static MatrixGame2Config Foundation => new() { ActionBlocks = null, LocalAttnSize = -1 };

    /// <summary>The matching WanVideoConfig for constructing the reused Wan blocks.</summary>
    public WanVideoConfig ToWanConfig() => new()
    {
        PatchSize = PatchSize,
        NumHeads = NumHeads,
        HeadDim = HeadDim,
        InChannels = InChannels,
        OutChannels = OutChannels,
        TextDim = ClipContextDim,
        FreqDim = FreqDim,
        FfnDim = FfnDim,
        NumLayers = NumLayers,
        Eps = Eps,
        RopeTheta = RopeTheta,
        RopeMaxSeqLen = RopeMaxSeqLen,
        VaeSpatialCompression = VaeSpatialCompression,
        VaeTemporalCompression = VaeTemporalCompression,
    };
}
