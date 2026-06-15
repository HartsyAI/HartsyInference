namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 shape-generation configuration: the VecSet flow-match DiT plus the ShapeVAE decoder.
/// Exact values for a given checkpoint come from its config; the defaults here track Hunyuan3D-2 and the
/// dims marked in <c>docs/Research/HUNYUAN3D_2_ARCHITECTURE.md</c> are <b>validation-gated</b>.</summary>
public sealed record Hunyuan3DConfig
{
    // --- VecSet latent / DiT ---
    /// <summary>Number of latent set tokens N the DiT denoises.</summary>
    public required int LatentTokens { get; init; }

    /// <summary>Latent channel dim C (the noised tensor's per-token width).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>DiT hidden width.</summary>
    public required int Width { get; init; }

    /// <summary>Number of DiT blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>DiT attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Conditioning token dim (DINOv2 hidden) projected to <see cref="Width"/>.</summary>
    public required int CondDim { get; init; }

    /// <summary>FFN intermediate size in the DiT (typically 4× <see cref="Width"/>).</summary>
    public required int MlpDim { get; init; }

    /// <summary>Sinusoidal timestep embedding dim before its MLP.</summary>
    public int TimestepEmbedDim { get; init; } = 256;

    // --- ShapeVAE decoder ---
    /// <summary>ShapeVAE decoder hidden width.</summary>
    public required int VaeWidth { get; init; }

    /// <summary>ShapeVAE decoder cross-attn/MLP block count.</summary>
    public required int VaeDepth { get; init; }

    /// <summary>ShapeVAE decoder attention heads.</summary>
    public required int VaeNumHeads { get; init; }

    /// <summary>Number of Fourier frequency bands for query-point positional encoding (per axis).</summary>
    public int FourierBands { get; init; } = 8;

    // --- Sampling / extraction defaults ---
    /// <summary>Default flow-match denoise steps.</summary>
    public int NumInferenceSteps { get; init; } = 50;

    /// <summary>Default CFG scale.</summary>
    public float GuidanceScale { get; init; } = 5.0f;

    /// <summary>Flow-match timestep shift.</summary>
    public float FlowShift { get; init; } = 1.0f;

    /// <summary>Default marching-cubes grid resolution per axis.</summary>
    public int GridResolution { get; init; } = 256;

    /// <summary>Iso level for surface extraction from the decoded field.</summary>
    public float IsoLevel { get; init; }

    /// <summary>Half-extent of the cubic query box in object space (grid spans [-Bound, Bound]³).</summary>
    public float BoundingBox { get; init; } = 1.01f;

    /// <summary>Default config for the <c>tencent/Hunyuan3D-2</c> shape model. <b>All dims are
    /// validation-gated</b> — reconcile against the checkpoint's <c>config.json</c> during the diff pass.
    /// Pairs with <c>Dinov2Preset.Large</c> (CondDim = 1024).</summary>
    public static Hunyuan3DConfig Hunyuan3D2 => new()
    {
        LatentTokens = 3072, LatentChannels = 64, Width = 1024, Depth = 21, NumHeads = 16, CondDim = 1024,
        MlpDim = 4096, TimestepEmbedDim = 256,
        VaeWidth = 1024, VaeDepth = 16, VaeNumHeads = 16, FourierBands = 8,
        NumInferenceSteps = 50, GuidanceScale = 5.0f, FlowShift = 1.0f,
        GridResolution = 256, IsoLevel = 0f, BoundingBox = 1.01f,
    };
}
