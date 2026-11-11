namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Hunyuan3D-2 shape-generation configuration for the VecSet flow-match DiT plus ShapeVAE decoder; dims are read from the checkpoint's <c>config.yaml</c> (see <c>docs/Research/HUNYUAN3D_ARCHITECTURE.md</c>).</summary>
public sealed record Hunyuan3DConfig
{
    // --- VecSet latent / DiT ---
    /// <summary>Number of latent set tokens N the DiT denoises.</summary>
    public required int LatentTokens { get; init; }

    /// <summary>Latent channel dim C (the noised tensor's per-token width).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>DiT hidden width (hidden_size).</summary>
    public required int Width { get; init; }

    /// <summary>Number of Flux <b>double</b>-stream blocks (config <c>depth</c>).</summary>
    public required int DepthDouble { get; init; }

    /// <summary>Number of Flux <b>single</b>-stream blocks (config <c>depth_single_blocks</c>).</summary>
    public required int DepthSingle { get; init; }

    /// <summary>DiT attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Conditioning token dim (DINOv2-giant hidden = 1536) projected to <see cref="Width"/> by <c>cond_in</c>.</summary>
    public required int CondDim { get; init; }

    /// <summary>FFN intermediate size in the DiT (<c>mlp_ratio</c>× <see cref="Width"/> = 4096).</summary>
    public required int MlpDim { get; init; }

    /// <summary>Sinusoidal timestep embedding dim before its MLP (Flux uses 256).</summary>
    public int TimestepEmbedDim { get; init; } = 256;

    /// <summary>Timestep scale applied before the sinusoid (<c>time_factor</c>, Flux uses 1000).</summary>
    public float TimeFactor { get; init; } = 1000f;

    // --- ShapeVAE decoder ---
    /// <summary>ShapeVAE decoder hidden width.</summary>
    public required int VaeWidth { get; init; }

    /// <summary>ShapeVAE decoder cross-attn/MLP block count.</summary>
    public required int VaeDepth { get; init; }

    /// <summary>ShapeVAE decoder attention heads.</summary>
    public required int VaeNumHeads { get; init; }

    /// <summary>Number of Fourier frequency bands for query-point positional encoding (per axis).</summary>
    public int FourierBands { get; init; } = 8;

    /// <summary>ShapeVAE latent scale (config <c>scale_factor</c>); the sampled latent is divided by this before decode.</summary>
    public float VaeScaleFactor { get; init; } = 1.0f;

    /// <summary>Geo-decoder width divisor (the turbo VAE halves it to 512 and projects the latents down first).</summary>
    public int GeoDownsample { get; init; } = 1;

    /// <summary>Geo-decoder MLP expansion (the turbo VAE uses 1).</summary>
    public int GeoMlpRatio { get; init; } = 4;

    /// <summary>Whether the geo-decoder applies <c>ln_post</c>; without it QK-norm is off too.</summary>
    public bool GeoLnPost { get; init; } = true;

    // --- Distilled and multiview variants ---
    /// <summary>The DiT adds a guidance embedding to the timestep vector and runs one forward per step instead of CFG (Fast, Turbo).</summary>
    public bool GuidanceEmbed { get; init; }

    /// <summary>Sigma schedule of the checkpoint's scheduler.</summary>
    public Hunyuan3DScheduler Scheduler { get; init; } = Hunyuan3DScheduler.FlowMatchEuler;

    /// <summary>The consistency scheduler's <c>pcm_timesteps</c>; read only when <see cref="Scheduler"/> is consistency.</summary>
    public int PcmTimesteps { get; init; } = 50;

    /// <summary>Number of conditioning views (4 for the multiview checkpoints, else 1).</summary>
    public int ViewCount { get; init; } = 1;

    // --- Sampling / extraction defaults ---
    /// <summary>Default flow-match denoise steps.</summary>
    public int NumInferenceSteps { get; init; } = 50;

    /// <summary>Default CFG scale.</summary>
    public float GuidanceScale { get; init; } = 5.0f;

    /// <summary>Flow-match timestep shift.</summary>
    public float FlowShift { get; init; } = 1.0f;

    /// <summary>Default marching-cubes grid resolution per axis: the number of samples, one fewer than upstream's <c>octree_resolution + 1</c> for the same value.</summary>
    public int GridResolution { get; init; } = 256;

    /// <summary>Iso level for surface extraction from the decoded field.</summary>
    public float IsoLevel { get; init; }

    /// <summary>Half-extent of the cubic query box in object space (grid spans [-Bound, Bound]³).</summary>
    public float BoundingBox { get; init; } = 1.01f;

    /// <summary>Default config for the <c>tencent/Hunyuan3D-2</c> shape model; dims match the checkpoint's <c>config.yaml</c>.</summary>
    public static Hunyuan3DConfig Hunyuan3D2 => new()
    {
        LatentTokens = 3072, LatentChannels = 64, Width = 1024, DepthDouble = 16, DepthSingle = 32, NumHeads = 16,
        CondDim = 1536, MlpDim = 4096, TimestepEmbedDim = 256, TimeFactor = 1000f,
        VaeWidth = 1024, VaeDepth = 16, VaeNumHeads = 16, FourierBands = 8, VaeScaleFactor = 0.9990943042622529f,
        NumInferenceSteps = 50, GuidanceScale = 5.0f, FlowShift = 1.0f,
        GridResolution = 256, IsoLevel = 0f, BoundingBox = 1.01f,
    };
}
