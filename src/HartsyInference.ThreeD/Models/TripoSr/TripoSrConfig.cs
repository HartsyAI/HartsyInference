namespace HartsyInference.ThreeD.Models.TripoSr;

/// <summary>TripoSR (Stability AI / Tripo, MIT) configuration — a feed-forward single-image→3D LRM: a DINO
/// image tokenizer feeds a triplane transformer that produces a triplane, decoded by a NeRF MLP into
/// density+color and meshed via marching cubes. Deterministic (no diffusion). Dims marked here track the
/// public TripoSR checkpoint and are <b>validation-gated</b> (see the research notes).</summary>
public sealed record TripoSrConfig
{
    // --- Triplane ---
    /// <summary>Channels per triplane plane.</summary>
    public required int TriplaneChannels { get; init; }

    /// <summary>Triplane spatial resolution (H = W).</summary>
    public required int TriplaneResolution { get; init; }

    // --- Image→triplane transformer ---
    /// <summary>Transformer hidden width.</summary>
    public required int Width { get; init; }

    /// <summary>Number of transformer blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Image token dim (DINO hidden) projected to <see cref="Width"/>.</summary>
    public required int ImageTokenDim { get; init; }

    /// <summary>FFN intermediate size.</summary>
    public required int MlpDim { get; init; }

    // --- NeRF decoder ---
    /// <summary>NeRF MLP hidden width.</summary>
    public required int NerfHidden { get; init; }

    /// <summary>NeRF MLP hidden-layer count (excludes the final density+rgb projection).</summary>
    public required int NerfLayers { get; init; }

    // --- Extraction ---
    /// <summary>Density iso level for marching cubes.</summary>
    public float DensityThreshold { get; init; } = 10f;

    /// <summary>Default marching-cubes grid resolution per axis.</summary>
    public int GridResolution { get; init; } = 256;

    /// <summary>Half-extent of the query cube in object space (grid spans [-Bound, Bound]³).</summary>
    public float BoundingBox { get; init; } = 1.0f;

    /// <summary>Number of triplane tokens the transformer produces (3 planes × res²).</summary>
    public int TriplaneTokens => 3 * TriplaneResolution * TriplaneResolution;

    /// <summary>Default config for the public <c>stabilityai/TripoSR</c> checkpoint. <b>Validation-gated.</b>
    /// Pairs with a DINO ViT-B/16 tokenizer (ImageTokenDim = 768).</summary>
    public static TripoSrConfig TripoSr => new()
    {
        TriplaneChannels = 40, TriplaneResolution = 64,
        Width = 1024, Depth = 16, NumHeads = 16, ImageTokenDim = 768, MlpDim = 4096,
        NerfHidden = 64, NerfLayers = 9,
        DensityThreshold = 10f, GridResolution = 256, BoundingBox = 1.0f,
    };
}
