namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Geometry of the Synchformer video feature extractor (MotionFormer <c>divided_224_16x4.yaml</c> with a spatial
/// transformer-encoder aggregation and identity temporal aggregation).</summary>
public sealed record ControlFoleySynchformerConfig
{
    /// <summary>The released checkpoint.</summary>
    public static ControlFoleySynchformerConfig Released { get; } = new()
    {
        EmbedDim = 768, Depth = 12, Heads = 12, PatchSize = 16, InputSize = 224,
    };

    /// <summary>Token width.</summary>
    public required int EmbedDim { get; init; }

    /// <summary>Divided space-time blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Attention heads (blocks and aggregation layer).</summary>
    public required int Heads { get; init; }

    /// <summary>Spatial patch edge in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Frame edge in pixels; the official positional table is sized for 224 whatever the patch size.</summary>
    public int InputSize { get; init; } = 224;

    /// <summary>Frames merged by the 3D patch convolution.</summary>
    public int TemporalPatch { get; init; } = 2;

    /// <summary>Frames per segment.</summary>
    public int SegmentFrames { get; init; } = 16;

    /// <summary>Frames between segment starts.</summary>
    public int StrideFrames { get; init; } = 8;

    /// <summary>MLP expansion ratio.</summary>
    public int MlpRatio { get; init; } = 4;

    /// <summary>LayerNorm epsilon (block norms, final norm and aggregation layer).</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>Patch grid edge.</summary>
    public int GridSize => InputSize / PatchSize;

    /// <summary>Spatial tokens per frame.</summary>
    public int SpatialTokens => GridSize * GridSize;

    /// <summary>Temporal tokens per segment (output rows per segment).</summary>
    public int TemporalTokens => SegmentFrames / TemporalPatch;
}
