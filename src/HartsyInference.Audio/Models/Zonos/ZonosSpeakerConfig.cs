namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Configuration for <see cref="ZonosSpeakerEncoder"/> — the ResNet293 + SimAM + ASP speaker model
/// that produces the 128-d Zonos speaker embedding. Defaults follow the research spec; the channel schedule
/// and block counts are checkpoint-reconcile items (the official ResNet293 uses a much deeper stack — the
/// defaults here are the structurally-faithful, smaller-by-default config used for synthetic forwards).</summary>
public sealed record ZonosSpeakerConfig
{
    public int SampleRate { get; init; } = 16_000;
    public int NumMels { get; init; } = 80;

    /// <summary>Stem output channels.</summary>
    public int BaseWidth { get; init; } = 32;

    /// <summary>Per-stage output channel counts (the four ResNet stages).</summary>
    public IReadOnlyList<int> StageWidths { get; init; } = [32, 64, 128, 256];

    /// <summary>Per-stage residual-block counts. Official ResNet293 is very deep; the default is a compact
    /// stand-in with the same structure.</summary>
    public IReadOnlyList<int> StageBlocks { get; init; } = [3, 4, 6, 3];

    /// <summary>Frequency dimension after the (3×) stride-2 stages reduce the 80-mel axis. With strides
    /// [1,2,2,2] over 80 → 10.</summary>
    public int PooledFreq { get; init; } = 10;

    public int NormGroups { get; init; } = 32;
    public float SimAmLambda { get; init; } = 1e-4f;

    /// <summary>Linear bottleneck width before the LDA projection.</summary>
    public int BottleneckDim { get; init; } = 256;

    /// <summary>Final speaker embedding dimension.</summary>
    public int EmbedDim { get; init; } = 128;

    public static ZonosSpeakerConfig Default => new();
}
