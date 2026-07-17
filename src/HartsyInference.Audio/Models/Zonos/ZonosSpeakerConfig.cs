namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Configuration for <see cref="ZonosSpeakerEncoder"/> — the <c>ResNet293_based</c> speaker model
/// (Zyphra/Zonos-v0.1-speaker-embedding) that produces the 128-d Zonos speaker embedding. Values are the exact
/// checkpoint schedule: a 64-channel stem, four SimAM-BasicBlock stages with block counts <c>[10,20,64,3]</c> and
/// output widths <c>[64,128,256,512]</c>, attentive statistics pooling (ASP) over the folded (channel·freq)
/// axis, a linear bottleneck to 256-d, then a separate LDA projection to 128-d.</summary>
public sealed record ZonosSpeakerConfig
{
    public int SampleRate { get; init; } = 16_000;
    public int NumMels { get; init; } = 80;

    /// <summary>Stem output channels (ResNet293 <c>in_planes</c>).</summary>
    public int BaseWidth { get; init; } = 64;

    /// <summary>Per-stage output channel counts (the four ResNet293 stages: in_planes·{1,2,4,8}).</summary>
    public IReadOnlyList<int> StageWidths { get; init; } = [64, 128, 256, 512];

    /// <summary>Per-stage residual-block counts — the ResNet293 schedule.</summary>
    public IReadOnlyList<int> StageBlocks { get; init; } = [10, 20, 64, 3];

    /// <summary>Frequency dimension after the three stride-2 stages reduce the 80-mel axis (80 → 10).</summary>
    public int PooledFreq { get; init; } = 10;

    /// <summary>SimAM energy-weighting lambda (reference <c>lambda_p</c>).</summary>
    public float SimAmLambda { get; init; } = 1e-4f;

    /// <summary>BatchNorm epsilon (PyTorch default), folded into each conv at load.</summary>
    public float BnEps { get; init; } = 1e-5f;

    /// <summary>ASP attention bottleneck width (Conv1d 5120→128→5120).</summary>
    public int AspAttentionDim { get; init; } = 128;

    /// <summary>Linear bottleneck width (ASP out 10240 → 256).</summary>
    public int BottleneckDim { get; init; } = 256;

    /// <summary>Final speaker embedding dimension (LDA 256 → 128).</summary>
    public int EmbedDim { get; init; } = 128;

    public static ZonosSpeakerConfig Default => new();
}
