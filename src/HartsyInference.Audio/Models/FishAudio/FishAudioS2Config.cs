namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Architecture contract for Fish Audio's S2 family. The S2 checkpoint is a distinct Dual-AR
/// implementation from Fish-Speech 1.5: a Qwen3 slow transformer predicts semantic frames and a depth
/// transformer predicts the residual acoustic codebooks.</summary>
public sealed record FishAudioS2Config
{
    /// <summary>Slow autoregressive transformer configuration.</summary>
    public FishAudioTransformerConfig Slow { get; init; } = new()
    {
        HiddenSize = 2_560,
        HeadDim = 128,
        NumHiddenLayers = 36,
        NumAttentionHeads = 32,
        NumKeyValueHeads = 8,
        IntermediateSize = 9_728,
        VocabSize = 155_776,
        MaxPositionEmbeddings = 32_768,
        RopeTheta = 1_000_000f,
        RmsNormEps = 1e-6f,
    };

    /// <summary>Fast autoregressive depth transformer configuration.</summary>
    public FishAudioTransformerConfig Fast { get; init; } = new()
    {
        HiddenSize = 2_560,
        HeadDim = 128,
        NumHiddenLayers = 4,
        NumAttentionHeads = 32,
        NumKeyValueHeads = 8,
        IntermediateSize = 9_728,
        VocabSize = 4_096,
        MaxPositionEmbeddings = 11,
        RopeTheta = 1_000_000f,
        RmsNormEps = 1e-6f,
        QkNorm = false,
    };

    public int NumCodebooks { get; init; } = 10;
    public int CodebookSize { get; init; } = 4_096;
    public int SemanticCodebookSize { get; init; } = 4_096;
    public int ResidualCodebookSize { get; init; } = 1_024;
    public int SampleRate { get; init; } = 44_100;
    public int FrameRate { get; init; } = 21;

    /// <summary>Published S2 Pro architecture contract.</summary>
    public static FishAudioS2Config S2Pro => new();
}
