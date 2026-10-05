namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Architecture contract for Fish Audio's S2 family. The S2 checkpoint is a distinct Dual-AR
/// implementation from Fish-Speech 1.5: a Qwen3 slow transformer predicts semantic frames and a depth
/// transformer predicts the residual acoustic codebooks.</summary>
public sealed record FishAudioS2Config
{
    /// <summary>Slow autoregressive transformer configuration.</summary>
    public FishAudioTransformerConfig Slow { get; init; } = new()
    {
        HiddenSize = 4_096,
        NumHiddenLayers = 36,
        NumAttentionHeads = 32,
        NumKeyValueHeads = 8,
        IntermediateSize = 11_008,
        VocabSize = 151_936,
        MaxPositionEmbeddings = 32_768,
        RopeTheta = 1_000_000f,
        RmsNormEps = 1e-6f,
    };

    /// <summary>Fast autoregressive depth transformer configuration.</summary>
    public FishAudioTransformerConfig Fast { get; init; } = new()
    {
        HiddenSize = 1_024,
        NumHiddenLayers = 4,
        NumAttentionHeads = 16,
        NumKeyValueHeads = 2,
        IntermediateSize = 4_096,
        VocabSize = 4_096,
        MaxPositionEmbeddings = 16,
        RopeTheta = 1_000_000f,
        RmsNormEps = 1e-6f,
    };

    public int NumCodebooks { get; init; } = 10;
    public int CodebookSize { get; init; } = 4_096;
    public int SampleRate { get; init; } = 44_100;
    public int FrameRate { get; init; } = 21;

    /// <summary>Published S2 Pro architecture contract.</summary>
    public static FishAudioS2Config S2Pro => new();
}

/// <summary>Transformer dimensions kept separate from <c>Qwen2Config</c> until the checkpoint loader accounts
/// for S2's QK normalization and tokenizer contract.</summary>
public sealed record FishAudioTransformerConfig
{
    public required int HiddenSize { get; init; }
    public required int NumHiddenLayers { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int NumKeyValueHeads { get; init; }
    public required int IntermediateSize { get; init; }
    public required int VocabSize { get; init; }
    public required int MaxPositionEmbeddings { get; init; }
    public float RopeTheta { get; init; } = 1_000_000f;
    public float RmsNormEps { get; init; } = 1e-6f;
    public bool QkNorm { get; init; } = true;
    public int HeadDim => HiddenSize / NumAttentionHeads;
}
