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
        NumHiddenLayers = 36,
        NumAttentionHeads = 32,
        NumKeyValueHeads = 8,
        IntermediateSize = 9_728,
        VocabSize = 155_776,
        MaxPositionEmbeddings = 32_768,
        HeadDim = 128,
        RopeTheta = 1_000_000f,
        RmsNormEps = 1e-6f,
    };

    /// <summary>Fast autoregressive depth transformer configuration.</summary>
    public FishAudioTransformerConfig Fast { get; init; } = new()
    {
        HiddenSize = 2_560,
        NumHiddenLayers = 4,
        NumAttentionHeads = 32,
        NumKeyValueHeads = 8,
        IntermediateSize = 9_728,
        VocabSize = 4_096,
        MaxPositionEmbeddings = 11,
        HeadDim = 128,
        RopeTheta = 1_000_000f,
        RmsNormEps = 1e-6f,
        QkNorm = false,
    };

    public int NumCodebooks { get; init; } = 10;
    public int CodebookSize { get; init; } = 4_096;
    public int SemanticCodebookSize { get; init; } = 4_096;
    /// <summary>The ModifiedDAC codec vocabulary size for each of its nine residual codebooks.</summary>
    public int ResidualCodebookSize { get; init; } = 1_024;
    /// <summary>Vocab id of <c>&lt;|semantic:0|&gt;</c> (config.json <c>semantic_start_token_id</c>).</summary>
    public int SemanticBeginId { get; init; } = 151_678;

    /// <summary>Vocab id of <c>&lt;|semantic:4095|&gt;</c> (config.json <c>semantic_end_token_id</c>).</summary>
    public int SemanticEndId { get; init; } = 155_773;

    /// <summary>Vocab id of <c>&lt;|im_end|&gt;</c>, which ends generation (config.json <c>eos_token_id</c>).</summary>
    public int ImEndId { get; init; } = 151_645;

    // Sampling defaults from fish-speech's generate_long: temperature 1.0, top-p 0.9, top-k 30, and Repetition
    // Aware Sampling (a 10-frame window; a repeated semantic token is re-drawn at temperature 1.0 / top-p 0.9).
    public float Temperature { get; init; } = 1.0f;
    public float TopP { get; init; } = 0.9f;
    public int TopK { get; init; } = 30;
    public int RasWindow { get; init; } = 10;
    public float RasTemperature { get; init; } = 1.0f;
    public float RasTopP { get; init; } = 0.9f;

    public int SampleRate { get; init; } = 44_100;
    public int FrameRate { get; init; } = 21;

    /// <summary>Published S2 Pro architecture contract.</summary>
    public static FishAudioS2Config S2Pro => new();
}
