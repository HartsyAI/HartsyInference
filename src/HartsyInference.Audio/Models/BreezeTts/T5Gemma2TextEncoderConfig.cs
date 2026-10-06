namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>Configuration of the T5Gemma2 text encoder Breeze TTS 2 conditions on (checkpoint <c>text_encoder_config</c>):
/// a Gemma-3-style stack with per-head Q/K norm and sandwich norms, but bidirectional attention, a symmetric sliding
/// window on most layers and a linearly scaled global RoPE.</summary>
public sealed record T5Gemma2TextEncoderConfig
{
    public int VocabSize { get; init; } = 262_158;
    public int HiddenSize { get; init; } = 1_152;
    public int IntermediateSize { get; init; } = 6_912;
    public int NumLayers { get; init; } = 26;
    public int NumHeads { get; init; } = 4;
    public int NumKvHeads { get; init; } = 1;
    public int HeadDim { get; init; } = 256;
    public float RmsNormEps { get; init; } = 1e-6f;
    public float QueryPreAttnScalar { get; init; } = 256f;
    public int SlidingWindow { get; init; } = 512;
    /// <summary>Every <c>SlidingPatternPeriod</c>-th layer (1-based) is full attention; the rest slide.</summary>
    public int SlidingPatternPeriod { get; init; } = 6;
    public float LocalRopeTheta { get; init; } = 10_000f;
    public float GlobalRopeTheta { get; init; } = 1_000_000f;
    public float GlobalRopeLinearFactor { get; init; } = 8f;
    public int EoiTokenIndex { get; init; } = 256_000;

    public bool IsFullAttention(int layer) => (layer + 1) % SlidingPatternPeriod == 0;

    public static T5Gemma2TextEncoderConfig Breeze => new();
}
