namespace HartsyInference.Audio.Models.FishAudio;

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
    public int HeadDim { get; init; }
}
