namespace HartsyInference.Audio.Models.Wav2Vec2Bert;

/// <summary>Config for <see cref="Wav2Vec2BertExtractor"/> — HuggingFace <c>Wav2Vec2BertModel</c>
/// (<c>facebook/w2v-bert-2.0</c>'s real <c>config.json</c>, <c>position_embeddings_type: "relative_key"</c>).
/// Not IndexTTS-specific: any future model consuming a w2v-bert-2.0-family checkpoint can reuse this folder
/// without an IndexTTS-scoped name in the way.</summary>
public sealed record Wav2Vec2BertConfig
{
    public required int HiddenSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int ConvDepthwiseKernelSize { get; init; }
    public required int LeftMaxPositionEmbeddings { get; init; }
    public required int RightMaxPositionEmbeddings { get; init; }
    public required int FeatureProjectionInputDim { get; init; }
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Encoder layers to actually run. The real <c>facebook/w2v-bert-2.0</c> has 24 (<c>num_hidden_layers</c>
    /// in its config.json), but every consumer of this checkpoint only ever reads a specific intermediate
    /// <c>hidden_states[k]</c> entry — running (and even loading) the layers after it is dead compute. Callers
    /// request the cut directly rather than this record carrying an IndexTTS-specific constant.</summary>
    public required int NumLayersToRun { get; init; }

    /// <summary>The real <c>facebook/w2v-bert-2.0</c> preset. <paramref name="numLayersToRun"/> defaults to all 24;
    /// pass the real consumer's needed depth (IndexTTS-2 needs 17, for its real <c>hidden_states[17]</c> read —
    /// see <see cref="Wav2Vec2BertExtractor"/>) to skip loading/running the rest.</summary>
    public static Wav2Vec2BertConfig V2(int numLayersToRun = 24) => new()
    {
        HiddenSize = 1024,
        IntermediateSize = 4096,
        NumAttentionHeads = 16,
        ConvDepthwiseKernelSize = 31,
        LeftMaxPositionEmbeddings = 64,
        RightMaxPositionEmbeddings = 8,
        FeatureProjectionInputDim = 160,
        NumLayersToRun = numLayersToRun,
    };
}
