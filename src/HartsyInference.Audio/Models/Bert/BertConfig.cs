namespace HartsyInference.Audio.Models.Bert;

/// <summary>Configuration for a standard HuggingFace BERT encoder. Defaults match
/// <c>chinese-roberta-wwm-ext-large</c> (the GPT-SoVITS text-conditioning BERT): 24 post-norm layers, hidden
/// 1024, 16 heads, intermediate 4096, vocab 21128, learned absolute positions, exact GELU, LayerNorm eps 1e-12.</summary>
public sealed record BertConfig
{
    public int Hidden { get; init; } = 1024;
    public int NumLayers { get; init; } = 24;
    public int NumHeads { get; init; } = 16;
    public int Intermediate { get; init; } = 4096;
    public int VocabSize { get; init; } = 21128;
    public int MaxPositions { get; init; } = 512;
    public int TypeVocab { get; init; } = 2;
    public float LayerNormEps { get; init; } = 1e-12f;

    public int HeadDim => Hidden / NumHeads;     // 64

    /// <summary>chinese-roberta-wwm-ext-large preset (GPT-SoVITS).</summary>
    public static BertConfig ChineseRobertaWwmExtLarge => new();
}
