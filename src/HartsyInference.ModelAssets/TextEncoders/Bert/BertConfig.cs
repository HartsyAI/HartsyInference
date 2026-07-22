namespace HartsyInference.ModelAssets.TextEncoders.Bert;

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
    /// <summary>Padding token id used to mask out positions in the attention mask.</summary>
    public int PadTokenId { get; init; } = 0;

    public int HeadDim => Hidden / NumHeads;     // 64

    /// <summary>chinese-roberta-wwm-ext-large preset (GPT-SoVITS).</summary>
    public static BertConfig ChineseRobertaWwmExtLarge => new();

    /// <summary>bert-base-uncased preset (MeloTTS English prosody BERT): 12 layers, hidden 768, 12 heads,
    /// intermediate 3072, vocab 30522.</summary>
    public static BertConfig BaseUncased => new()
    {
        Hidden = 768, NumLayers = 12, NumHeads = 12, Intermediate = 3072, VocabSize = 30522,
    };

    /// <summary>cl-tohoku/bert-base-japanese-v3 preset (MeloTTS Japanese prosody BERT): 12 layers, hidden 768,
    /// 12 heads, intermediate 3072, vocab 32768.</summary>
    public static BertConfig JapaneseV3 => new()
    {
        Hidden = 768, NumLayers = 12, NumHeads = 12, Intermediate = 3072, VocabSize = 32768,
    };

    /// <summary>hfl/chinese-roberta-wwm-ext preset (base size): 12 layers, hidden 768, 12 heads,
    /// intermediate 3072, vocab 21128.</summary>
    public static BertConfig ChineseRobertaWwmExtBase => new()
    {
        Hidden = 768, NumLayers = 12, NumHeads = 12, Intermediate = 3072, VocabSize = 21128,
    };

    /// <summary>Grounding DINO text tower (<c>bert-base-uncased</c> topology): 12 layers, hidden 768, 12 heads,
    /// intermediate 3072, vocab 30522.</summary>
    public static BertConfig GroundingDino => BaseUncased;
}
