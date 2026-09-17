namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>Shape of SheetSage2's score decoder, as the released checkpoint carries it.
///
/// <para>The decoder is post-LN, not the pre-LN arrangement every other encoder-decoder in this tree uses: a
/// layer normalises the residual SUM after each sub-block instead of its input, and there is no final norm after
/// the stack. Nothing errors when that is inverted — the decode simply writes a plausible wrong transcription.</para>
///
/// <para>The vocabulary is deliberately absent here. It is <see cref="ScoreTokenizer.TokenCount"/>, which is
/// derived from the field layout rather than written down, and a second copy of the number could disagree
/// with the one the output projection was trained against.</para></summary>
public sealed record SheetSage2Config
{
    /// <summary>Model width.</summary>
    public int Dim { get; init; } = 512;

    /// <summary>Feed-forward inner width.</summary>
    public int IntermediateSize { get; init; } = 2_048;

    /// <summary>Attention heads; head dim is <see cref="Dim"/> / this.</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Decoder layer count.</summary>
    public int Layers { get; init; } = 6;

    /// <summary>Longest token sequence one window may decode.</summary>
    public int MaxTokens { get; init; } = 5_120;

    /// <summary>Token index <c>i</c> reads position embedding <c>i + 2</c>, which is why the table is two rows
    /// longer than <see cref="MaxTokens"/>. The two skipped rows are never addressed.</summary>
    public int PositionOffset { get; init; } = 2;

    /// <summary>Memory tokens the cross-attention reads. Fixed: the encoder always sees a padded 300-second window.</summary>
    public int EncoderMemoryTokens { get; init; } = 7_500;

    /// <summary>Layer-norm epsilon.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Per-head width.</summary>
    public int HeadDim => Dim / NumHeads;

    /// <summary>Rows the position table must have.</summary>
    public int PositionRows => MaxTokens + PositionOffset;

    /// <summary>Attention scale, shared by self- and cross-attention.</summary>
    public float AttentionScale => 1f / MathF.Sqrt(HeadDim);

    /// <summary>The released configuration — the only one the checkpoint fits.</summary>
    public static SheetSage2Config Released => new();
}
