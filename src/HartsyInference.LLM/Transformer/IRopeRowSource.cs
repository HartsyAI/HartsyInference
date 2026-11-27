namespace HartsyInference.LLM.Transformer;

/// <summary>Per-position cos and sin rows of an interleaved-pair rotary table, <c>HalfDim</c> entries each.</summary>
public interface IRopeRowSource
{
    /// <summary>Half the rotary width: one angle per element pair.</summary>
    int HalfDim { get; }

    /// <summary>Number of positions the table covers.</summary>
    int Length { get; }

    /// <summary>The cosines of position <paramref name="pos"/>.</summary>
    ReadOnlySpan<float> CosRow(int pos);

    /// <summary>The sines of position <paramref name="pos"/>.</summary>
    ReadOnlySpan<float> SinRow(int pos);
}
