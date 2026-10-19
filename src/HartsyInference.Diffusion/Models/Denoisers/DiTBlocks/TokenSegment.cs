namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>A contiguous run of tokens in a DiT sequence that shares one modulation group (one row of a
/// <see cref="SegmentModulation"/> table).</summary>
public readonly record struct TokenSegment(int Start, int Length, int Group)
{
    /// <summary>One past the last token.</summary>
    public int Stop => Start + Length;
}
