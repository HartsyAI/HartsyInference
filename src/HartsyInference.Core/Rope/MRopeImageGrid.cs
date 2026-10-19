namespace HartsyInference.Core.Rope;

/// <summary>One image's token grid in the language model's sequence, after the vision tower's spatial merge: its run of
/// placeholder tokens is <c>T·H·W</c> long, raster ordered.</summary>
public readonly record struct MRopeImageGrid(int T, int H, int W)
{
    /// <summary>Placeholder tokens the image occupies.</summary>
    public int Tokens => T * H * W;
}
