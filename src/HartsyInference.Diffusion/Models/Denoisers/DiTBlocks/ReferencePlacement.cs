namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Where a reference's tokens sit on a model's RoPE axes, relative to the target grid: the frame index <c>t</c>, and a
/// row and a column offset.</summary>
/// <remarks>The row offset is a fraction of the target's patched height, rounded half to even (Python's <c>round</c>, which the
/// ComfyUI reference adapter applies). Shifting the reference past the target grid is what keeps it from being copied
/// pixel-aligned: the model reads it as appearance, not as a pose.</remarks>
public readonly record struct ReferencePlacement
{
    /// <summary>Creates a placement; <paramref name="tFrameIndex"/> and <paramref name="columnOffset"/> are non-negative.</summary>
    public ReferencePlacement(int tFrameIndex, float rowOffsetHeights, int columnOffset)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tFrameIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(columnOffset);
        if (!float.IsFinite(rowOffsetHeights) || rowOffsetHeights < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(rowOffsetHeights), rowOffsetHeights, "A row offset must be finite and non-negative.");
        }
        TFrameIndex = tFrameIndex;
        RowOffsetHeights = rowOffsetHeights;
        ColumnOffset = columnOffset;
    }

    /// <summary>The frame index on the RoPE time axis.</summary>
    public int TFrameIndex { get; }

    /// <summary>The row offset in units of the target's patched height.</summary>
    public float RowOffsetHeights { get; }

    /// <summary>The column offset in patched tokens.</summary>
    public int ColumnOffset { get; }

    /// <summary>The row offset in patched tokens for a target whose patched height is <paramref name="targetPatchedHeight"/>.</summary>
    public int RowOffset(int targetPatchedHeight) =>
        (int)Math.Round((double)RowOffsetHeights * targetPatchedHeight, MidpointRounding.ToEven);
}
