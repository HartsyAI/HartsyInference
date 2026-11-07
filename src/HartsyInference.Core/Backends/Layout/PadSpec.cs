namespace HartsyInference.Core.Backends.Layout;

/// <summary>Per-axis padding for <see cref="IBackend.Pad"/>: how many positions go before and after the input on each
/// axis, and one <see cref="PadMode"/> for every padded position.</summary>
public readonly struct PadSpec
{
    private readonly StridedExtents _before;
    private readonly StridedExtents _after;

    /// <summary>Pads every axis by <paramref name="before"/> and <paramref name="after"/> positions (one entry per axis, zero for an unpadded axis).</summary>
    public PadSpec(PadMode mode, ReadOnlySpan<long> before, ReadOnlySpan<long> after)
    {
        if (before.Length < 1 || before.Length > StridedView.MaxRank)
            throw new ArgumentException($"A pad spec covers 1 to {StridedView.MaxRank} axes; got {before.Length}.", nameof(before));
        if (after.Length != before.Length)
            throw new ArgumentException($"A pad spec needs the same number of before and after entries; got {before.Length} and {after.Length}.", nameof(after));
        for (int axis = 0; axis < before.Length; axis++)
        {
            if (before[axis] < 0 || after[axis] < 0)
                throw new ArgumentOutOfRangeException(nameof(before), $"Axis {axis} pads by ({before[axis]}, {after[axis]}); padding must not be negative.");
        }
        Mode = mode;
        Rank = before.Length;
        before.CopyTo(_before);
        after.CopyTo(_after);
    }

    /// <summary>How the padded positions are filled.</summary>
    public PadMode Mode { get; }

    /// <summary>Number of axes.</summary>
    public int Rank { get; }

    /// <summary>Positions added before the input on <paramref name="axis"/>.</summary>
    public long Before(int axis)
    {
        CheckAxis(axis);
        return _before[axis];
    }

    /// <summary>Positions added after the input on <paramref name="axis"/>.</summary>
    public long After(int axis)
    {
        CheckAxis(axis);
        return _after[axis];
    }

    /// <summary>Pads one axis of a rank-<paramref name="rank"/> tensor and leaves the others alone.</summary>
    public static PadSpec Axis(PadMode mode, int rank, int axis, long before, long after)
    {
        if ((uint)axis >= (uint)rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis must be in [0,{rank}).");
        Span<long> b = stackalloc long[rank];
        Span<long> a = stackalloc long[rank];
        b[axis] = before;
        a[axis] = after;
        return new PadSpec(mode, b, a);
    }

    public override string ToString()
    {
        long[] before = new long[Rank];
        long[] after = new long[Rank];
        ((ReadOnlySpan<long>)_before)[..Rank].CopyTo(before);
        ((ReadOnlySpan<long>)_after)[..Rank].CopyTo(after);
        return $"{Mode} before [{string.Join(",", before)}] after [{string.Join(",", after)}]";
    }

    private void CheckAxis(int axis)
    {
        if ((uint)axis >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis must be in [0,{Rank}).");
    }
}
