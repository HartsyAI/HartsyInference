namespace HartsyInference.Engine.Recipes;

/// <summary>The frame counts a video family can generate: <c>Step·n + Offset</c> from <see cref="Offset"/> up to
/// <see cref="Max"/>, and what planning does with a count off that grid.</summary>
public sealed record FrameGrid
{
    /// <summary>Creates a grid; the step and offset must be positive and a declared maximum must itself be on the grid.</summary>
    public FrameGrid(int step, int offset, FrameNormalization normalization, int? max = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(step, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(offset, 1);
        if (max is int cap && (cap < offset || (cap - offset) % step != 0))
        {
            throw new ArgumentOutOfRangeException(nameof(max), cap, $"Maximum must be a count on the {step}n+{offset} grid.");
        }
        Step = step;
        Offset = offset;
        Normalization = normalization;
        Max = max;
    }

    /// <summary>Spacing between valid counts; the VAE's temporal compression for <c>step·n + 1</c> families.</summary>
    public int Step { get; }

    /// <summary>The smallest valid count; every valid count is <c>Offset + Step·n</c>.</summary>
    public int Offset { get; }

    /// <summary>What planning does with a requested count off the grid.</summary>
    public FrameNormalization Normalization { get; }

    /// <summary>The largest count the family generates in one pass; null when it declares no cap.</summary>
    public int? Max { get; }

    /// <summary>The <c>step·n + 1</c> grid a causal video VAE with temporal compression <paramref name="step"/> encodes.</summary>
    public static FrameGrid StepPlusOne(int step, FrameNormalization normalization, int? max = null) =>
        new FrameGrid(step, 1, normalization, max);

    /// <summary>Whether <paramref name="frames"/> is a count this family generates without normalization.</summary>
    public bool Contains(int frames) =>
        frames >= Offset && (frames - Offset) % Step == 0 && (Max is null || frames <= Max);

    /// <summary>The count planning runs for a requested <paramref name="frames"/>: the request itself when it is on the
    /// grid, otherwise per <see cref="Normalization"/>, clamped to <see cref="Max"/>; null when the grid rejects it.</summary>
    public int? Normalize(int frames)
    {
        if (Contains(frames))
        {
            return frames;
        }
        if (Normalization == FrameNormalization.Reject)
        {
            return null;
        }
        double steps = (frames - (double)Offset) / Step;
        double n = Normalization == FrameNormalization.SnapUp ? Math.Ceiling(steps) : Math.Round(steps);
        long snapped = Offset + (long)Math.Max(0, n) * Step;
        // Snapping past the cap lands on the cap: it is on the grid and the closest count the family can generate.
        return (int)Math.Min(snapped, Max ?? int.MaxValue);
    }

    /// <inheritdoc/>
    public override string ToString() => Max is int cap ? $"{Step}n+{Offset} (max {cap})" : $"{Step}n+{Offset}";
}
