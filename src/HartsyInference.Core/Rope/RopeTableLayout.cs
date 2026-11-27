namespace HartsyInference.Core.Rope;

/// <summary>How one per-pair cos/sin row is laid out for the apply op that reads it.</summary>
public enum RopeTableLayout
{
    /// <summary>Pairs <c>(i, i + half)</c>; <c>c[i]</c> and <c>c[i + half]</c> are read. Read by <c>RotaryViews.Create</c> as an entry per element.</summary>
    RotateHalf,

    /// <summary>Pairs <c>(2i, 2i + 1)</c>; <c>c[i]</c> is read from the first half. Read by <c>RotaryViews.Create</c> as one entry per pair.</summary>
    InterleavedPrefix,

    /// <summary>Pairs <c>(2i, 2i + 1)</c> with duplicated values <c>c[2i] = c[2i + 1]</c>. Read by <c>RotaryViews.Create</c> at every even index.</summary>
    InterleavedPair,
}
