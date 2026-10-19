namespace HartsyInference.Core.Rope;

/// <summary>How one per-pair cos/sin row is laid out for the apply op that reads it.</summary>
public enum RopeTableLayout
{
    /// <summary>Pairs <c>(i, i + half)</c>; <c>c[i]</c> and <c>c[i + half]</c> are read. Used by <c>IBackend.ApplyRope</c>.</summary>
    RotateHalf,

    /// <summary>Pairs <c>(2i, 2i + 1)</c>; <c>c[i]</c> is read from the first half. Used by <c>IBackend.ApplyRopeInterleaved</c>.</summary>
    InterleavedPrefix,

    /// <summary>Pairs <c>(2i, 2i + 1)</c> with duplicated values <c>c[2i] = c[2i + 1]</c>. Used by <c>IBackend.WanRopeInterleaved</c>.</summary>
    InterleavedPair,
}
