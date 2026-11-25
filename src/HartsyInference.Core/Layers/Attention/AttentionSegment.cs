namespace HartsyInference.Core.Layers.Attention;

/// <summary>One attention call over a self-attention sequence: the queries in <c>[QueryStart, QueryStart + QueryCount)</c> attend only the
/// keys and values in <c>[KeyStart, KeyStart + KeyCount)</c>; all counts are token rows.</summary>
public readonly record struct AttentionSegment(int QueryStart, int QueryCount, int KeyStart, int KeyCount)
{
    /// <summary>The segment that attends every token to every token.</summary>
    public static AttentionSegment Dense(int tokens) => new(0, tokens, 0, tokens);

    /// <summary>Throws unless <paramref name="segments"/> tile the query rows <c>[0, tokens)</c> in order and every key window
    /// lies inside the sequence.</summary>
    public static void Validate(ReadOnlySpan<AttentionSegment> segments, int tokens)
    {
        if (segments.IsEmpty)
            throw new ArgumentException("An attention plan needs at least one segment.", nameof(segments));
        int next = 0;
        foreach (AttentionSegment s in segments)
        {
            if (s.QueryStart != next || s.QueryCount <= 0)
                throw new ArgumentException(
                    $"Segments must tile the query rows in order; expected a segment at row {next}, got {s}.", nameof(segments));
            if (s.KeyStart < 0 || s.KeyCount <= 0 || (long)s.KeyStart + s.KeyCount > tokens)
                throw new ArgumentException($"Segment {s} has a key window outside the {tokens}-token sequence.", nameof(segments));
            next += s.QueryCount;
        }
        if (next != tokens)
            throw new ArgumentException($"Segments cover {next} of {tokens} query rows.", nameof(segments));
    }
}
