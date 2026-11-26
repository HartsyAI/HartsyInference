namespace HartsyInference.Core.Layers.Attention;

/// <summary>Key and value rows <c>[Start, Start + Count)</c>.</summary>
public readonly record struct AttentionKeyRange(int Start, int Count)
{
    /// <summary>One past the last row.</summary>
    public int End => Start + Count;
}

/// <summary>One attention call over operands of different lengths: the queries in <c>[QueryStart, QueryStart + QueryCount)</c> attend the keys
/// and values of every range in <see cref="Keys"/> and nothing else. The keys may be longer than the queries, as a history followed by the
/// current rows is. <see cref="AttentionSegment"/> is the case of one range over one sequence.</summary>
public readonly record struct KeyRangeSegment(int QueryStart, int QueryCount, IReadOnlyList<AttentionKeyRange> Keys)
{
    /// <summary>Key rows this segment reads.</summary>
    public int KeyCount
    {
        get
        {
            int count = 0;
            for (int i = 0; i < Keys.Count; i++) count += Keys[i].Count;
            return count;
        }
    }

    /// <summary>The segment over one range.</summary>
    public static KeyRangeSegment Of(int queryStart, int queryCount, int keyStart, int keyCount) =>
        new(queryStart, queryCount, [new AttentionKeyRange(keyStart, keyCount)]);

    /// <summary>Throws unless <paramref name="segments"/> tile the query rows <c>[0, queryTokens)</c> in order, and every segment names at least
    /// one key row, inside <c>[0, keyTokens)</c>, in ascending ranges that do not overlap. A key named twice would count twice in the softmax.</summary>
    public static void Validate(ReadOnlySpan<KeyRangeSegment> segments, int queryTokens, int keyTokens)
    {
        if (segments.IsEmpty)
            throw new ArgumentException("An attention plan needs at least one segment.", nameof(segments));
        int next = 0;
        foreach (KeyRangeSegment s in segments)
        {
            if (s.QueryStart != next || s.QueryCount <= 0)
                throw new ArgumentException(
                    $"Segments must tile the query rows in order; expected a segment at row {next}, got [{s.QueryStart}, +{s.QueryCount}).", nameof(segments));
            if (s.Keys is null || s.Keys.Count == 0)
                throw new ArgumentException($"The segment at query row {s.QueryStart} reads no keys; its softmax would be undefined.", nameof(segments));
            int from = 0;
            for (int i = 0; i < s.Keys.Count; i++)
            {
                AttentionKeyRange range = s.Keys[i];
                if (range.Count <= 0 || range.Start < from || (long)range.Start + range.Count > keyTokens)
                    throw new ArgumentException(
                        $"The segment at query row {s.QueryStart} names keys [{range.Start}, +{range.Count}); its ranges must ascend, not overlap, and lie "
                        + $"inside {keyTokens} keys.", nameof(segments));
                from = range.End;
            }
            next += s.QueryCount;
        }
        if (next != queryTokens)
            throw new ArgumentException($"Segments cover {next} of {queryTokens} query rows.", nameof(segments));
    }
}
