using System.Text.RegularExpressions;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>One HF <c>Split</c> pre-tokenizer stage: a regex, a delimiter behavior and the invert flag.</summary>
internal sealed class SplitStage
{
    private readonly Regex _regex;
    private readonly SplitBehavior _behavior;
    private readonly bool _invert;

    public SplitStage(string pattern, SplitBehavior behavior, bool invert)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        _regex = new Regex(pattern, RegexOptions.Compiled | RegexOptions.CultureInvariant);
        _behavior = behavior;
        _invert = invert;
    }

    /// <summary>Splits every piece of <paramref name="pieces"/> (offsets into <paramref name="text"/>) and appends the result to <paramref name="output"/>.</summary>
    public void Apply(string text, List<(int Start, int End)> pieces, List<(int Start, int End)> output)
    {
        List<(int Start, int End, bool IsMatch)> spans = [];
        List<(int Start, int End)> kept = [];
        foreach ((int start, int end) in pieces)
        {
            spans.Clear();
            FindSpans(text, start, end, spans);
            ApplyBehavior(spans, kept);
            output.AddRange(kept);
            kept.Clear();
        }
    }

    // Spans tile [start, end) exactly; empty regex matches are skipped like HF's find_matches.
    private void FindSpans(string text, int start, int end, List<(int Start, int End, bool IsMatch)> spans)
    {
        ReadOnlySpan<char> piece = text.AsSpan(start, end - start);
        int previous = 0;
        foreach (ValueMatch m in _regex.EnumerateMatches(piece))
        {
            if (m.Length == 0) continue;
            if (m.Index > previous) spans.Add((start + previous, start + m.Index, _invert));
            spans.Add((start + m.Index, start + m.Index + m.Length, !_invert));
            previous = m.Index + m.Length;
        }
        if (previous < end - start) spans.Add((start + previous, end, _invert));
    }

    // Mirrors tokenizers' NormalizedString::split: merge behaviors only fuse a match into a neighbour when the
    // preceding span (in fold direction) is not itself a match.
    private void ApplyBehavior(List<(int Start, int End, bool IsMatch)> spans, List<(int Start, int End)> kept)
    {
        switch (_behavior)
        {
            case SplitBehavior.Removed:
                foreach ((int s, int e, bool isMatch) in spans)
                    if (!isMatch) kept.Add((s, e));
                break;
            case SplitBehavior.Isolated:
                foreach ((int s, int e, bool _) in spans) kept.Add((s, e));
                break;
            case SplitBehavior.MergedWithPrevious:
                MergeForward(spans, kept);
                break;
            case SplitBehavior.MergedWithNext:
                MergeBackward(spans, kept);
                break;
            case SplitBehavior.Contiguous:
                Contiguous(spans, kept);
                break;
            default:
                throw new NotSupportedException($"Split behavior {_behavior} is not supported.");
        }
    }

    private static void MergeForward(List<(int Start, int End, bool IsMatch)> spans, List<(int Start, int End)> kept)
    {
        bool previousMatch = false;
        foreach ((int s, int e, bool isMatch) in spans)
        {
            if (isMatch && !previousMatch && kept.Count > 0) kept[^1] = (kept[^1].Start, e);
            else kept.Add((s, e));
            previousMatch = isMatch;
        }
    }

    private static void MergeBackward(List<(int Start, int End, bool IsMatch)> spans, List<(int Start, int End)> kept)
    {
        bool previousMatch = false;
        List<(int Start, int End)> reversed = new(spans.Count);
        for (int i = spans.Count - 1; i >= 0; i--)
        {
            (int s, int e, bool isMatch) = spans[i];
            if (isMatch && !previousMatch && reversed.Count > 0) reversed[^1] = (s, reversed[^1].End);
            else reversed.Add((s, e));
            previousMatch = isMatch;
        }
        for (int i = reversed.Count - 1; i >= 0; i--) kept.Add(reversed[i]);
    }

    private static void Contiguous(List<(int Start, int End, bool IsMatch)> spans, List<(int Start, int End)> kept)
    {
        bool previousMatch = false;
        foreach ((int s, int e, bool isMatch) in spans)
        {
            if (isMatch == previousMatch && kept.Count > 0) kept[^1] = (kept[^1].Start, e);
            else kept.Add((s, e));
            previousMatch = isMatch;
        }
    }
}
