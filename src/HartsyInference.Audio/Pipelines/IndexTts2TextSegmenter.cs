namespace HartsyInference.Audio.Pipelines;

/// <summary>Token-level text segmentation for IndexTTS-2.0 — a literal port of the reference
/// <c>TextTokenizer.split_segments_by_token</c> / <c>split_segments</c> (<c>indextts/utils/front.py</c>). The
/// reference tokenizes the WHOLE text once, then cuts the piece list at sentence-ending punctuation pieces
/// (<c>. ! ? ▁. ▁? ▁...</c>), recursively at commas and hyphens, and finally merges short neighbours back together,
/// so the segments (and therefore the GPT's per-segment prosody) match the Python implementation exactly. Quirks of
/// the original are kept on purpose: an apostrophe following a sentence end is appended to that segment AND starts the
/// next one (the reference's <c>i += 1</c> inside a <c>for</c> loop has no effect), and the merge rules use
/// <c>max/2</c> as a float.</summary>
internal static class IndexTts2TextSegmenter
{
    /// <summary>One tokenizer piece (<c>▁</c>-prefixed word starts included) and its id.</summary>
    internal readonly record struct Token(string Piece, int Id);

    private static readonly string[] SentenceEnds = [".", "!", "?", "▁.", "▁?", "▁..."];
    private static readonly string[] CommaTokens = [",", "▁,"];
    private static readonly string[] HyphenTokens = ["-"];
    private const int MaxRecursion = 32;

    /// <summary><c>tokenizer.split_segments(tokens, max_text_tokens_per_segment, quick_streaming_tokens)</c>.</summary>
    /// <param name="quickStreamingTokens">While fewer than this many tokens have been consumed, short neighbouring
    /// segments are NOT merged — a small first segment, so streaming consumers hear audio sooner. 0 disables it.</param>
    public static List<Token[]> Split(IReadOnlyList<Token> tokens, int maxTokensPerSegment, int quickStreamingTokens = 0)
    {
        List<List<Token>> segments = SplitByToken([.. tokens], SentenceEnds, maxTokensPerSegment, quickStreamingTokens, 0);
        List<Token[]> result = new(segments.Count);
        foreach (List<Token> s in segments) result.Add([.. s]);
        return result;
    }

    private static bool Contains(string[] set, string piece)
    {
        foreach (string s in set) if (s == piece) return true;
        return false;
    }

    private static bool AnyOf(List<Token> segment, string[] set)
    {
        foreach (Token t in segment) if (Contains(set, t.Piece)) return true;
        return false;
    }

    private static List<List<Token>> SplitByToken(List<Token> tokens, string[] splitTokens, int max, int quick, int depth)
    {
        List<List<Token>> segments = [];
        if (tokens.Count == 0) return segments;

        List<Token> current = [];
        bool splitHasComma = Contains(splitTokens, ",") || Contains(splitTokens, "▁,");
        bool splitHasHyphen = Contains(splitTokens, "-");
        for (int i = 0; i < tokens.Count; i++)
        {
            Token token = tokens[i];
            current.Add(token);

            List<List<Token>> sub;
            if (!splitHasComma && AnyOf(current, CommaTokens) && depth < MaxRecursion)
            {
                sub = SplitByToken(current, CommaTokens, max, quick, depth + 1);
            }
            else if (!splitHasHyphen && AnyOf(current, HyphenTokens) && depth < MaxRecursion)
            {
                sub = SplitByToken(current, HyphenTokens, max, quick, depth + 1);
            }
            else if (current.Count <= max)
            {
                if (Contains(splitTokens, token.Piece) && current.Count > 2)
                {
                    if (i < tokens.Count - 1 && (tokens[i + 1].Piece == "'" || tokens[i + 1].Piece == "▁'"))
                    {
                        // Reference quirk: the apostrophe is appended here and ALSO seen again as the next token.
                        current.Add(tokens[i + 1]);
                    }
                    segments.Add(current);
                    current = [];
                }
                continue;
            }
            else
            {
                sub = [];
                for (int j = 0; j < current.Count; j += max)
                    sub.Add(current.GetRange(j, Math.Min(max, current.Count - j)));
            }

            segments.AddRange(sub);
            current = [];
        }
        if (current.Count > 0) segments.Add(current);

        // Merge adjacent short segments (again a literal port, including the float `max / 2`).
        List<List<Token>> merged = [];
        int total = 0;
        foreach (List<Token> segment in segments)
        {
            total += segment.Count;
            if (segment.Count == 0) continue;
            if (merged.Count == 0) merged.Add(segment);
            else if (merged[^1].Count + segment.Count <= max && total > quick) merged[^1].AddRange(segment);
            else if (merged[^1].Count + segment.Count <= max / 2.0) merged[^1].AddRange(segment);
            else merged.Add(segment);
        }
        return merged;
    }
}
