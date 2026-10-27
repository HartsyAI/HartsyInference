using HartsyInference.Audio.Frontends;

namespace HartsyInference.Audio.Text;

/// <summary>Packs a passage into chunks of at most a word budget, cut only at sentence or clause boundaries.</summary>
public static class TtsTextSegmenter
{
    /// <summary>Splits text into chunks of at most <paramref name="maxWords"/> words for text-to-speech.</summary>
    /// <param name="text">The passage. Null, empty or whitespace yields an empty list.</param>
    /// <param name="maxWords">Largest chunk in words. Must be at least 1.</param>
    /// <returns>Chunks in order, each with words joined by single spaces. Every input word appears exactly once.</returns>
    /// <remarks>Consecutive sentences are merged while they fit. A sentence longer than the budget is cut at clause
    /// delimiters, and a clause still longer than the budget is cut into pieces of exactly <paramref name="maxWords"/>
    /// words; those pieces are never merged with neighbours.</remarks>
    public static IReadOnlyList<string> Pack(string text, int maxWords = 40)
    {
        if (maxWords < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxWords), maxWords, "maxWords must be at least 1");
        }
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        List<string> chunks = [];
        List<string> buffer = [];
        IReadOnlyList<string> sentences = SentenceSplitter.Split(text);
        foreach (string sentence in sentences)
        {
            string[] words = sentence.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length <= maxWords)
            {
                AppendUnit(words, maxWords, buffer, chunks);
                continue;
            }

            foreach (string[] clause in SplitClauses(words))
            {
                if (clause.Length <= maxWords)
                {
                    AppendUnit(clause, maxWords, buffer, chunks);
                    continue;
                }
                Flush(buffer, chunks);
                for (int start = 0; start < clause.Length; start += maxWords)
                {
                    int count = Math.Min(maxWords, clause.Length - start);
                    chunks.Add(string.Join(" ", clause, start, count));
                }
            }
        }
        Flush(buffer, chunks);
        return chunks;
    }

    /// <summary>Adds a unit of at most the budget to the open chunk, starting a new chunk when it would not fit.</summary>
    private static void AppendUnit(string[] unit, int maxWords, List<string> buffer, List<string> chunks)
    {
        if (buffer.Count + unit.Length > maxWords)
        {
            Flush(buffer, chunks);
        }
        buffer.AddRange(unit);
    }

    private static void Flush(List<string> buffer, List<string> chunks)
    {
        if (buffer.Count == 0)
        {
            return;
        }
        chunks.Add(string.Join(' ', buffer));
        buffer.Clear();
    }

    /// <summary>Groups words into clauses, closing each clause after a word that ends in a clause delimiter.</summary>
    private static List<string[]> SplitClauses(string[] words)
    {
        List<string[]> clauses = [];
        List<string> current = [];
        foreach (string word in words)
        {
            current.Add(word);
            if (IsClauseDelimiter(word[^1]))
            {
                clauses.Add(current.ToArray());
                current.Clear();
            }
        }
        if (current.Count > 0)
        {
            clauses.Add(current.ToArray());
        }
        return clauses;
    }

    private static bool IsClauseDelimiter(char c) => c is ',' or ';' or ':' or '—' or '–';
}
