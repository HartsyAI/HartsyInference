namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>ZipVoice's long-text chunking: upstream <c>chunk_tokens_punctuation</c>, which splits the token stream after
/// sentence punctuation and packs the sentences into chunks of at most a token budget.</summary>
public static class ZipVoiceTextChunker
{
    /// <summary>The marks upstream treats as ending a sentence, ASCII and full-width.</summary>
    public const string SentencePunctuation = ";:,.!?；：，。！？";

    /// <summary>Splits <paramref name="tokens"/> into sentences, then merges consecutive sentences while the chunk stays
    /// within <paramref name="maxTokens"/>; a sentence longer than the budget is kept whole, and a budget below one puts
    /// every sentence in its own chunk.</summary>
    public static List<List<string>> Chunk(IReadOnlyList<string> tokens, int maxTokens)
    {
        List<List<string>> chunks = new();
        List<string> current = new();
        foreach (List<string> sentence in SplitSentences(tokens))
        {
            if (current.Count + sentence.Count <= maxTokens)
            {
                current.AddRange(sentence);
                continue;
            }
            if (current.Count > 0) chunks.Add(current);
            current = sentence;
        }
        if (current.Count > 0) chunks.Add(current);
        return chunks;
    }

    private static List<List<string>> SplitSentences(IReadOnlyList<string> tokens)
    {
        List<List<string>> sentences = new();
        List<string> sentence = new();
        foreach (string token in tokens)
        {
            if (sentence.Count == 0 && sentences.Count > 0 && (IsPunctuation(token) || token == " "))
            {
                sentences[^1].Add(token);
                continue;
            }
            sentence.Add(token);
            if (!IsPunctuation(token)) continue;
            sentences.Add(sentence);
            sentence = new List<string>();
        }
        if (sentence.Count > 0) sentences.Add(sentence);
        return sentences;
    }

    private static bool IsPunctuation(string token) => token.Length == 1 && SentencePunctuation.Contains(token[0]);
}
