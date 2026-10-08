namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>Drafts by n-gram lookup: the continuation of the most recent earlier occurrence of the context's longest matching suffix. Deterministic, so <see cref="DraftBlock.Probs"/> is null.</summary>
public sealed class PromptLookupProposer : IDraftProposer
{
    private readonly int _maxNgram;

    public PromptLookupProposer(int maxNgram = 3)
    {
        if (maxNgram < 1) throw new ArgumentOutOfRangeException(nameof(maxNgram));
        _maxNgram = maxNgram;
    }

    public DraftBlock Propose(ReadOnlySpan<int> context, int maxTokens)
    {
        if (maxTokens <= 0) return new DraftBlock([], null);
        for (int n = Math.Min(_maxNgram, context.Length - 1); n >= 1; n--)
        {
            int suffix = context.Length - n;
            // the most recent earlier occurrence of the suffix whose continuation starts before the suffix itself
            for (int start = suffix - 1; start >= 0; start--)
            {
                if (!context.Slice(start, n).SequenceEqual(context.Slice(suffix, n))) continue;
                int from = start + n, count = Math.Min(maxTokens, context.Length - from);
                if (count <= 0) continue;
                return new DraftBlock(context.Slice(from, count).ToArray(), null);
            }
        }
        return new DraftBlock([], null);
    }
}
