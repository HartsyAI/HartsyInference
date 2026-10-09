using HartsyInference.LLM.Generation.Speculative;

namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>Suffix n-gram drafting: matches the longest recent suffix of the context, from <c>maxNgram</c> tokens down to one,
/// and drafts the continuation of its most recent earlier occurrence.</summary>
/// <remarks>Unlike <see cref="PromptLookupDraftProvider"/>, which uses one fixed n-gram size, this tries several sizes
/// and prefers the longest match, so a longer repeated phrase beats a short coincidental one.
/// The matching is delegated to <see cref="PromptLookupProposer"/>, the existing generic proposer, so the logic is not
/// duplicated here. Proposes nothing when no suffix of length one or more occurs earlier in the window.</remarks>
public sealed class NGramSuffixDraftProvider : ISpeculativeDraftProvider
{
    /// <summary>The longest suffix tried.</summary>
    public const int DefaultMaxNgram = 4;

    private readonly PromptLookupProposer _proposer;
    private readonly int _maxLookback;

    public NGramSuffixDraftProvider(int maxNgram = DefaultMaxNgram, int maxLookback = PromptLookupDraftProvider.DefaultMaxLookback)
    {
        if (maxNgram < 1) throw new ArgumentOutOfRangeException(nameof(maxNgram), maxNgram, "The longest suffix must be at least 1 token.");
        if (maxLookback < 1) throw new ArgumentOutOfRangeException(nameof(maxLookback), maxLookback, "The lookback must be at least 1 token.");
        _proposer = new PromptLookupProposer(maxNgram);
        _maxLookback = maxLookback;
    }

    /// <inheritdoc />
    public string Name => "ngram-suffix";

    /// <inheritdoc />
    public int[] Propose(int[] promptIds, IReadOnlyList<int> generated, int maxDraftLen)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(generated);
        if (maxDraftLen <= 0) return [];
        int[] context = SpeculationWindow.Tail(promptIds, generated, _maxLookback);
        return _proposer.Propose(context, maxDraftLen).Tokens;
    }
}
