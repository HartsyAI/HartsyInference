using HartsyInference.LLM.Generation.Speculation;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculation;

public sealed class NGramSuffixDraftProviderTests
{
    [Fact]
    public void HandBuiltContext_PrefersTheLongestSuffixOverTheNearestShorterOne()
    {
        // Context: 1 2 3 4 9 2 3 8 1 2 3. The one-token suffix [3] and two-token suffix [2,3] both recur most recently
        // followed by 8, but the three-token suffix [1,2,3] recurs at index 0 and is followed by 4 9 2 3 8.
        NGramSuffixDraftProvider provider = new();
        int[] prompt = [1, 2, 3, 4, 9, 2, 3, 8, 1, 2, 3];
        Assert.Equal(new[] { 4, 9, 2, 3, 8 }, provider.Propose(prompt, new List<int>(), 5));
    }

    [Fact]
    public void HandBuiltContext_CapsTheDraftAtMaxDraftLength()
    {
        NGramSuffixDraftProvider provider = new();
        int[] prompt = [1, 2, 3, 4, 9, 2, 3, 8, 1, 2, 3];
        Assert.Equal(new[] { 4, 9 }, provider.Propose(prompt, new List<int>(), 2));
    }

    [Fact]
    public void MatchAcrossPromptAndGeneratedOutput_IsFound()
    {
        // Context 1 2 3 9 1 2: the suffix [1,2] occurs at index 0 and is followed by 3 9 1 2.
        NGramSuffixDraftProvider provider = new();
        int[] prompt = [1, 2, 3];
        List<int> generated = [9, 1, 2];
        Assert.Equal(new[] { 3, 9 }, provider.Propose(prompt, generated, 2));
    }

    [Fact]
    public void NoSuffixMatches_ProposesNothing()
    {
        // The last token 5 never occurred earlier, so no suffix of any length recurs.
        NGramSuffixDraftProvider provider = new();
        Assert.Empty(provider.Propose([1, 2, 3, 4, 5], new List<int>(), 8));
    }

    [Fact]
    public void SingleTokenContext_ProposesNothing()
    {
        NGramSuffixDraftProvider provider = new();
        Assert.Empty(provider.Propose([7], new List<int>(), 8));
    }

    [Fact]
    public void NonPositiveDraftLength_ProposesNothing()
    {
        NGramSuffixDraftProvider provider = new();
        Assert.Empty(provider.Propose([1, 2, 3, 1, 2, 3], new List<int>(), 0));
    }

    [Fact]
    public void LookbackExcludesAMatchOutsideTheWindow()
    {
        // With a lookback of 3, only the tail 4 5 1 ... is searchable, and the earlier 1 2 match is out of reach.
        NGramSuffixDraftProvider provider = new(maxNgram: 2, maxLookback: 3);
        int[] prompt = [1, 2, 7, 4, 5, 1];
        Assert.Empty(provider.Propose(prompt, new List<int>(), 4));
    }

    [Fact]
    public void Name_IsStable()
    {
        Assert.Equal("ngram-suffix", new NGramSuffixDraftProvider().Name);
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NGramSuffixDraftProvider(maxNgram: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NGramSuffixDraftProvider(maxLookback: 0));
    }
}
