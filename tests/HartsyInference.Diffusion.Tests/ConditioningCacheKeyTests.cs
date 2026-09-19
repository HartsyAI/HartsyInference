using Xunit;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The rule that stops a cached conditioning tensor being served to the wrong request once prompt
/// weighting exists.</summary>
/// <remarks>This is here because the failure it guards is invisible to the real-weight gates: every CLI
/// invocation is a fresh process with a cold cache, so the second-generation case never runs there. It would show
/// up only on a repeat of the same prompt inside one long-lived process — the SwarmUI service's normal mode.</remarks>
public sealed class ConditioningCacheKeyTests
{
    private static readonly int[] Ids = [7, 8, 9];

    /// <summary>The whole point. A weighted and an unweighted prompt tokenize to the SAME ids once the recipe has
    /// taken the emphasis off the text, so ids alone would hand one request the other's conditioning.</summary>
    [Fact]
    public void SameIdsWithDifferentEmphasisAreDifferentEntries()
    {
        Assert.False(ConditioningCacheKey.Matches(Ids, null, Ids, [1f, 0.5f, 1f]));
        Assert.False(ConditioningCacheKey.Matches(Ids, [1f, 0.5f, 1f], Ids, null));
        Assert.False(ConditioningCacheKey.Matches(Ids, [1f, 0.5f, 1f], Ids, [1f, 1.5f, 1f]));
    }

    /// <summary>A genuine repeat still hits, in both the weighted and unweighted directions — otherwise the cache
    /// would never pay for itself and every generation would re-run the encoder.</summary>
    [Fact]
    public void AGenuineRepeatStillHits()
    {
        Assert.True(ConditioningCacheKey.Matches(Ids, null, [7, 8, 9], null));
        Assert.True(ConditioningCacheKey.Matches(Ids, [1f, 0.5f, 1f], [7, 8, 9], [1f, 0.5f, 1f]));
    }

    /// <summary>Different prompts never share an entry, whatever their emphasis.</summary>
    [Fact]
    public void DifferentIdsNeverMatch()
    {
        Assert.False(ConditioningCacheKey.Matches(Ids, null, [7, 8, 10], null));
        Assert.False(ConditioningCacheKey.Matches(Ids, [1f, 0.5f, 1f], [7, 8, 10], [1f, 0.5f, 1f]));
    }

    /// <summary>An empty cache misses rather than throwing — the first generation of the process.</summary>
    [Fact]
    public void AColdCacheMisses()
    {
        Assert.False(ConditioningCacheKey.Matches(null, null, Ids, null));
        Assert.False(ConditioningCacheKey.Matches(null, null, Ids, [1f, 0.5f, 1f]));
    }

    /// <summary>Length alone does not decide it: a weight array that differs only in its last row is a different
    /// emphasis, and comparing prefixes would collapse the two.</summary>
    [Fact]
    public void EveryRowOfTheWeightArrayCounts()
    {
        Assert.False(ConditioningCacheKey.Matches(Ids, [1f, 1f, 0.5f], Ids, [1f, 1f, 0.6f]));
    }
}
