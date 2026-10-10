using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Proves <see cref="SortHelpers.SortDescendingByValue"/> (the primitive two-array sort every fixed
/// call site below now shares) reproduces the EXACT permutation the pre-fix delegate-comparator
/// <c>Array.Sort(order, (a, b) =&gt; values[b].CompareTo(values[a]))</c> produced — the real correctness proof
/// for this change, the same role <c>TopPSortRefactorIdentityTests</c> plays for PR #215's LLM-package fix.
/// <see cref="OldDelegateSort"/> is kept verbatim as that reference. Every audio sampler fixed in this PR
/// (<c>NucleusSampler</c>'s fallback, <c>LogitSampling.SampleTopK</c>, <c>DiaPipeline.SampleDiaChannel</c>,
/// <c>BarkCausalStage.SampleTopPWithProb</c>) reduces to this one primitive, so this file is the load-bearing
/// test; the per-call-site tests alongside it exist to catch a wiring mistake at each site, not to re-derive
/// this proof.</summary>
public sealed class SortHelpersTests
{
    /// <summary>Verbatim pre-fix algorithm (every call site's old <c>ArgsortDescending</c>/inline sort), kept
    /// ONLY as the test oracle.</summary>
    private static int[] OldDelegateSort(float[] values, int count)
    {
        int[] idx = new int[count];
        for (int i = 0; i < count; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => values[b].CompareTo(values[a]));
        return idx;
    }

    private static int[] NewPrimitiveSort(float[] values, int count)
    {
        float[] copy = (float[])values.Clone();
        int[] order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        SortHelpers.SortDescendingByValue(copy, order, count);
        return order;
    }

    private static void AssertSamePermutation(float[] values, int count, string label)
    {
        int[] oldOrder = OldDelegateSort(values, count);
        int[] newOrder = NewPrimitiveSort(values, count);
        Assert.Equal(oldOrder.Length, newOrder.Length);
        for (int rank = 0; rank < oldOrder.Length; rank++)
        {
            Assert.True(oldOrder[rank] == newOrder[rank],
                $"{label} (n={count}): rank {rank} old={oldOrder[rank]} new={newOrder[rank]}");
        }
    }

    private static float[] RandomDistinct(int n, Random rng)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 10.0 - 5.0);
        return v;
    }

    private static float[] AllEqual(int n, float value = 1.0f)
    {
        float[] v = new float[n];
        Array.Fill(v, value);
        return v;
    }

    private static float[] LowCardinality(int n, int distinctValues, Random rng)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = rng.Next(distinctValues);
        return v;
    }

    private static float[] Increasing(int n)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = i;
        return v;
    }

    private static float[] Decreasing(int n)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = n - i;
        return v;
    }

    private static float[] OrganPipe(int n)
    {
        float[] v = new float[n];
        int mid = n / 2;
        for (int i = 0; i < n; i++) v[i] = i <= mid ? i : n - i;
        return v;
    }

    /// <summary>A classic median-of-three quicksort killer: repeatedly fools "median of first/middle/last"
    /// pivot selection, pushing the old delegate sort and the new primitive sort toward introsort's
    /// heapsort fallback where a comparator-dispatch difference would be most likely to surface.</summary>
    private static float[] MedianOfThreeKiller(int n)
    {
        float[] v = new float[n];
        int mid = n / 2;
        for (int i = 0; i < n; i++) v[i] = (i % 2 == 0) ? i / 2 : mid + (i / 2);
        return v;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(64)]
    [InlineData(102048)] // FishSpeech TextVocab
    public void MatchesOldDelegateSort_AcrossShapesAndSizes(int n)
    {
        Random rng = new(1000 + n);
        AssertSamePermutation(RandomDistinct(n, rng), n, "random-distinct");
        AssertSamePermutation(AllEqual(n), n, "all-equal");
        if (n >= 4) AssertSamePermutation(LowCardinality(n, 3, rng), n, "low-card-3");
        AssertSamePermutation(Increasing(n), n, "increasing");
        AssertSamePermutation(Decreasing(n), n, "decreasing");
        AssertSamePermutation(OrganPipe(n), n, "organ-pipe");
        AssertSamePermutation(MedianOfThreeKiller(n), n, "median3-killer");
    }

    /// <summary>Exact ties AT the rank boundary that matters most for a sampler: a handful of distinct
    /// "winner" values scattered through a sea of identical "loser" values, at several winner counts and
    /// several vocab sizes. This is the shape a tie-break difference would most likely be visible in.</summary>
    [Theory]
    [InlineData(10)]
    [InlineData(1028)]
    [InlineData(10001)]
    public void MatchesOldDelegateSort_AtExactTieBoundaries(int n)
    {
        for (int winners = 1; winners <= Math.Min(5, n); winners++)
        {
            float[] v = AllEqual(n, 0.0f);
            List<int> positions = [0, n / 2, n - 1];
            Random rng = new(2000 + n + winners);
            while (positions.Count < winners) positions.Add(rng.Next(n));
            for (int w = 0; w < winners; w++) v[positions[w % positions.Count]] = 1.0f + w * 0.001f;
            AssertSamePermutation(v, n, $"tie-boundary winners={winners}");
        }
    }

}
