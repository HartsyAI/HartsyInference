using HartsyInference.Core.Moe;
using Xunit;

namespace HartsyInference.LLM.Tests.Moe;

/// <summary>Checks the expert-parallel dispatch against a single-device MoE evaluation on the same routes.</summary>
public class ExpertParallelDispatchTests
{
    private const float Tol = 1e-4f;

    /// <summary>Deterministic per-expert linear map: y = (e+1) * x + e (elementwise), so every expert differs.</summary>
    private static void Expert(int expert, ReadOnlySpan<float> x, Span<float> y)
    {
        for (int j = 0; j < x.Length; j++) y[j] = (expert + 1) * x[j] + expert;
    }

    [Fact]
    public void OwnerRank_SplitsExpertsContiguously()
    {
        // 8 experts over 4 ranks -> 2 per rank: experts 0,1 -> rank 0; 6,7 -> rank 3.
        Assert.Equal(0, ExpertParallelDispatch.OwnerRank(0, 8, 4));
        Assert.Equal(0, ExpertParallelDispatch.OwnerRank(1, 8, 4));
        Assert.Equal(2, ExpertParallelDispatch.OwnerRank(5, 8, 4));
        Assert.Equal(3, ExpertParallelDispatch.OwnerRank(7, 8, 4));
    }

    [Fact]
    public void ExpertsPerRank_RejectsUnevenSplit()
    {
        Assert.Throws<ArgumentException>(() => ExpertParallelDispatch.ExpertsPerRank(10, 4));
    }

    [Fact]
    public void Plan_CountsEveryRoutedPairOnce()
    {
        // 6 tokens, top-2, 8 experts, 4 ranks. Routes chosen so experts from every rank appear.
        int[] routes = { 0, 7, 1, 2, 6, 5, 3, 4, 7, 0, 2, 2 };
        int tokens = 6, topK = 2;
        IReadOnlyList<IReadOnlyList<ExpertParallelDispatch.Entry>> plan = ExpertParallelDispatch.Plan(tokens, 8, 4, routes, topK);

        int total = plan.Sum(list => list.Count);
        Assert.Equal(tokens * topK, total);
        // Each (token, slot) appears exactly once across destinations.
        HashSet<(int, int)> seen = new();
        foreach (IReadOnlyList<ExpertParallelDispatch.Entry> list in plan)
        {
            foreach (ExpertParallelDispatch.Entry e in list) Assert.True(seen.Add((e.Token, e.Slot)));
        }
        Assert.Equal(tokens * topK, seen.Count);
    }

    [Fact]
    public void Plan_SendCountsMatchEntryOwnership()
    {
        int[] routes = { 0, 7, 1, 2, 6, 5, 3, 4, 7, 0, 2, 2 };
        IReadOnlyList<IReadOnlyList<ExpertParallelDispatch.Entry>> plan = ExpertParallelDispatch.Plan(6, 8, 4, routes, 2);
        int[,] counts = ExpertParallelDispatch.SendCounts(plan, 4);
        int sum = 0;
        for (int s = 0; s < 4; s++)
        {
            for (int d = 0; d < 4; d++) sum += counts[s, d];
        }
        Assert.Equal(12, sum);
        // Source rank 0 owns tokens 0-1 (chunk of ceil(6/4)=2). Its pairs: experts 0,7,1,2 -> ranks 0,3,0,1.
        Assert.Equal(2, counts[0, 0]);
        Assert.Equal(1, counts[0, 1]);
        Assert.Equal(1, counts[0, 3]);
    }

    [Fact]
    public void Forward_MatchesSingleDeviceEvaluation()
    {
        // 5 tokens, dim 3, top-2, 8 experts over 4 ranks. Weights vary per slot.
        const int tokens = 5, dim = 3, topK = 2, experts = 8, ranks = 4;
        float[] hidden = new float[tokens * dim];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = 0.25f * i - 1f;
        int[] routes = { 1, 4, 3, 3, 7, 0, 5, 2, 6, 6 };
        float[] weights = { 0.7f, 0.3f, 0.5f, 0.5f, 0.9f, 0.1f, 0.2f, 0.8f, 0.4f, 0.6f };

        float[] ep = ExpertParallelDispatch.Forward(tokens, dim, experts, ranks, topK, hidden, routes, weights, Expert);

        // Single-device reference: apply each routed expert directly, combine in slot order.
        float[] reference = new float[tokens * dim];
        float[] y = new float[dim];
        for (int t = 0; t < tokens; t++)
        {
            for (int s = 0; s < topK; s++)
            {
                Expert(routes[t * topK + s], hidden.AsSpan(t * dim, dim), y);
                for (int j = 0; j < dim; j++) reference[t * dim + j] += weights[t * topK + s] * y[j];
            }
        }
        for (int i = 0; i < reference.Length; i++) Assert.Equal(reference[i], ep[i], Tol);
    }

    [Fact]
    public void Forward_EachRankOnlyComputesItsOwnExperts()
    {
        const int tokens = 4, dim = 1, topK = 1, experts = 4, ranks = 2;
        float[] hidden = { 1f, 2f, 3f, 4f };
        int[] routes = { 0, 3, 1, 2 };
        float[] weights = { 1f, 1f, 1f, 1f };
        List<int> computed = new();
        ExpertParallelDispatch.Forward(tokens, dim, experts, ranks, topK, hidden, routes, weights,
            (int expert, ReadOnlySpan<float> x, Span<float> y) =>
            {
                computed.Add(expert);
                Expert(expert, x, y);
            });
        // Every routed expert is computed exactly once, in rank order: rank 0 owns 0,1 and rank 1 owns 2,3.
        computed.Sort();
        Assert.Equal(new[] { 0, 1, 2, 3 }, computed);
    }
}
