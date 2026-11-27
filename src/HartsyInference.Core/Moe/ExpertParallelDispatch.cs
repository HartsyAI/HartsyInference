namespace HartsyInference.Core.Moe;

/// <summary>
/// Host reference for expert-parallel MoE dispatch: experts are split into contiguous blocks across <c>Ranks</c>
/// devices, tokens are split across the same ranks as sources, every (token, slot) pair is sent to the rank that owns
/// its expert, computed there, and sent back to be combined with its routing weight. This is the exact data movement a
/// device all-to-all performs, run in one process so the routing, counting and combine logic can be verified on any host.
/// It is the oracle for the NCCL/peer-copy path; it does not model transfer cost.
/// </summary>
/// <summary>One expert applied to one row: <c>y = expert(x)</c>.</summary>
public delegate void ExpertKernel(int expert, ReadOnlySpan<float> x, Span<float> y);

public static class ExpertParallelDispatch
{
    /// <summary>One routed pair that crosses from its source rank to its expert's rank.</summary>
    public readonly record struct Entry(int SourceRank, int Token, int Slot, int Expert);

    /// <summary>Contiguous expert placement: rank <c>r</c> owns experts <c>[r*E/R, (r+1)*E/R)</c>.</summary>
    /// <exception cref="ArgumentException">The expert count does not divide evenly across the ranks.</exception>
    public static int ExpertsPerRank(int experts, int ranks)
    {
        if (ranks <= 0) throw new ArgumentOutOfRangeException(nameof(ranks), "At least one rank is required.");
        if (experts <= 0 || experts % ranks != 0)
            throw new ArgumentException($"{experts} experts do not divide evenly across {ranks} ranks; expert parallelism needs an even split.", nameof(ranks));
        return experts / ranks;
    }

    /// <summary>The rank that owns a global expert id.</summary>
    public static int OwnerRank(int expert, int experts, int ranks) => expert / ExpertsPerRank(experts, ranks);

    /// <summary>Source rank of a token: tokens are split into contiguous chunks of <c>ceil(tokens / ranks)</c>.</summary>
    public static int SourceRank(int token, int tokens, int ranks)
    {
        int perRank = (tokens + ranks - 1) / ranks;
        return token / perRank;
    }

    /// <summary>
    /// The receive layout of every destination rank: the entries it must compute, in a deterministic order (source rank,
    /// then token, then slot). Index <c>r</c> of the result is the list for rank <c>r</c>.
    /// </summary>
    public static IReadOnlyList<IReadOnlyList<Entry>> Plan(int tokens, int experts, int ranks, ReadOnlySpan<int> routes, int topK)
    {
        ExpertsPerRank(experts, ranks);
        if (routes.Length != tokens * topK) throw new ArgumentException($"routes must hold tokens*topK = {tokens * topK} expert ids.", nameof(routes));
        List<Entry>[] perDest = new List<Entry>[ranks];
        for (int r = 0; r < ranks; r++) perDest[r] = new List<Entry>();
        for (int src = 0; src < ranks; src++)
        {
            for (int t = 0; t < tokens; t++)
            {
                if (SourceRank(t, tokens, ranks) != src) continue;
                for (int s = 0; s < topK; s++)
                {
                    int expert = routes[t * topK + s];
                    if ((uint)expert >= (uint)experts) throw new ArgumentOutOfRangeException(nameof(routes), $"expert id {expert} is outside [0, {experts}).");
                    perDest[OwnerRank(expert, experts, ranks)].Add(new Entry(src, t, s, expert));
                }
            }
        }
        IReadOnlyList<Entry>[] result = new IReadOnlyList<Entry>[ranks];
        for (int r = 0; r < ranks; r++) result[r] = perDest[r];
        return result;
    }

    /// <summary>
    /// Runs the expert-parallel forward. <paramref name="hidden"/> is <c>[tokens, dim]</c>; <paramref name="weights"/> and
    /// <paramref name="routes"/> are <c>[tokens, topK]</c>. <paramref name="expertFn"/> computes one expert's output for one
    /// input row (<c>y = expert(x)</c>, both <c>dim</c> wide) and is called only on the owning rank's entries.
    /// Returns <c>out[t] = Σ_s weights[t,s] · expert_{routes[t,s]}(hidden[t])</c>, accumulated in the plan's order.
    /// </summary>
    public static float[] Forward(int tokens, int dim, int experts, int ranks, int topK, ReadOnlySpan<float> hidden,
        ReadOnlySpan<int> routes, ReadOnlySpan<float> weights, ExpertKernel expertFn)
    {
        if (hidden.Length != tokens * dim) throw new ArgumentException($"hidden must hold tokens*dim = {tokens * dim} values.", nameof(hidden));
        if (weights.Length != tokens * topK) throw new ArgumentException($"weights must hold tokens*topK = {tokens * topK} values.", nameof(weights));
        IReadOnlyList<IReadOnlyList<Entry>> plan = Plan(tokens, experts, ranks, routes, topK);
        float[] output = new float[tokens * dim];
        float[] y = new float[dim];
        for (int r = 0; r < ranks; r++)
        {
            foreach (Entry e in plan[r])
            {
                expertFn(e.Expert, hidden.Slice(e.Token * dim, dim), y);
                float w = weights[e.Token * topK + e.Slot];
                for (int j = 0; j < dim; j++) output[e.Token * dim + j] += w * y[j];
            }
        }
        return output;
    }

    /// <summary>How many routed pairs travel from each source rank to each destination rank (the all-to-all send matrix).</summary>
    public static int[,] SendCounts(IReadOnlyList<IReadOnlyList<Entry>> plan, int ranks)
    {
        int[,] counts = new int[ranks, ranks];
        for (int dst = 0; dst < ranks; dst++)
        {
            foreach (Entry e in plan[dst]) counts[e.SourceRank, dst]++;
        }
        return counts;
    }
}
