namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>One example's token layout for a budget (upstream <c>Layout</c> and <c>build_layout</c>): a top-down split of the octree. <c>k4</c> of the 512 cubes open into their eight cells and <c>k2</c> of
/// the cells they expose open into their two Sobol points, so the token count is exact: <c>N = 512 + 7·k4 + k2</c>, any of 512 or 519 to 8192. The token sequence is <c>[s = 4 tokens | s = 2 tokens | s = 1 tokens]</c>:
/// <see cref="Cubes"/> are the cubes left whole, <see cref="Cells"/> the cells left whole inside open cubes, and <see cref="Pairs"/> the cells opened into their two points, each in ascending index order.</summary>
public sealed class BudgetLayout
{
    /// <summary>The merge thresholds <c>(τ4, τ2)</c> the budget balance starts from.</summary>
    public static readonly (double Cube, double Cell) Thresholds = (2.1, 1.5);

    private BudgetLayout(int[] cubes, int[] cells, int[] pairs, int k4, int k2)
    {
        Cubes = cubes;
        Cells = cells;
        Pairs = pairs;
        K4 = k4;
        K2 = k2;
        CubeTokens = BudgetOctree.Cubes - k4;
        CellTokens = 8 * k4 - k2;
        PointTokens = 2 * k2;
        if (k2 + CellTokens + 8 * CubeTokens != BudgetOctree.Cells || cubes.Length != CubeTokens || cells.Length != CellTokens || pairs.Length != k2)
            throw new InvalidOperationException($"Layout counts do not cover the octree: k4 {k4}, k2 {k2}.");
    }

    /// <summary>Cube indices that carry one s = 4 token each.</summary>
    public int[] Cubes { get; }

    /// <summary>Cell indices that carry one s = 2 token each.</summary>
    public int[] Cells { get; }

    /// <summary>Cell indices that opened into their two Sobol points (s = 1).</summary>
    public int[] Pairs { get; }

    /// <summary>Cubes opened into their cells.</summary>
    public int K4 { get; }

    /// <summary>Cells opened into their points.</summary>
    public int K2 { get; }

    /// <summary>s = 4 tokens.</summary>
    public int CubeTokens { get; }

    /// <summary>s = 2 tokens.</summary>
    public int CellTokens { get; }

    /// <summary>s = 1 tokens, two per opened cell.</summary>
    public int PointTokens { get; }

    /// <summary>The sequence length.</summary>
    public int Tokens => CubeTokens + CellTokens + PointTokens;

    /// <summary>Whether every Sobol point is its own token: the layout of Idol itself.</summary>
    public bool IsDense => K4 == BudgetOctree.Cubes && K2 == BudgetOctree.Cells;

    /// <summary>The 8192-token layout.</summary>
    public static BudgetLayout Dense() => new([], [], Enumerable.Range(0, BudgetOctree.Cells).ToArray(), BudgetOctree.Cubes, BudgetOctree.Cells);

    /// <summary>Round a requested token count to the nearest realisable one: 512, or 519 to 8192 (no padding ever).</summary>
    public static int SnapTokens(double requested)
    {
        int n = (int)Math.Round(requested, MidpointRounding.ToEven);
        n = Math.Max(BudgetOctree.Cubes, Math.Min(BudgetOctree.Points, n));
        if (n > BudgetOctree.Cubes && n < BudgetOctree.Cubes + 7)
            n = n - BudgetOctree.Cubes < 4 ? BudgetOctree.Cubes : BudgetOctree.Cubes + 7;
        return n;
    }

    /// <summary>The range of cube splits that realises <paramref name="tokens"/> with a valid tree: <c>ceil((N − 512) / 15) ≤ k4 ≤ min(512, floor((N − 512) / 7))</c>.</summary>
    public static (int Low, int High) K4Bounds(int tokens)
    {
        if (tokens == BudgetOctree.Cubes)
            return (0, 0);
        int low = (tokens - BudgetOctree.Cubes + 14) / 15, high = Math.Min(BudgetOctree.Cubes, (tokens - BudgetOctree.Cubes) / 7);
        if (tokens < BudgetOctree.Cubes || high < low)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "A budget realises 512 tokens or 519 to 8192.");
        return (low, high);
    }

    /// <summary><c>(k4, k2)</c> for a token count; <paramref name="k4"/> is clamped into the valid range, and without it the coarse-first <c>min(512, (N − 512) / 7)</c> is used.</summary>
    public static (int K4, int K2) SplitCounts(int tokens, int? k4 = null)
    {
        (int low, int high) = K4Bounds(tokens);
        int k = k4 is { } v ? Math.Min(Math.Max(v, low), high) : high;
        return (k, tokens - BudgetOctree.Cubes - 7 * k);
    }

    /// <summary>How a budget is spent between the two split levels, from the content (upstream <c>balance_splits</c>): a shared offset shifts both merge thresholds, 24 bisection steps find the one that lands the
    /// threshold rule on <paramref name="requestedTokens"/>, and the number of cubes that rule opens is then realised exactly by rank. Without it the budget would be spent coarse-first and a 50 % layout would
    /// degenerate to the uniform s = 2 grid, ignoring salience.</summary>
    public static (int K4, int K2) BalanceSplits(ReadOnlySpan<float> cubeScores, ReadOnlySpan<float> cellScores, int requestedTokens, int iterations = 24)
    {
        int n = SnapTokens(requestedTokens);
        (int lowK4, int highK4) = K4Bounds(n);
        if (lowK4 == highK4)
            return SplitCounts(n, lowK4);
        (double tau4, double tau2) = Thresholds;
        double lo = -16.0, hi = 16.0;
        for (int i = 0; i < iterations; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (CountAt(cubeScores, cellScores, tau4, tau2, mid).Tokens > n)
                lo = mid;
            else
                hi = mid;
        }

        return SplitCounts(n, (int)Math.Round(CountAt(cubeScores, cellScores, tau4, tau2, hi).Cubes, MidpointRounding.ToEven));
    }

    // The tokens and the open cubes the threshold rule gives at a shared offset delta. The score tensors are float32 and a Python float is cast to float32 for the comparison.
    private static (double Cubes, double Tokens) CountAt(ReadOnlySpan<float> cubeScores, ReadOnlySpan<float> cellScores, double tau4, double tau2, double delta)
    {
        float t4 = (float)(tau4 + delta), t2 = (float)(tau2 + delta);
        int open4 = 0, open2 = 0;
        bool[] cubeOpen = new bool[BudgetOctree.Cubes];
        for (int j = 0; j < BudgetOctree.Cubes; j++)
        {
            cubeOpen[j] = cubeScores[j] > t4;
            if (cubeOpen[j])
                open4++;
        }

        for (int c = 0; c < BudgetOctree.Cells; c++)
            if (cubeOpen[BudgetOctree.CubeOfCell(c)] && cellScores[c] > t2)
                open2++;
        return (open4, BudgetOctree.Cubes + 7.0 * open4 + open2);
    }

    /// <summary>The exact-budget layout from per-example salience (upstream <c>build_layout</c>): the <c>k4</c> highest-scoring cubes open, and among the cells they expose the <c>k2</c> highest-scoring open into
    /// Sobol points. Ties break by the lower index.</summary>
    /// <param name="cubeScores">512 scores, <see cref="BudgetOctree.DetailScores"/>.</param>
    /// <param name="cellScores">4096 scores.</param>
    /// <param name="requestedTokens">The token budget; snapped to a realisable count.</param>
    public static BudgetLayout Build(ReadOnlySpan<float> cubeScores, ReadOnlySpan<float> cellScores, int requestedTokens)
    {
        if (cubeScores.Length != BudgetOctree.Cubes || cellScores.Length != BudgetOctree.Cells)
            throw new ArgumentException($"A layout is built from {BudgetOctree.Cubes} cube scores and {BudgetOctree.Cells} cell scores; got {cubeScores.Length} and {cellScores.Length}.");
        int n = SnapTokens(requestedTokens);
        (int k4, int k2) = BalanceSplits(cubeScores, cellScores, n);
        bool[] split4 = new bool[BudgetOctree.Cubes];
        foreach (int j in TopK(cubeScores, k4, null))
            split4[j] = true;
        bool[] eligible = new bool[BudgetOctree.Cells];
        for (int c = 0; c < BudgetOctree.Cells; c++)
            eligible[c] = split4[BudgetOctree.CubeOfCell(c)];
        bool[] split2 = new bool[BudgetOctree.Cells];
        foreach (int c in TopK(cellScores, k2, eligible))
            split2[c] = true;

        List<int> cubes = [], cells = [], pairs = [];
        for (int j = 0; j < BudgetOctree.Cubes; j++)
            if (!split4[j])
                cubes.Add(j);
        for (int c = 0; c < BudgetOctree.Cells; c++)
        {
            if (split2[c])
                pairs.Add(c);
            else if (eligible[c])
                cells.Add(c);
        }

        return new BudgetLayout([.. cubes], [.. cells], [.. pairs], k4, k2);
    }

    /// <summary>A layout from index lists, checked against the octree's coverage (a mismatched list throws).</summary>
    public static BudgetLayout FromIndices(int[] cubes, int[] cells, int[] pairs, int k4, int k2) => new(cubes, cells, pairs, k4, k2);

    // The k highest scores, the lower index first among equal scores, among the entries allowed by eligible (null allows all).
    private static int[] TopK(ReadOnlySpan<float> scores, int k, bool[]? eligible)
    {
        if (k == 0)
            return [];
        List<int> candidates = [];
        for (int i = 0; i < scores.Length; i++)
            if (eligible is null || eligible[i])
                candidates.Add(i);
        float[] s = scores.ToArray();
        int[] order = [.. candidates.OrderByDescending(i => s[i]).ThenBy(i => i)];
        if (order.Length < k)
            throw new InvalidOperationException($"{k} of {order.Length} eligible entries cannot be opened.");
        return order[..k];
    }
}
