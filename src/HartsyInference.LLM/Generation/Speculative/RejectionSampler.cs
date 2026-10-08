namespace HartsyInference.LLM.Generation.Speculative;

/// <summary>How much of a draft was kept, and the token that follows it.</summary>
/// <param name="Accepted">Drafted tokens kept, from the start of the draft.</param>
/// <param name="NextToken">The correction that replaced the first rejected token, or the bonus token drawn after a fully accepted draft.</param>
public readonly record struct SpeculativeOutcome(int Accepted, int NextToken);

/// <summary>Exact speculative sampling (Leviathan et al. 2023; Chen et al. 2023): accepts a drafted token with probability min(1, p/q) and, on rejection, draws from the normalized residual max(0, p - q). The emitted tokens follow the target distribution exactly, whatever the proposer does.</summary>
/// <remarks>The distributions are float and the acceptance ratio and residual are computed in double, so exactness holds up to the rounding already in the inputs.</remarks>
public static class RejectionSampler
{
    /// <summary>Verifies <paramref name="draft"/> against the target distributions.</summary>
    /// <param name="draft">The drafted tokens.</param>
    /// <param name="target">Target distributions: row i is the distribution at draft position i, and row <c>draft.Length</c> is the bonus position. Each row sums to one over the vocabulary.</param>
    /// <param name="proposal">The proposer's distributions, one per drafted position, or null when the proposer is deterministic (a point mass on the drafted token).</param>
    /// <param name="uniform">A source of independent draws uniform on [0, 1).</param>
    public static SpeculativeOutcome Verify(ReadOnlySpan<int> draft, float[][] target, float[][]? proposal, Func<double> uniform)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(uniform);
        if (target.Length != draft.Length + 1) throw new ArgumentException("Target needs one row per drafted token plus the bonus row.", nameof(target));
        if (proposal is not null && proposal.Length != draft.Length) throw new ArgumentException("Proposal needs one row per drafted token.", nameof(proposal));

        for (int i = 0; i < draft.Length; i++)
        {
            int x = draft[i];
            float[] p = target[i];
            if ((uint)x >= (uint)p.Length) throw new ArgumentOutOfRangeException(nameof(draft), x, "A drafted token is outside the vocabulary.");
            double px = p[x];
            double qx = proposal is null ? 1.0 : proposal[i][x];
            double ratio = qx > 0.0 ? px / qx : (px > 0.0 ? double.PositiveInfinity : 0.0);
            if (uniform() < Math.Min(1.0, ratio)) continue;
            return new SpeculativeOutcome(i, SampleResidual(p, proposal?[i], x, uniform));
        }
        return new SpeculativeOutcome(draft.Length, Draw(target[draft.Length], uniform));
    }

    private static int SampleResidual(float[] p, float[]? q, int x, Func<double> uniform)
    {
        double total = 0.0;
        for (int j = 0; j < p.Length; j++) total += Residual(p, q, x, j);
        // p and q only agree here when the rejection was a rounding artifact; the target itself is then the right draw
        if (total <= 0.0) return Draw(p, uniform);
        double r = uniform() * total, acc = 0.0;
        int last = -1;
        for (int j = 0; j < p.Length; j++)
        {
            double w = Residual(p, q, x, j);
            if (w <= 0.0) continue;
            last = j;
            acc += w;
            if (r < acc) return j;
        }
        return last;
    }

    private static double Residual(float[] p, float[]? q, int x, int j)
    {
        double qj = q is null ? (j == x ? 1.0 : 0.0) : q[j];
        return Math.Max(0.0, p[j] - qj);
    }

    private static int Draw(float[] p, Func<double> uniform)
    {
        double total = 0.0;
        for (int j = 0; j < p.Length; j++) total += p[j];
        if (total <= 0.0) throw new InvalidOperationException("The target distribution has no probability mass.");
        double r = uniform() * total, acc = 0.0;
        int last = -1;
        for (int j = 0; j < p.Length; j++)
        {
            if (p[j] <= 0.0f) continue;
            last = j;
            acc += p[j];
            if (r < acc) return j;
        }
        // rounding can leave r just past the accumulated mass; the last positive entry is the right draw
        return last;
    }
}
