using HartsyInference.LLM.Generation.Speculative;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculative;

/// <summary>Algorithm 1 against the objective it maximizes. Brute force is the reference, computed from the paper's definitions and not from the scheduler's own loop.
/// Algorithm 1 must match it whenever the objective is unimodal along the walk; a jagged profile shows the first-non-improvement stop the paper's section 5.2 addresses.</summary>
public sealed class ConfidenceSchedulerTests
{
    private const int Block = 5;

    /// <summary>Θ(l) = (1 + Σ_{j≤l} a_j) · SPS(1 + l) for one sequence. <paramref name="sps"/> is indexed by batch size minus one, so <c>sps[l]</c> is SPS(1 + l).</summary>
    private static double Objective(double[] survival, double[] sps, int length)
    {
        double tau = 1;
        for (int j = 0; j < length; j++) tau += survival[j];
        return tau * sps[length];
    }

    /// <summary>The survival products a_j = c_1 · … · c_j for confidence logits in draft order, with c = sigmoid(logit).</summary>
    private static double[] Survival(float[] logits)
    {
        double[] a = new double[logits.Length];
        double s = 1;
        for (int j = 0; j < logits.Length; j++)
        {
            s *= 1.0 / (1.0 + Math.Exp(-logits[j]));
            a[j] = s;
        }
        return a;
    }

    /// <summary>The length that maximizes Θ over 0 to <paramref name="limit"/>; a tie keeps the shorter length.</summary>
    private static int BruteForce(double[] survival, double[] sps, int limit)
    {
        int best = 0;
        double bestValue = Objective(survival, sps, 0);
        for (int l = 1; l <= limit; l++)
        {
            double value = Objective(survival, sps, l);
            if (value > bestValue)
            {
                bestValue = value;
                best = l;
            }
        }
        return best;
    }

    /// <summary>True when Θ rises and then never rises again. Only then does Algorithm 1's early stop reach the global maximum.</summary>
    private static bool Unimodal(double[] survival, double[] sps, int limit)
    {
        bool falling = false;
        double previous = Objective(survival, sps, 0);
        for (int l = 1; l <= limit; l++)
        {
            double value = Objective(survival, sps, l);
            if (value > previous)
            {
                if (falling) return false;
            }
            else
            {
                falling = true;
            }
            previous = value;
        }
        return true;
    }

    [Fact]
    public void Algorithm_One_Matches_Brute_Force_Whenever_The_Objective_Is_Unimodal()
    {
        int checkedCases = 0;
        for (int seed = 1; seed <= 400; seed++)
        {
            Random random = new(seed);
            double s0 = 200 + 800 * random.NextDouble(), decay = 0.5 * random.NextDouble();
            double[] sps = Enumerable.Range(0, Block + 1).Select(b => s0 / (1 + decay * b)).ToArray();
            float[] logits = Enumerable.Range(0, Block).Select(_ => (float)(8 * random.NextDouble() - 4)).ToArray();

            int chosen = new ConfidenceScheduler(new SpsProfile(sps), Block).Choose(logits, Block);
            double[] survival = Survival(logits);
            if (!Unimodal(survival, sps, Block)) continue;

            checkedCases++;
            Assert.Equal(BruteForce(survival, sps, Block), chosen);
        }
        Assert.True(checkedCases >= 100, $"only {checkedCases} unimodal cases; the gate would be vacuous");
    }

    [Fact]
    public void A_Jagged_Profile_Stops_At_The_First_Dip_Where_Brute_Force_Goes_On()
    {
        // SPS dips at batch 3 and recovers. Θ falls at length 2 and rises again after it, so the walk stops early. The paper's section 5.2 search covers this case.
        double[] sps = [100, 100, 60, 100, 90, 80];
        float[] logits = Enumerable.Repeat(4.6f, Block).ToArray();   // sigmoid(4.6) ≈ 0.99 at each position
        double[] survival = Survival(logits);

        Assert.Equal(1, new ConfidenceScheduler(new SpsProfile(sps), Block).Choose(logits, Block));
        Assert.Equal(Block, BruteForce(survival, sps, Block));
    }

    [Fact]
    public void A_Flat_Profile_Verifies_The_Whole_Block()
    {
        // with SPS constant, every position adds survival to τ and so improves Θ
        ConfidenceScheduler scheduler = new(new SpsProfile([100, 100, 100, 100, 100, 100]), Block);
        Assert.Equal(Block, scheduler.Choose([4.6f, 1.2f, -0.6f, -2.1f, -4.3f], Block));
    }

    [Fact]
    public void An_Expensive_Second_Position_Stops_The_Walk_At_Once()
    {
        // one token costs 1000 steps a second, anything larger almost nothing: verifying more never pays
        ConfidenceScheduler scheduler = new(new SpsProfile([1000, 1, 1, 1, 1, 1]), Block);
        Assert.Equal(0, scheduler.Choose([4.6f, 4.6f, 4.6f, 4.6f, 4.6f], Block));
    }

    [Fact]
    public void Nothing_Is_Chosen_Without_A_Budget_Or_Positions()
    {
        ConfidenceScheduler scheduler = new(new SpsProfile([100, 100, 100, 100, 100, 100]), Block);
        Assert.Equal(0, scheduler.Choose([4.6f, 4.6f], 0));
        Assert.Equal(0, scheduler.Choose(ReadOnlySpan<float>.Empty, Block));
    }

    [Fact]
    public void A_Profile_Rejects_Non_Positive_Entries_And_Uncovered_Batches()
    {
        Assert.Throws<ArgumentException>(() => new SpsProfile([]));
        Assert.Throws<ArgumentException>(() => new SpsProfile([10, 0]));
        Assert.Throws<ArgumentException>(() => new SpsProfile([10, double.NaN]));

        SpsProfile profile = new([10, 9]);
        Assert.Equal(9, profile.StepsPerSecond(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => profile.StepsPerSecond(3));
        Assert.Throws<ArgumentException>(() => new ConfidenceScheduler(profile, 2));
    }
}
