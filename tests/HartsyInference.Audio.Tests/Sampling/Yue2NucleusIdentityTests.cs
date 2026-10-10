using HartsyInference.Audio.Models.Music;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Proves <see cref="Yue2LogitProcessor.ApplyNucleus"/> still masks the EXACT same set of entries to
/// <c>-Infinity</c> the pre-fix algorithm did. Unlike the other fixes in this PR, the pre-fix sort here was
/// already the primitive <c>Array.Sort(float[], int[])</c> overload — no delegate-comparator anti-pattern — so
/// <see cref="ReferenceApplyNucleus"/> (kept verbatim as the oracle) differs from the fix ONLY in allocating a
/// fresh <c>List&lt;int&gt;</c>/<c>List&lt;float&gt;</c> plus array copies every call; this suite exists to
/// prove that allocation-only change is still exactly output-preserving.</summary>
public sealed class Yue2NucleusIdentityTests
{
    /// <summary>Verbatim pre-fix <c>Yue2LogitProcessor.ApplyNucleus</c>, kept ONLY as the test oracle.</summary>
    private static void ReferenceApplyNucleus(Span<float> scores, float topP, int alwaysKeep)
    {
        List<int> candidates = [];
        List<float> values = [];
        for (int i = 0; i < scores.Length; i++)
        {
            if (!float.IsFinite(scores[i])) continue;
            candidates.Add(i);
            values.Add(-scores[i]);
        }
        if (candidates.Count <= alwaysKeep) return;

        int[] order = [.. candidates];
        Array.Sort([.. values], order);

        float max = scores[order[0]];
        double total = 0;
        double[] probabilities = new double[order.Length];
        for (int i = 0; i < order.Length; i++)
        {
            probabilities[i] = Math.Exp(scores[order[i]] - max);
            total += probabilities[i];
        }

        double cumulative = 0;
        for (int i = 0; i < order.Length; i++)
        {
            double p = probabilities[i] / total;
            if (i >= alwaysKeep && cumulative > topP) scores[order[i]] = float.NegativeInfinity;
            cumulative += p;
        }
    }

    private static float[] MakeScores(int n, Random rng, double maskedFraction = 0.0)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = rng.NextDouble() < maskedFraction ? float.NegativeInfinity : (float)(rng.NextDouble() * 12.0 - 6.0);
        return v;
    }

    private static void AssertIdenticalMask(float[] scores, float topP, int alwaysKeep, string label)
    {
        float[] viaOld = (float[])scores.Clone();
        float[] viaNew = (float[])scores.Clone();
        ReferenceApplyNucleus(viaOld, topP, alwaysKeep);
        Yue2LogitProcessor.ApplyNucleus(viaNew, topP, alwaysKeep);
        Assert.Equal(viaOld.Length, viaNew.Length);
        for (int i = 0; i < viaOld.Length; i++)
        {
            bool oldMasked = float.IsNegativeInfinity(viaOld[i]);
            bool newMasked = float.IsNegativeInfinity(viaNew[i]);
            Assert.True(oldMasked == newMasked, $"{label}: index {i} old-masked={oldMasked} new-masked={newMasked}");
            if (!oldMasked) Assert.Equal(viaOld[i], viaNew[i]);
        }
    }

    [Theory]
    [InlineData(0.1)]  // narrow nucleus
    public void RandomScores_MatchesReference(double topP)
    {
        Random rng = new(5000 + (int)(topP * 1_000_000));
        for (int trial = 0; trial < 15; trial++)
        {
            float[] scores = MakeScores(4096, rng);
            AssertIdenticalMask(scores, (float)topP, alwaysKeep: 1, $"topP={topP} trial={trial}");
        }
    }

}
