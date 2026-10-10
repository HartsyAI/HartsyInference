using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Bark;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Proves <see cref="BarkCausalStage.SampleTopPWithProb"/> (Bark's semantic-stage sampler, up to 768
/// calls per generation) still returns the EXACT token AND <c>eosProb</c> the pre-fix algorithm did.
/// <see cref="ReferenceSampleTopPWithProb"/> is the pre-fix algorithm kept verbatim as the oracle. Unlike
/// <see cref="NucleusSamplerFallbackIdentityTests"/>, this sampler's final draw walks NATURAL index order (see
/// its own remarks), so the sort here only ever needs to reproduce the old cutoff SET — this suite still checks
/// token-for-token equality rather than relying on that argument alone.</summary>
public sealed class BarkSemanticSamplingIdentityTests
{
    /// <summary>Verbatim pre-fix <c>BarkCausalStage.SampleTopPWithProb</c>, kept ONLY as the test oracle.</summary>
    private static int ReferenceSampleTopPWithProb(Span<float> logits, float temperature, float topP, ref uint rng,
        int eosSlot, out float eosProb)
    {
        int n = logits.Length;
        float[] work = logits.ToArray();
        if (topP > 0f && topP < 1f)
        {
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            float[] w = work;
            Array.Sort(order, (a, b) => w[b].CompareTo(w[a]));
            double sum = 0;
            float max = w[order[0]];
            double[] probs = new double[n];
            for (int r = 0; r < n; r++) { probs[r] = Math.Exp(w[order[r]] - max); sum += probs[r]; }
            double cum = 0;
            for (int r = 0; r < n; r++)
            {
                cum += probs[r] / sum;
                if (cum > topP && r > 0)
                {
                    for (int rr = r; rr < n; rr++) work[order[rr]] = float.NegativeInfinity;
                    break;
                }
            }
        }
        float temp = temperature > 0 ? temperature : 1f;
        float mx = float.NegativeInfinity;
        for (int i = 0; i < n; i++) { work[i] /= temp; if (work[i] > mx) mx = work[i]; }
        double total = 0;
        for (int i = 0; i < n; i++) { work[i] = MathF.Exp(work[i] - mx); total += work[i]; }
        float inv = (float)(1.0 / total);
        for (int i = 0; i < n; i++) work[i] *= inv;
        eosProb = (uint)eosSlot < (uint)n ? work[eosSlot] : 0f;
        float r2 = DeterministicRng.NextUniform(ref rng);
        float acc = 0f;
        for (int i = 0; i < n; i++)
        {
            acc += work[i];
            if (r2 <= acc) return i;
        }
        return n - 1;
    }

    private const int SemVocabPlusEos = 10_001; // 10,000 real semantic ids + the contiguous EOS slot

    private static float[] MakeLogits(int n, Random rng)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 16.0 - 8.0);
        return v;
    }

    private static void AssertIdentical(float[] logits, float temperature, float topP, int eosSlot, int seed, string label)
    {
        uint rngOld = DeterministicRng.Seed(seed);
        uint rngNew = DeterministicRng.Seed(seed);
        int tokenOld = ReferenceSampleTopPWithProb((float[])logits.Clone(), temperature, topP, ref rngOld, eosSlot, out float eosProbOld);
        int tokenNew = BarkCausalStage.SampleTopPWithProb((float[])logits.Clone(), temperature, topP, ref rngNew, eosSlot, out float eosProbNew);
        Assert.True(tokenOld == tokenNew, $"{label}: token old={tokenOld} new={tokenNew}");
        Assert.Equal(eosProbOld, eosProbNew);
        Assert.Equal(rngOld, rngNew);
    }

    [Theory]
    [InlineData(0.0f)]      // disabled: topP must be strictly in (0,1)
    [InlineData(1.0f)]      // disabled
    public void TopPEdgeValues_MatchReference(float topP)
    {
        Random rng = new(900);
        float[] logits = MakeLogits(SemVocabPlusEos, rng);
        for (int trial = 0; trial < 10; trial++)
        {
            AssertIdentical(logits, temperature: 0.7f, topP, eosSlot: SemVocabPlusEos - 1,
                seed: 1 + trial, label: $"topP={topP} trial={trial}");
        }
    }

    /// <summary>Exact ties right at the cumulative-cut rank, which decides whether the entry that crosses the
    /// threshold survives (the reference's <c>r &gt; 0</c> guard keeps at least one, even at the boundary).</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(20)]
    public void ExactTiesAtTop_MatchesReference(int numTied)
    {
        Random rng = new(902 + numTied);
        float[] logits = MakeLogits(SemVocabPlusEos, rng);
        for (int i = 0; i < numTied; i++) logits[i] = 9.0f;
        for (int trial = 0; trial < 10; trial++)
        {
            AssertIdentical(logits, temperature: 0.7f, topP: 0.5f, eosSlot: SemVocabPlusEos - 1,
                seed: 10 + trial, label: $"numTied={numTied} trial={trial}");
        }
    }

}
