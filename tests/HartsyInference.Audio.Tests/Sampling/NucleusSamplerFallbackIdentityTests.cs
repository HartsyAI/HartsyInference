using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests.Sampling;

/// <summary>Proves <see cref="NucleusSampler.Draw"/>'s unbounded fallback (hit whenever <c>topK&lt;=0</c>, or
/// <c>topK</c> bounds nothing because it exceeds the bounded fast path's 1024 cap — Chatterbox, Zonos, Dia,
/// Bark's coarse/fine/generic stages and FishSpeech's codebook pass all hit this today) still returns the EXACT
/// token the pre-fix algorithm did, for the same logits/seed. <see cref="ReferenceDraw"/> is the pre-fix
/// algorithm kept verbatim as the oracle; <see cref="SortHelpersTests"/> proves the underlying sort swap is
/// permutation-identical, so this file's job is to catch a wiring mistake in how the fallback reads the sorted
/// arrays afterward (the rank-indexed <c>probs[rank]</c>/<c>order[rank]</c> convention the fix switched to).</summary>
public sealed class NucleusSamplerFallbackIdentityTests
{
    /// <summary>Verbatim pre-fix <c>NucleusSampler.Draw</c> (both the bounded fast path, unchanged by this PR,
    /// and the unbounded fallback this PR rewrote), kept ONLY as the test oracle.</summary>
    private static int ReferenceDraw(Span<float> logits, int count, float temperature, int topK, float topP,
        ref uint rng, int maskToken = -1, float minP = 0f)
    {
        float temp = temperature > 0 ? temperature : 1f;
        float invTemp = 1f / temp;
        int k = topK > 0 ? Math.Min(topK, count) : count;

        if (k < count && k <= 1024)
        {
            Span<int> ki = stackalloc int[k];
            Span<float> kv = stackalloc float[k];
            int n = 0, minPos = 0;
            float minVal = float.PositiveInfinity, gmax = float.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                if (i == maskToken) continue;
                float v = logits[i] * invTemp;
                if (v > gmax) gmax = v;
                if (n < k)
                {
                    ki[n] = i; kv[n] = v; n++;
                    if (n == k) { minVal = float.PositiveInfinity; for (int j = 0; j < k; j++) if (kv[j] < minVal) { minVal = kv[j]; minPos = j; } }
                }
                else if (v > minVal)
                {
                    ki[minPos] = i; kv[minPos] = v;
                    minVal = float.PositiveInfinity; for (int j = 0; j < k; j++) if (kv[j] < minVal) { minVal = kv[j]; minPos = j; }
                }
            }
            for (int a = 1; a < n; a++)
            {
                int ii = ki[a]; float vv = kv[a]; int b = a - 1;
                while (b >= 0 && kv[b] < vv) { kv[b + 1] = kv[b]; ki[b + 1] = ki[b]; b--; }
                kv[b + 1] = vv; ki[b + 1] = ii;
            }
            bool needZ = (topP > 0f && topP < 1f) || minP > 0f;
            float invZ = 0f;
            if (needZ)
            {
                double z = 0; for (int i = 0; i < count; i++) { if (i == maskToken) continue; z += MathF.Exp(logits[i] * invTemp - gmax); }
                invZ = (float)(1.0 / z);
            }
            float minPThreshold = minP > 0f ? minP * (MathF.Exp(kv[0] - gmax) * invZ) : 0f;
            int keep = 0; float cumulative = 0f;
            for (int rank = 0; rank < n; rank++)
            {
                if (minPThreshold > 0f && rank > 0 && MathF.Exp(kv[rank] - gmax) * invZ < minPThreshold) break;
                keep = rank + 1;
                if (topP > 0f && topP < 1f) { cumulative += MathF.Exp(kv[rank] - gmax) * invZ; if (cumulative >= topP) break; }
            }
            float kmax = kv[0], sumE = 0f;
            Span<float> ew = stackalloc float[keep];
            for (int rank = 0; rank < keep; rank++) { float ex = MathF.Exp(kv[rank] - kmax); ew[rank] = ex; sumE += ex; }
            if (sumE <= 0f) return ki[0];
            float r = DeterministicRng.NextUniform(ref rng) * sumE;
            float acc = 0f;
            for (int rank = 0; rank < keep; rank++) { acc += ew[rank]; if (r <= acc) return ki[rank]; }
            return ki[keep - 1];
        }

        // Fallback: unbounded top-k (k == count). Full softmax + sort -- the pre-fix shape this PR rewrote.
        float[] probs = new float[count];
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            float v = logits[i] / temp;
            probs[i] = v;
            if (v > max) max = v;
        }
        double sum = 0;
        for (int i = 0; i < count; i++)
        {
            float e = MathF.Exp(probs[i] - max);
            probs[i] = e;
            sum += e;
        }
        float inv = (float)(1.0 / sum);
        for (int i = 0; i < count; i++) probs[i] *= inv;
        if ((uint)maskToken < (uint)count) probs[maskToken] = 0f;

        int[] order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        Array.Sort(order, (a, b) => probs[b].CompareTo(probs[a]));
        float minPThresholdF = minP > 0f ? minP * probs[order[0]] : 0f;
        float cumulativeF = 0f;
        int keepF = 0;
        for (int rank = 0; rank < k; rank++)
        {
            if (minPThresholdF > 0f && rank > 0 && probs[order[rank]] < minPThresholdF) break;
            cumulativeF += probs[order[rank]];
            keepF = rank + 1;
            if (topP > 0 && topP < 1f && cumulativeF >= topP) break;
        }

        float keptSum = 0f;
        for (int rank = 0; rank < keepF; rank++) keptSum += probs[order[rank]];
        if (keptSum <= 0f) return order[0];
        float rF = DeterministicRng.NextUniform(ref rng) * keptSum;
        float accF = 0f;
        for (int rank = 0; rank < keepF; rank++)
        {
            accF += probs[order[rank]];
            if (rF <= accF) return order[rank];
        }
        return order[keepF - 1];
    }

    private static float[] MakeLogits(int count, Random rng)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
        return v;
    }

    /// <summary>Two consecutive draws from the SAME starting rng state must match on BOTH the returned token
    /// and the final rng state — proving the fix consumes the deterministic rng stream identically (no added
    /// or skipped draw, e.g. around the <c>keptSum &lt;= 0</c> early-out that returns without drawing).</summary>
    private static void AssertIdenticalTokenAndRngAdvance(float[] logits, int count, float temperature, int topK,
        float topP, float minP, int maskToken, int seed, string label)
    {
        uint rngOld = DeterministicRng.Seed(seed);
        uint rngNew = DeterministicRng.Seed(seed);

        int tokenOld1 = ReferenceDraw((float[])logits.Clone(), count, temperature, topK, topP, ref rngOld, maskToken, minP);
        int tokenNew1 = NucleusSampler.Draw((float[])logits.Clone(), count, temperature, topK, topP, ref rngNew, maskToken, minP);
        Assert.True(tokenOld1 == tokenNew1, $"{label}: first draw old={tokenOld1} new={tokenNew1}");

        // Draw again from each advanced state -- only matches if the rng was consumed identically the first time.
        int tokenOld2 = ReferenceDraw((float[])logits.Clone(), count, temperature, topK, topP, ref rngOld, maskToken, minP);
        int tokenNew2 = NucleusSampler.Draw((float[])logits.Clone(), count, temperature, topK, topP, ref rngNew, maskToken, minP);
        Assert.True(tokenOld2 == tokenNew2, $"{label}: second draw (rng-advance check) old={tokenOld2} new={tokenNew2}");
        Assert.Equal(rngOld, rngNew);
    }

    [Theory]
    [InlineData(1025)]  // Zonos/GptSoVits OutputVocab
    [InlineData(1028)]  // Dia AudioVocab
    [InlineData(8194)]  // Chatterbox SpeechVocab
    [InlineData(10001)] // Bark semantic
    [InlineData(102048)] // FishSpeech TextVocab
    public void UnboundedTopK_RandomLogits_MatchesReference(int count)
    {
        Random rng = new(42 + count);
        for (int trial = 0; trial < 20; trial++)
        {
            float[] logits = MakeLogits(count, rng);
            float topP = (float)rng.NextDouble();
            float minP = trial % 3 == 0 ? (float)(rng.NextDouble() * 0.2) : 0f;
            AssertIdenticalTokenAndRngAdvance(logits, count, 0.8f, topK: 0, topP, minP, maskToken: -1,
                seed: 1000 + trial, label: $"n={count} trial={trial}");
        }
    }

    [Theory]
    [InlineData(0.0f)]    // disabled (condition is strictly 0<topP<1)
    [InlineData(0.001f)]  // tiny
    [InlineData(0.5f)]
    [InlineData(0.999999f)] // huge, just under 1
    [InlineData(1.0f)]    // disabled (>= 1)
    public void UnboundedTopK_TopPEdgeValues_MatchReference(float topP)
    {
        Random rng = new(777);
        float[] logits = MakeLogits(1028, rng);
        AssertIdenticalTokenAndRngAdvance(logits, 1028, 1.0f, topK: 0, topP, minP: 0f, maskToken: -1,
            seed: 55, label: $"topP={topP}");
    }

    [Fact]
    public void UnboundedTopK_TemperatureZero_MatchesReference()
    {
        Random rng = new(99);
        float[] logits = MakeLogits(2051, rng); // CSM AudioVocab
        AssertIdenticalTokenAndRngAdvance(logits, 2051, temperature: 0f, topK: 0, topP: 0.9f, minP: 0f,
            maskToken: -1, seed: 7, label: "temperature=0");
    }

    [Fact]
    public void UnboundedTopK_MinPOnly_NoTopP_MatchesReference()
    {
        Random rng = new(123);
        float[] logits = MakeLogits(1025, rng);
        AssertIdenticalTokenAndRngAdvance(logits, 1025, temperature: 1f, topK: 0, topP: 0f, minP: 0.1f,
            maskToken: -1, seed: 11, label: "minP-only");
    }

    [Fact]
    public void UnboundedTopK_WithMaskToken_MatchesReference()
    {
        Random rng = new(321);
        float[] logits = MakeLogits(8194, rng);
        AssertIdenticalTokenAndRngAdvance(logits, 8194, temperature: 0.8f, topK: 0, topP: 1f, minP: 0.05f,
            maskToken: 42, seed: 13, label: "maskToken");
    }

    /// <summary>topK above the bounded fast path's 1024 cap but still below vocab count ALSO falls into the
    /// fallback (k stays topK, not count) -- a second, less common way to reach the same code this PR rewrote.</summary>
    [Fact]
    public void TopKAboveFastPathCap_MatchesReference()
    {
        Random rng = new(246);
        float[] logits = MakeLogits(8194, rng);
        AssertIdenticalTokenAndRngAdvance(logits, 8194, temperature: 1f, topK: 1500, topP: 0.9f, minP: 0f,
            maskToken: -1, seed: 17, label: "topK=1500 (>1024, <count)");
    }

    /// <summary>Exact ties: several tokens share the identical maximal logit. Constructed so the tie sits right
    /// at the top (most likely to matter for the multinomial draw, which walks sorted order).</summary>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(50)]
    public void UnboundedTopK_ExactTiesAtTop_MatchesReference(int numTied)
    {
        int count = 1028;
        Random rng = new(900 + numTied);
        float[] logits = MakeLogits(count, rng);
        for (int i = 0; i < Math.Min(numTied, count); i++) logits[i] = 5.0f; // identical top value across several ids
        for (int trial = 0; trial < 10; trial++)
        {
            AssertIdenticalTokenAndRngAdvance(logits, count, 1f, topK: 0, topP: 0.5f, minP: 0f, maskToken: -1,
                seed: 500 + trial, label: $"numTied={numTied} trial={trial}");
        }
    }

    /// <summary>Regression guard for the BOUNDED fast path (topK small, &lt; count), which this PR does not
    /// touch -- confirms the untouched code still agrees with the kept reference copy of the whole function.</summary>
    [Theory]
    [InlineData(50)]
    [InlineData(250)]
    public void BoundedFastPath_Unaffected_MatchesReference(int topK)
    {
        Random rng = new(31415);
        float[] logits = MakeLogits(8194, rng);
        AssertIdenticalTokenAndRngAdvance(logits, 8194, 0.8f, topK, topP: 0.9f, minP: 0f, maskToken: -1,
            seed: 19, label: $"fastpath topK={topK}");
    }
}
