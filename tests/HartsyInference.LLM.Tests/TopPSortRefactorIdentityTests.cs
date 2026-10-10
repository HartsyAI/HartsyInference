using System;
using System.Collections.Generic;
using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>perf/llm-short-reply-decode identity proof. <see cref="TopPStep.Apply"/> now reuses per-generation
/// scratch buffers and sorts via <see cref="SamplerMath.SortDescendingByValue"/> (a primitive
/// <c>Array.Sort(float[], int[])</c> pair, negate/sort/un-negate for descending order) instead of allocating two
/// fresh vocab-sized arrays and sorting through a <c>Comparison&lt;int&gt;</c> delegate every token — see that
/// method's remarks for why (measured ~20+ ms/token at Qwen3's ~152K vocabulary, the dominant cost behind the
/// voice-agent short-reply slowdown this change fixes; <c>benchmarks/results/2026-10-01_llm_short_reply_decode.md</c>).
///
/// <para><see cref="OldTopPStep"/> below is a byte-for-byte copy of <c>TopPStep.Apply</c> and
/// <c>SamplerMath.Softmax</c>/<c>ArgsortDescending</c> exactly as they stood before that change (transcribed from
/// the pre-fix source, not reconstructed from memory) — the reference this test checks the new code against.
/// Asserting the two mask the SAME logits to <see cref="float.NegativeInfinity"/>, for the same input, across many
/// seeded pseudo-random distributions — peaked (ordinary post-temperature logits), near-flat (high temperature,
/// so the cumulative walk needs most of the vocabulary to reach <c>p</c>), and one with a long pre-masked tail
/// (as if <c>RepetitionPenaltyStep</c>/<c>TopKStep</c> already ran) — at vocabulary sizes from 1 up to Qwen3's
/// real ~152K is the actual correctness proof for this change: <c>tests/regression-ab.sh --expect identical</c>
/// cannot reach it, because its identical-output arms are greedy, and <see cref="SamplerChain.Next"/> short-
/// circuits non-greedy-only code (TopPStep's masking, the final multinomial draw) out of the greedy path
/// entirely.</para></summary>
public sealed class TopPSortRefactorIdentityTests
{

    [Theory]
    [InlineData(1, 64, 0.95f)]
    public void OldAndNewTopP_MaskTheSameLogits_PeakedDistribution(int seed, int vocab, float p)
    {
        float[] logits = RandomLogits(seed, vocab, spread: 8.0f); // resembles real post-temperature logit spread
        AssertSameMasking(logits, p);
    }

    [Theory]
    [InlineData(42, 20_000)]
    public void OldAndNewTopP_MaskTheSameLogits_AtTheCandidateSumRoundingBoundary(int seed, int vocab)
    {
        // The fast path's safety check sums candidate probabilities in VOCABULARY-INDEX order
        // (candidateSum); the cumulative walk that decides what to keep sums the SAME values again, in
        // SORTED order. Float addition isn't associative, so the two sums can round to different floats —
        // TopPStep.Apply's remarks call this out and fall back to the full sort whenever the sorted walk
        // exhausts every candidate without reaching p, even though candidateSum said it would. This computes
        // both sums independently (a third, from-scratch reference, not OldTopPStep's own code) and sets p
        // at and around wherever they land, including strictly between them when they differ, which is
        // exactly where that fallback is the only way to stay correct.
        float[] logits = RandomLogits(seed, vocab, spread: 8.0f);
        float[] probs = new float[logits.Length];
        OldTopPStep.Softmax(logits, probs);

        int[] candidateIndices = new int[probs.Length];
        int candidateCount = 0;
        float indexOrderSum = 0.0f;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] > 1e-6f)
            {
                candidateIndices[candidateCount++] = i;
                indexOrderSum += probs[i];
            }
        }
        Assert.True(candidateCount > 0, "test setup: expected at least one candidate above the floor");

        Array.Sort(candidateIndices, 0, candidateCount, Comparer<int>.Create((a, b) => probs[b].CompareTo(probs[a])));
        float sortedOrderSum = 0.0f;
        for (int rank = 0; rank < candidateCount; rank++)
        {
            sortedOrderSum += probs[candidateIndices[rank]];
        }

        List<float> pValues = [indexOrderSum, sortedOrderSum];
        if (sortedOrderSum != indexOrderSum)
        {
            // Strictly between the two sums: candidateSum (index-order) clears this p, but the sorted walk's
            // own total (sortedOrderSum) cannot — the exact scenario the fallback-on-no-reach exists for.
            float lo = Math.Min(indexOrderSum, sortedOrderSum);
            float hi = Math.Max(indexOrderSum, sortedOrderSum);
            pValues.Add(lo + (hi - lo) / 2.0f);
        }
        foreach (float p in pValues)
        {
            if (p is > 0.0f and < 1.0f)
            {
                AssertSameMasking((float[])logits.Clone(), p);
            }
        }
    }

    private static void AssertSameMasking(float[] logits, float p)
    {
        float[] oldLogits = (float[])logits.Clone();
        float[] newLogits = (float[])logits.Clone();
        OldTopPStep.Apply(oldLogits, p);
        new TopPStep(p).Apply(newLogits, []);
        Assert.Equal(oldLogits, newLogits);
    }

    private static float[] RandomLogits(int seed, int vocab, float spread)
    {
        Random rng = new(seed);
        float[] logits = new float[vocab];
        for (int i = 0; i < vocab; i++)
        {
            logits[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * spread);
        }
        return logits;
    }

    /// <summary>Byte-for-byte copy of <c>TopPStep.Apply</c> and <c>SamplerMath</c>'s <c>Softmax</c>/
    /// <c>ArgsortDescending</c> exactly as they stood immediately before perf/llm-short-reply-decode (delegate-
    /// comparer <c>Array.Sort</c>, fresh arrays every call) — the reference the new, production code is checked
    /// against above.</summary>
    private static class OldTopPStep
    {
        public static void Apply(float[] logits, float p)
        {
            if (p >= 1.0f)
            {
                return;
            }
            int count = logits.Length;
            float[] probs = new float[count];
            Softmax(logits, probs);
            int[] order = ArgsortDescending(probs);
            float cumulative = 0.0f;
            int keep = 0;
            for (int rank = 0; rank < count; rank++)
            {
                cumulative += probs[order[rank]];
                keep = rank + 1;
                if (cumulative >= p)
                {
                    break;
                }
            }
            for (int rank = keep; rank < count; rank++)
            {
                logits[order[rank]] = float.NegativeInfinity;
            }
        }

        // internal rather than private: the boundary-rounding test above calls this directly to compute
        // ground-truth candidate probabilities, independent of the production SamplerMath.Softmax it mirrors.
        internal static void Softmax(ReadOnlySpan<float> logits, Span<float> probs)
        {
            int count = logits.Length;
            float max = float.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                if (logits[i] > max)
                {
                    max = logits[i];
                }
            }
            double sum = 0.0;
            for (int i = 0; i < count; i++)
            {
                float logit = logits[i];
                if (float.IsNegativeInfinity(logit))
                {
                    probs[i] = 0.0f;
                    continue;
                }
                float e = MathF.Exp(logit - max);
                probs[i] = e;
                sum += e;
            }
            if (sum <= 0.0)
            {
                return;
            }
            float inv = (float)(1.0 / sum);
            for (int i = 0; i < count; i++)
            {
                probs[i] *= inv;
            }
        }

        private static int[] ArgsortDescending(float[] values)
        {
            int[] order = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                order[i] = i;
            }
            Array.Sort(order, (int a, int b) => values[b].CompareTo(values[a]));
            return order;
        }
    }
}
