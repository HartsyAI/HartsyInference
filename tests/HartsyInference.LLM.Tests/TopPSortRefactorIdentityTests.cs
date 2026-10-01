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
    private const int Qwen3VocabSize = 151_936;

    [Theory]
    [InlineData(1, 64, 0.95f)]
    [InlineData(2, 1000, 0.95f)]
    [InlineData(3, 1000, 0.5f)]
    [InlineData(4, 1000, 0.01f)]
    [InlineData(5, Qwen3VocabSize, 0.95f)]
    [InlineData(6, Qwen3VocabSize, 0.5f)]
    [InlineData(7, Qwen3VocabSize, 0.999f)]
    public void OldAndNewTopP_MaskTheSameLogits_PeakedDistribution(int seed, int vocab, float p)
    {
        float[] logits = RandomLogits(seed, vocab, spread: 8.0f); // resembles real post-temperature logit spread
        AssertSameMasking(logits, p);
    }

    [Theory]
    [InlineData(11, 1000, 0.95f)]
    [InlineData(12, Qwen3VocabSize, 0.95f)]
    public void OldAndNewTopP_MaskTheSameLogits_NearFlatDistribution(int seed, int vocab, float p)
    {
        // A tiny spread makes softmax nearly uniform: the nucleus needs MOST of the vocabulary to reach p, a very
        // different cumulative-walk length than the peaked cases above.
        float[] logits = RandomLogits(seed, vocab, spread: 0.01f);
        AssertSameMasking(logits, p);
    }

    [Theory]
    [InlineData(21, Qwen3VocabSize, 0.95f)]
    [InlineData(22, Qwen3VocabSize, 0.5f)]
    public void OldAndNewTopP_MaskTheSameLogits_WithAPreMaskedTail(int seed, int vocab, float p)
    {
        // Simulates RepetitionPenaltyStep/TopKStep already having run: most of the vocabulary is -Infinity
        // before TopP ever sees it (Softmax gives those exactly 0.0f probability either way).
        float[] logits = RandomLogits(seed, vocab, spread: 8.0f);
        Random rng = new(seed);
        for (int i = 0; i < logits.Length; i++)
        {
            if (rng.NextDouble() < 0.98) logits[i] = float.NegativeInfinity;
        }
        AssertSameMasking(logits, p);
    }

    [Fact]
    public void OldAndNewTopP_MaskTheSameLogits_SingleToken()
    {
        AssertSameMasking([5.0f], 0.95f);
    }

    [Fact]
    public void SamplerChain_Draw_IsDeterministicForSeed_AtRealVocabSize()
    {
        // SamplingAndTemplateTests.Sampling_IsDeterministicForSeed covers this property at 6 elements; this is
        // the same property at the scale that actually exercises SortDescendingByValue and the per-generation
        // scratch-buffer reuse end to end, through the production SamplerChain (OldTopPStep is not used here).
        SamplerChain a = SamplerChain.FromOptions(new SamplingOptions { Temperature = 0.7f, TopP = 0.95f, Seed = 42 });
        SamplerChain b = SamplerChain.FromOptions(new SamplingOptions { Temperature = 0.7f, TopP = 0.95f, Seed = 42 });
        List<int> history = [];
        for (int i = 0; i < 16; i++)
        {
            float[] logits = RandomLogits(seed: 99 + i, Qwen3VocabSize, spread: 8.0f);
            int ta = a.Next((float[])logits.Clone(), history);
            int tb = b.Next((float[])logits.Clone(), history);
            Assert.Equal(ta, tb);
            history.Add(ta);
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

        private static void Softmax(ReadOnlySpan<float> logits, Span<float> probs)
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
