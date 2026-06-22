using System;

namespace HartsyInference.LLM.Sampling;

/// <summary>Shared numeric helpers for sampler steps: a numerically stable softmax that respects masked (negative-infinity) logits and a descending argsort.</summary>
internal static class SamplerMath
{
    /// <summary>Writes the softmax of <paramref name="logits"/> into <paramref name="probs"/>; entries at negative infinity contribute zero probability.</summary>
    public static void Softmax(ReadOnlySpan<float> logits, Span<float> probs)
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

    /// <summary>Returns the indices of <paramref name="values"/> ordered from largest to smallest.</summary>
    public static int[] ArgsortDescending(float[] values)
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
