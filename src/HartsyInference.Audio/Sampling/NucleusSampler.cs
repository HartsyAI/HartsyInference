using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Sampling;

/// <summary>Shared autoregressive token-sampling core: temperature → softmax → top-k → top-p (nucleus)
/// → multinomial draw. Used by every AR audio LM that samples discrete tokens (CosyVoice's speech-token
/// sampler, Spark-TTS's BiCodec-token sampler, …) so there is ONE filtered-draw implementation rather
/// than a copy per model. Callers layer model-specific concerns (repetition penalty, RAS, candidate
/// windows) on top by pre-shaping the logit buffer and choosing the candidate count.</summary>
public static class NucleusSampler
{
    /// <summary>Draws one token index in <c>[0, count)</c> from <paramref name="logits"/>. Applies
    /// temperature, softmax, top-k and top-p filtering, then samples with the supplied RNG state.
    /// <paramref name="maskToken"/> (if in range) is forced to zero probability — used for ancestral /
    /// repetition re-rolls. Deterministic for a fixed RNG state.</summary>
    public static int Draw(Span<float> logits, int count, float temperature, int topK, float topP,
        ref uint rng, int maskToken = -1, float minP = 0f)
    {
        float temp = temperature > 0 ? temperature : 1f;
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

        // top-k + top-p (+ optional min-p) over an index list sorted by probability (descending).
        int[] order = ArgsortDescending(probs, count);
        int k = topK > 0 ? Math.Min(topK, count) : count;
        float minPThreshold = minP > 0f ? minP * probs[order[0]] : 0f;
        float cumulative = 0f;
        int keep = 0;
        for (int rank = 0; rank < k; rank++)
        {
            // min-p: drop tail tokens whose probability is below minP × top probability.
            if (minPThreshold > 0f && rank > 0 && probs[order[rank]] < minPThreshold) break;
            cumulative += probs[order[rank]];
            keep = rank + 1;
            if (topP > 0 && topP < 1f && cumulative >= topP) break;
        }

        // Renormalize the kept set and draw.
        float keptSum = 0f;
        for (int rank = 0; rank < keep; rank++) keptSum += probs[order[rank]];
        if (keptSum <= 0f) return order[0];
        float r = DeterministicRng.NextUniform(ref rng) * keptSum;
        float acc = 0f;
        for (int rank = 0; rank < keep; rank++)
        {
            acc += probs[order[rank]];
            if (r <= acc) return order[rank];
        }
        return order[keep - 1];
    }

    private static int[] ArgsortDescending(float[] values, int count)
    {
        int[] idx = new int[count];
        for (int i = 0; i < count; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => values[b].CompareTo(values[a]));
        return idx;
    }
}
