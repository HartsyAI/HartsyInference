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
        float invTemp = 1f / temp;
        int k = topK > 0 ? Math.Min(topK, count) : count;

        // Fast path: when top-k bounds the kept set to a small K (every AR audio sampler: K=50 over a 2k–3k vocab),
        // select the top-K by (temperature-scaled) logit with a bounded scan instead of allocating two length-`count`
        // arrays and running a full O(count·log count) delegate-comparator Array.Sort every single draw. The draw is
        // mathematically identical — softmax is monotonic, so top-K by logit == top-K by probability, and the
        // multinomial walk over the renormalized kept set only depends on the kept logits' relative exp weights
        // (the full-vocab softmax denominator cancels in the renormalization). Only the top-p / min-p thresholds
        // (which compare against absolute probabilities) need the full denominator, computed on demand below.
        if (k < count && k <= 1024)
        {
            Span<int> ki = stackalloc int[k];
            Span<float> kv = stackalloc float[k];   // temperature-scaled logits of the current top-k
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
            // Sort the k selected entries descending by scaled logit (insertion sort — k is tiny).
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
            // Draw over the kept set using exp weights relative to the top logit (the denominator cancels).
            float kmax = kv[0], sumE = 0f;
            Span<float> ew = stackalloc float[keep];
            for (int rank = 0; rank < keep; rank++) { float ex = MathF.Exp(kv[rank] - kmax); ew[rank] = ex; sumE += ex; }
            if (sumE <= 0f) return ki[0];
            float r = DeterministicRng.NextUniform(ref rng) * sumE;
            float acc = 0f;
            for (int rank = 0; rank < keep; rank++) { acc += ew[rank]; if (r <= acc) return ki[rank]; }
            return ki[keep - 1];
        }

        // Fallback: unbounded top-k (k == count). Full softmax + sort.
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

        int[] order = ArgsortDescending(probs, count);
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

    private static int[] ArgsortDescending(float[] values, int count)
    {
        int[] idx = new int[count];
        for (int i = 0; i < count; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => values[b].CompareTo(values[a]));
        return idx;
    }
}
