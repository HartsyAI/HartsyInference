using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Moonshine;

/// <summary>Rotary Position Embedding (RoPE) — **interleaved (GPT-J) convention**, with
/// optional partial rotation. Moonshine uses this convention (verified against
/// <c>transformers/models/moonshine/modeling_moonshine.py</c>) — pairs of consecutive
/// dims share a frequency:
/// <code>
///   for d in 0 .. rotary_dim, step 2:
///     i = d / 2
///     c, s = cos(p * inv_freq[i]), sin(p * inv_freq[i])
///     q'[d]   = q[d]   * c - q[d+1] * s
///     q'[d+1] = q[d+1] * c + q[d]   * s
///   q'[rotary_dim..head_dim]  =  q[rotary_dim..head_dim]    (pass-through)
/// </code>
/// This is the GPT-J / Phi convention. Llama / GPT-NeoX use the alternative
/// "split-halves" form (used by <c>WhisperOps</c> if we ever add it). They are NOT
/// interchangeable — picking the wrong one produces fluent-looking but degenerate
/// transcriptions (we found this the hard way on the first Moonshine run).
///
/// <para><b>Table layout:</b> we store only the <c>rotary_dim/2</c> unique cos / sin
/// values per position, indexed by frequency pair. Apply reads <c>cos[pos][i]</c>
/// once and uses it for both <c>q[2i]</c> and <c>q[2i+1]</c>.</para></summary>
internal static class RotaryEmbedding
{
    private static readonly Dictionary<(int RotaryDim, float Theta, int MaxPos), (float[] Cos, float[] Sin)> _cache = new();
    private static readonly object _cacheLock = new();

    /// <summary>Returns cos / sin tables of shape <c>[maxPos, rotaryDim/2]</c>. Each
    /// row holds the <c>rotaryDim/2</c> unique frequencies for that position; the
    /// apply step pairs them with consecutive q/k dims (interleaved convention).</summary>
    public static (float[] Cos, float[] Sin) GetTables(int rotaryDim, float theta, int maxPos)
    {
        (int RotaryDim, float Theta, int MaxPos) key = (rotaryDim, theta, maxPos);
        lock (_cacheLock)
        {
            if (_cache.TryGetValue(key, out (float[] Cos, float[] Sin) cached)) return cached;

            int half = rotaryDim / 2;
            float[] cos = new float[maxPos * half];
            float[] sin = new float[maxPos * half];

            // inv_freq[i] = 1 / (theta ^ (2i / rotary_dim)) for i in [0, half)
            double[] invFreq = new double[half];
            for (int i = 0; i < half; i++)
                invFreq[i] = 1.0 / Math.Pow(theta, (double)(2 * i) / rotaryDim);

            for (int p = 0; p < maxPos; p++)
            {
                int row = p * half;
                for (int i = 0; i < half; i++)
                {
                    double angle = p * invFreq[i];
                    cos[row + i] = (float)Math.Cos(angle);
                    sin[row + i] = (float)Math.Sin(angle);
                }
            }

            _cache[key] = (cos, sin);
            return (cos, sin);
        }
    }

    /// <summary>Applies interleaved partial RoPE in-place to a tensor of shape
    /// <c>[1, H, S, D]</c>. The first <paramref name="rotaryDim"/> dims of each head
    /// are rotated in adjacent pairs <c>(2i, 2i+1)</c>; remaining dims pass through.</summary>
    public static unsafe void ApplyInPlace(Tensor t, int numHeads, int seqLen, int headDim, int rotaryDim, int posStart, float[] cosTable, float[] sinTable)
    {
        if (rotaryDim == 0) return;
        int half = rotaryDim / 2;
        float* p = (float*)t.DataPointer;

        // Layout [1, H, S, D]: element (h, s, d) at ((h*S)+s)*D + d.
        for (int h = 0; h < numHeads; h++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int pos = posStart + s;
                int baseOff = (h * seqLen + s) * headDim;
                int tableOff = pos * half;

                // Rotate adjacent pairs in the first rotaryDim dims.
                for (int i = 0; i < half; i++)
                {
                    int dEven = 2 * i;
                    int dOdd = 2 * i + 1;
                    float xe = p[baseOff + dEven];
                    float xo = p[baseOff + dOdd];
                    float c = cosTable[tableOff + i];
                    float si = sinTable[tableOff + i];
                    p[baseOff + dEven] = xe * c - xo * si;
                    p[baseOff + dOdd]  = xo * c + xe * si;
                }
                // Trailing dims [rotaryDim, headDim) untouched.
            }
        }
    }
}
