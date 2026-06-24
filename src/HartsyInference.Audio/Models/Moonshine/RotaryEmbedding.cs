using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

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
/// <summary>Llama-3 "NTK-by-parts" RoPE scaling parameters (HF <c>rope_scaling</c> with <c>rope_type: "llama3"</c>).
/// Rescales the base inverse frequencies so a model trained at <see cref="OriginalMaxPos"/> extrapolates to longer
/// context: high-frequency pairs are untouched, low-frequency pairs are divided by <see cref="Factor"/>, and the
/// medium band is smoothly interpolated between the two. Used by Chatterbox's T3 (theta 500000, factor 8) and the
/// Llama-3 family (CSM, Orpheus).</summary>
internal readonly record struct RopeScaling(float Factor, float LowFreqFactor, float HighFreqFactor, int OriginalMaxPos)
{
    /// <summary>Standard Llama-3 / Llama-3.1 defaults (factor 8, low 1, high 4, original context 8192).</summary>
    public static RopeScaling Llama3 => new(8f, 1f, 4f, 8192);
}

internal static class RotaryEmbedding
{
    private static readonly Dictionary<(int RotaryDim, float Theta, int MaxPos), (float[] Cos, float[] Sin)> _cache = new();
    private static readonly Dictionary<(int RotaryDim, float Theta, int MaxPos, float F, float Lo, float Hi, int Orig), (float[] Cos, float[] Sin)> _scaledCache = new();
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

            // inv_freq[i] = 1 / (theta ^ (2i / rotary_dim)) for i in [0, half)
            double[] invFreq = new double[half];
            for (int i = 0; i < half; i++)
                invFreq[i] = 1.0 / Math.Pow(theta, (double)(2 * i) / rotaryDim);

            (float[] Cos, float[] Sin) tables = BuildTables(invFreq, maxPos);
            _cache[key] = tables;
            return tables;
        }
    }

    /// <summary>Same as <see cref="GetTables(int,float,int)"/>, but applies Llama-3 NTK-by-parts scaling to the
    /// base inverse frequencies when <paramref name="scaling"/> is non-null. With null scaling this is bit-identical
    /// to the unscaled overload.</summary>
    public static (float[] Cos, float[] Sin) GetTables(int rotaryDim, float theta, int maxPos, RopeScaling? scaling)
    {
        if (scaling is null) return GetTables(rotaryDim, theta, maxPos);

        RopeScaling sc = scaling.Value;
        (int RotaryDim, float Theta, int MaxPos, float F, float Lo, float Hi, int Orig) key
            = (rotaryDim, theta, maxPos, sc.Factor, sc.LowFreqFactor, sc.HighFreqFactor, sc.OriginalMaxPos);
        lock (_cacheLock)
        {
            if (_scaledCache.TryGetValue(key, out (float[] Cos, float[] Sin) cached)) return cached;

            int half = rotaryDim / 2;

            // Base inv_freq, then the Llama-3 piecewise rescale (HF _compute_llama3_parameters).
            double lowFreqWavelen = sc.OriginalMaxPos / sc.LowFreqFactor;
            double highFreqWavelen = sc.OriginalMaxPos / sc.HighFreqFactor;
            double[] invFreq = new double[half];
            for (int i = 0; i < half; i++)
            {
                double baseInv = 1.0 / Math.Pow(theta, (double)(2 * i) / rotaryDim);
                double wavelen = 2.0 * Math.PI / baseInv;
                if (wavelen < highFreqWavelen)
                {
                    invFreq[i] = baseInv;                       // high frequency: untouched
                }
                else if (wavelen > lowFreqWavelen)
                {
                    invFreq[i] = baseInv / sc.Factor;           // low frequency: divided by factor
                }
                else
                {
                    // medium band: smooth interpolation between baseInv/factor and baseInv
                    double smooth = (sc.OriginalMaxPos / wavelen - sc.LowFreqFactor) / (sc.HighFreqFactor - sc.LowFreqFactor);
                    invFreq[i] = (1.0 - smooth) * (baseInv / sc.Factor) + smooth * baseInv;
                }
            }

            (float[] Cos, float[] Sin) tables = BuildTables(invFreq, maxPos);
            _scaledCache[key] = tables;
            return tables;
        }
    }

    /// <summary>Builds the <c>[maxPos, invFreq.Length]</c> cos / sin tables from a set of inverse frequencies.</summary>
    private static (float[] Cos, float[] Sin) BuildTables(double[] invFreq, int maxPos)
    {
        int half = invFreq.Length;
        float[] cos = new float[maxPos * half];
        float[] sin = new float[maxPos * half];
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
        return (cos, sin);
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
