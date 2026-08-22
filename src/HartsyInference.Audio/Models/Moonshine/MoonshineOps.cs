using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>Moonshine-specific low-level helpers; the headline op is <see cref="LayerNormNoBias"/>.</summary>
// Moonshine uses nn.LayerNorm(hidden, bias=False) for every norm, but IBackend's LayerNorm requires a
// bias tensor — rather than allocate a per-instance zero-bias tensor, we inline the computation here.
internal static unsafe class MoonshineOps
{
    /// <summary>Normalizes the last dim of a <c>[..., d]</c> tensor to zero mean / unit variance, then scales by <paramref name="weight"/> — no bias add.</summary>
    public static void LayerNormNoBias(Tensor output, Tensor input, Tensor weight, float eps)
    {
        int d = (int)input.Shape[input.Shape.Rank - 1];
        long total = input.ElementCount;
        long n = total / d;
        float* xp = (float*)input.DataPointer;
        float* yp = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;

        for (long row = 0; row < n; row++)
        {
            float* x = xp + row * d;
            float* y = yp + row * d;

            // mean over the last dim
            float mean = 0f;
            for (int i = 0; i < d; i++) mean += x[i];
            mean /= d;

            // variance over the last dim
            float sq = 0f;
            for (int i = 0; i < d; i++) { float v = x[i] - mean; sq += v * v; }
            float invStd = 1f / MathF.Sqrt(sq / d + eps);

            // affine (scale only — no bias)
            for (int i = 0; i < d; i++) y[i] = (x[i] - mean) * invStd * wp[i];
        }
    }

    /// <summary>Exact GELU (<c>0.5 * x * (1 + erf(x / sqrt(2)))</c>) in place, avoiding three extra <see cref="Tensor"/> allocations between the conv stem layers.</summary>
    public static void GeluInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        const float invSqrt2 = 0.7071067811865475f;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            p[i] = 0.5f * x * (1f + Erf(x * invSqrt2));
        }
    }

    /// <summary>Abramowitz &amp; Stegun approximation of <c>erf(x)</c>, accurate to ~1.5e-7 — well below FP32 rounding error.</summary>
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        const float p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    /// <summary>Element-wise tanh, in place.</summary>
    public static void TanhInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = MathF.Tanh(p[i]);
    }
}
