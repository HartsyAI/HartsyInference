using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Shared activation helpers for models that need exact (erf-based) GELU, which differs from the
/// backend's tanh-approximation <c>Gelu</c>. Used by the S3 speech tokenizer and the BERT encoder, both of
/// which were trained with exact GELU / exported with an Erf op.</summary>
internal static unsafe class Activations
{
    /// <summary>Exact GELU in place: <c>x · 0.5 · (1 + erf(x/√2))</c>.</summary>
    public static void ErfGelu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        const float invSqrt2 = 0.70710678118654752f;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            p[i] = v * 0.5f * (1f + Erf(v * invSqrt2));
        }
    }

    /// <summary>erf via Abramowitz &amp; Stegun 7.1.26 (max abs error ~1.5e-7).</summary>
    public static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        float ax = MathF.Abs(x);
        float tt = 1f / (1f + 0.3275911f * ax);
        float y = 1f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt
            + 0.254829592f) * tt * MathF.Exp(-ax * ax);
        return sign * y;
    }
}
