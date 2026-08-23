using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Layers;

/// <summary>Scalar activation helpers shared across the audio models: exact (erf-based) GELU, which differs
/// from the backend's tanh-approximation <c>Gelu</c> and is what PyTorch's default <c>nn.GELU()</c> computes,
/// plus the overflow-safe logistic sigmoid used by the recurrent cells.</summary>
internal static unsafe class Activations
{
    private const float InvSqrt2 = 0.70710678118654752f;

    /// <summary>Exact GELU in place: <c>x · 0.5 · (1 + erf(x/√2))</c>.</summary>
    public static void ErfGelu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) p[i] = ErfGelu(p[i]);
    }

    /// <summary>Exact GELU of a single value: <c>x · 0.5 · (1 + erf(x/√2))</c>.</summary>
    public static float ErfGelu(float x) => x * 0.5f * (1f + Erf(x * InvSqrt2));

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

    /// <summary>Logistic sigmoid computed from the sign-matching branch so <c>exp</c> never overflows to Inf.</summary>
    public static float SigmoidS(float x)
    {
        if (x >= 0f) return 1f / (1f + MathF.Exp(-x));
        float ex = MathF.Exp(x);
        return ex / (1f + ex);
    }
}
