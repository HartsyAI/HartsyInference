namespace HartsyInference.Vision;

/// <summary>Small activation helpers shared by ViT-family encoders. The exact (erf) GELU is the one HF
/// <c>ACT2FN["gelu"]</c> / <c>F.gelu</c> use — distinct from the backend's tanh-approximation <c>Gelu</c>,
/// which drifts ~1e-3 per layer and breaks transformer parity over many layers.</summary>
public static unsafe class VitOps
{
    /// <summary>Applies exact erf GELU in place over <paramref name="n"/> floats: <c>x·0.5·(1+erf(x/√2))</c>.</summary>
    public static void ExactGelu(float* p, long n)
    {
        const float invSqrt2 = 0.70710678118654752440f;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            p[i] = 0.5f * x * (1f + Erf(x * invSqrt2));
        }
    }

    /// <summary>Abramowitz-Stegun 7.1.26 erf approximation (max abs error ~1.5e-7).</summary>
    public static float Erf(float x)
    {
        int sign = x < 0 ? -1 : 1;
        x = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * x);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}
