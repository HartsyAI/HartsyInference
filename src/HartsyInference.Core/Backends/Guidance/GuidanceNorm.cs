namespace HartsyInference.Core.Backends.Guidance;

/// <summary>The rescale a guided prediction gets before the Euler update: over the elements <see cref="Kind"/> names,
/// <c>ratio = ‖cond‖₂ / (‖guided‖₂ + Eps)</c> (0 when that denominator is 0), clamped to <c>[Min, Max]</c>, multiplies the
/// guided prediction. <see cref="None"/> leaves it untouched.</summary>
public readonly record struct GuidanceNorm(GuidanceNormKind Kind, float Min, float Max, float Eps)
{
    /// <summary>No rescale.</summary>
    public static GuidanceNorm None => new(GuidanceNormKind.None, 0f, float.PositiveInfinity, 0f);

    /// <summary>One ratio over the whole tensor, clamped to <c>[min, max]</c> (Lance: <c>max</c> 1, <c>eps</c> 1e-8).</summary>
    public static GuidanceNorm Global(float min, float max = 1f, float eps = 1e-8f) => new(GuidanceNormKind.Global, min, max, eps);

    /// <summary>One unclamped ratio per last-dimension row (Lumina 2: <c>eps</c> 1e-12).</summary>
    public static GuidanceNorm PerRow(float eps = 1e-12f) => new(GuidanceNormKind.PerRow, 0f, float.PositiveInfinity, eps);
}
