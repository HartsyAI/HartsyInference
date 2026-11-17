namespace HartsyInference.ThreeD.Geometry;

/// <summary>The per-splat activations that the 3D-Gaussian decoders (TRELLIS SLAT, SAM 3D, Idol) share, written the way a
/// <see cref="GaussianSplatCloud"/> stores them: log scale, logit opacity and a w-first rotation quaternion. Each decoder owns its
/// layout, offsets and defaults; these are the formulas that do not differ between them.</summary>
public static class GaussianSplatActivations
{
    /// <summary>Cos and sin of 45°: the y-up to z-up turn is 90° about x, the quaternion <c>(√½, √½, 0, 0)</c>.</summary>
    private const float C45 = 0.70710678f;

    /// <summary><c>log(1 + exp(x))</c>, which torch's <c>F.softplus</c> returns unchanged above 20.</summary>
    public static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));

    /// <summary>The <c>b</c> with <c>softplus(b) = x</c>: <c>x + log(1 − exp(−x))</c>.</summary>
    public static float SoftplusInverse(float x) => x + MathF.Log(1f - MathF.Exp(-x));

    /// <summary>The log of the splat scale <c>sqrt(softplus(raw + scaleOffset)² + minKernel²)</c>: the decoder's raw scale channel,
    /// biased so a zero output gives the initial size, with a floor of the 3D filter's kernel.</summary>
    /// <param name="raw">The decoder's raw scale channel, already multiplied by its learning-rate factor.</param>
    /// <param name="scaleOffset"><see cref="SoftplusInverse"/> of the initial scale.</param>
    /// <param name="minKernel">The 3D filter kernel size the scale never goes below.</param>
    public static float LogScale(float raw, float scaleOffset, float minKernel)
    {
        float sp = Softplus(raw + scaleOffset);
        return 0.5f * MathF.Log(sp * sp + minKernel * minKernel);
    }

    /// <summary>The rotation <c>q_R ⊗ (w, x, y, z)</c> that turns a z-up splat to the y-up convention viewers use (a 90° turn about
    /// x, matching <c>(x, y, z) → (x, −z, y)</c> for positions), in the same w-first order.</summary>
    public static (float W, float X, float Y, float Z) TurnToYUp(float w, float x, float y, float z)
        => (C45 * w - C45 * x, C45 * x + C45 * w, C45 * y + C45 * z, C45 * z - C45 * y);
}
