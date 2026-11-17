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

    /// <summary>The rotation <c>(w, x, y, z) ⊗ q_R</c>, the turn <c>q_R = (√½, √½, 0, 0)</c> (90° about x, the same map as <c>(x, y, z) → (x, −z, y)</c> for positions) composed AFTER the
    /// splat's own rotation, in the same w-first order: as matrices <c>R_q · T</c>. This is what the TRELLIS and SAM 3D builders have always written, and the output is pinned byte for byte.
    /// It is not the product the TRELLIS reference writes: <c>save_ply</c> applies <c>T · R_q</c> (<c>q_R ⊗ q</c>), which differs by a rotation about x unless the splat's own rotation is about x;
    /// the two agree only for such splats. Changing the builders to the reference's order is a change of output and needs a render check, so it is a separate change.</summary>
    public static (float W, float X, float Y, float Z) TurnToYUpAfter(float w, float x, float y, float z)
        => (C45 * w - C45 * x, C45 * x + C45 * w, C45 * y + C45 * z, C45 * z - C45 * y);
}
