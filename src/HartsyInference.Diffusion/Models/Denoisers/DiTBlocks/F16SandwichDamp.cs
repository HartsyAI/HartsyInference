namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The F16 sandwich damp shared by the blocks whose sublayer projections feed straight into an RMSNorm
/// (<see cref="ZImageBlock"/>, <see cref="Ideogram4Block"/>). The projection weight is scaled by <see cref="Factor"/>
/// through <see cref="Core.Tensors.Tensor.Fp8ScaleFactor"/> (folded into the GEMM alpha, zero extra kernels) so its
/// raw output fits F16's 65504.
///
/// <para>The identity that makes it exact is <c>RMSNorm(c·x, c²·eps) ≡ RMSNorm(x, eps)</c>, not
/// <c>RMSNorm(c·x, eps)</c>: the norm that consumes a damped output must take <see cref="NormEps"/>. With the plain eps
/// the damp divides the mean square by 4096 while eps stays put, so a sublayer whose output is small is crushed toward
/// zero instead of normalized — the image loses contrast and detail, not range.</para></summary>
internal static class F16SandwichDamp
{
    /// <summary>Power of two, so F16 loses no relative precision to the shift.</summary>
    public const float Factor = 1.0f / 64.0f;

    /// <summary>The eps for the RMSNorm that consumes a projection damped by <see cref="Factor"/>.</summary>
    public static float NormEps(float eps, bool damped) => damped ? eps * Factor * Factor : eps;
}
