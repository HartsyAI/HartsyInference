using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.GuidedEulerStep"/>: three identically shaped, nonempty F32 tensors; the
/// two predictions may be one tensor but neither may share storage with the latent the step mutates; finite guidance and delta;
/// a norm whose bounds are ordered and whose epsilon is nonnegative. Returns the element count and the last-dimension rows a
/// per-row norm reduces.</summary>
public static class GuidedEulerContract
{
    public static GuidedEulerGeometry Validate(Tensor z, Tensor cond, Tensor uncond, in GuidedEulerSpec spec)
    {
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(uncond);
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException($"GuidedEulerStep takes F32 tensors; got z {z.DType}, cond {cond.DType}, uncond {uncond.DType}.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException($"GuidedEulerStep requires identical shapes; got z {z.Shape}, cond {cond.Shape}, uncond {uncond.Shape}.", nameof(cond));
        if (z.Shape.Rank < 1 || z.ElementCount <= 0)
            throw new ArgumentException("GuidedEulerStep requires nonempty tensors.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException("GuidedEulerStep predictions may share storage with each other, but neither may share it with the latent.", nameof(z));
        if (!float.IsFinite(spec.Guidance))
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Guidance, "Guidance must be finite.");
        if (!float.IsFinite(spec.Delta))
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Delta, "The Euler delta must be finite.");
        if (!Enum.IsDefined(spec.Form) || !Enum.IsDefined(spec.Norm.Kind))
            throw new ArgumentOutOfRangeException(nameof(spec), spec, "Unknown guidance form or norm.");
        GuidanceNorm norm = spec.Norm;
        if (norm.Kind != GuidanceNormKind.None)
        {
            if (!float.IsFinite(norm.Eps) || norm.Eps < 0f)
                throw new ArgumentOutOfRangeException(nameof(spec), norm.Eps, "The norm epsilon must be finite and nonnegative.");
            if (float.IsNaN(norm.Min) || float.IsNaN(norm.Max) || norm.Min < 0f || norm.Max < norm.Min)
                throw new ArgumentOutOfRangeException(nameof(spec), norm, "The norm bounds must satisfy 0 <= Min <= Max.");
        }
        long width = z.Shape[z.Shape.Rank - 1];
        if (width <= 0)
            throw new ArgumentException($"GuidedEulerStep requires a positive last dimension; got {z.Shape}.", nameof(z));
        return new GuidedEulerGeometry(z.ElementCount, z.ElementCount / width, width);
    }
}
