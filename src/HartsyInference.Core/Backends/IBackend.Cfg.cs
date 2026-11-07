using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Classifier-free-guidance Euler steps in their plain, renormalized and per-row-normalized forms.
public partial interface IBackend
{
    /// <summary>CFG combine + Euler step, in-place on <paramref name="z"/>:
    /// <c>v = guidance*pos + (1-guidance)*neg; z += v*delta</c>. <paramref name="pos"/> and
    /// <paramref name="neg"/> may be the same tensor; neither may alias the mutated <paramref name="z"/>.</summary>
    unsafe void CfgEulerStep(Tensor z, Tensor pos, Tensor neg, float guidance, float delta)
    {
        if (z.DType != DType.F32 || pos.DType != DType.F32 || neg.DType != DType.F32)
            throw new NotSupportedException("CfgEulerStep default fallback only supports F32.");
        if (!z.Shape.Equals(pos.Shape) || !z.Shape.Equals(neg.Shape))
            throw new ArgumentException(
                $"CfgEulerStep requires identical shapes; got z={z.Shape}, pos={pos.Shape}, neg={neg.Shape}.");
        if (z.ElementCount <= 0)
            throw new ArgumentException("CfgEulerStep requires nonempty tensors.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(pos) || z.HasOverlappingHostStorageWithoutSync(neg))
            throw new ArgumentException("CfgEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        long count = z.ElementCount;
        float* pZ = (float*)z.DataPointer;
        float* pPos = (float*)pos.DataPointer;
        float* pNeg = (float*)neg.DataPointer;
        for (long i = 0; i < count; i++)
        {
            float v = guidance * pPos[i] + (1.0f - guidance) * pNeg[i];
            pZ[i] = pZ[i] + v * delta;
        }
    }

    /// <summary>Global-renormalized CFG plus Euler update, in-place on <paramref name="z"/> while leaving <paramref name="cond"/> and <paramref name="uncond"/> unchanged: <c>guided = uncond + guidance·(cond−uncond)</c>, <c>scale = clamp(‖cond‖₂/(‖guided‖₂+1e-8), renormMin, 1)</c>, <c>z += delta·scale·guided</c>. The two prediction inputs may alias each other; neither may alias the mutated <paramref name="z"/>.</summary>
    unsafe void CfgRenormEulerStep(
        Tensor z, Tensor cond, Tensor uncond, float guidance, float delta, float renormMin)
    {
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException("CfgRenormEulerStep default fallback only supports F32.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"CfgRenormEulerStep requires identical shapes; got z={z.Shape}, cond={cond.Shape}, uncond={uncond.Shape}.");
        if (z.ElementCount <= 0)
            throw new ArgumentException("CfgRenormEulerStep requires nonempty tensors.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException(
                "CfgRenormEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        if (!float.IsFinite(renormMin) || renormMin < 0f || renormMin > 1f)
            throw new ArgumentOutOfRangeException(nameof(renormMin), "Renorm minimum must be finite and in [0,1].");

        long count = z.ElementCount;
        if (delta == 0f) return;

        float* pZ = (float*)z.DataPointer;
        float* pCond = (float*)cond.DataPointer;
        float* pUncond = (float*)uncond.DataPointer;
        double normCondSq = 0.0;
        double normGuidedSq = 0.0;
        for (long i = 0; i < count; i++)
        {
            float c = pCond[i];
            float guided = pUncond[i] + guidance * (c - pUncond[i]);
            normCondSq += (double)c * c;
            normGuidedSq += (double)guided * guided;
        }

        double ratio = Math.Sqrt(normCondSq) / (Math.Sqrt(normGuidedSq) + 1e-8);
        float scale = (float)Math.Clamp(ratio, renormMin, 1.0);
        for (long i = 0; i < count; i++)
        {
            float guided = pUncond[i] + guidance * (pCond[i] - pUncond[i]);
            guided *= scale;
            pZ[i] += guided * delta;
        }
    }

    /// <summary>Last-dimension-normalized CFG plus Euler update, in-place on <paramref name="z"/> while leaving <paramref name="cond"/> and <paramref name="uncond"/> unchanged. For every row formed by flattening all dimensions except the last: <c>guided = uncond + guidance·(cond−uncond)</c>, <c>guided *= ‖cond‖₂/(‖guided‖₂+eps)</c>, and <c>z += delta·guided</c>. The two prediction inputs may alias each other; neither may alias the mutated <paramref name="z"/>.</summary>
    unsafe void CfgNormalizedEulerStep(
        Tensor z, Tensor cond, Tensor uncond, float guidance, float delta, float eps = 1e-12f)
    {
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException("CfgNormalizedEulerStep default fallback only supports F32.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"CfgNormalizedEulerStep requires identical shapes; got z={z.Shape}, cond={cond.Shape}, uncond={uncond.Shape}.");
        if (z.Shape.Rank < 1 || z.ElementCount <= 0 || z.Shape[z.Shape.Rank - 1] <= 0)
            throw new ArgumentException("CfgNormalizedEulerStep requires a nonempty tensor with a positive last dimension.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException(
                "CfgNormalizedEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        if (!float.IsFinite(eps) || eps < 0f)
            throw new ArgumentOutOfRangeException(nameof(eps), "Normalization epsilon must be finite and nonnegative.");
        if (delta == 0f) return;

        long lastDim = z.Shape[z.Shape.Rank - 1];
        long rows = z.ElementCount / lastDim;
        float* pZ = (float*)z.DataPointer;
        float* pCond = (float*)cond.DataPointer;
        float* pUncond = (float*)uncond.DataPointer;
        for (long row = 0; row < rows; row++)
        {
            long rowOffset = row * lastDim;
            double condSq = 0.0;
            double guidedSq = 0.0;
            for (long d = 0; d < lastDim; d++)
            {
                long i = rowOffset + d;
                float c = pCond[i];
                float guided = pUncond[i] + guidance * (c - pUncond[i]);
                condSq += (double)c * c;
                guidedSq += (double)guided * guided;
            }

            // A zero denominator (eps=0 with an all-zero guided row) contributes nothing either way — use
            // ratio=0 rather than letting 0/0 produce NaN and poison z through 0·NaN.
            double denom = Math.Sqrt(guidedSq) + eps;
            float ratio = denom > 0.0 ? (float)(Math.Sqrt(condSq) / denom) : 0f;
            for (long d = 0; d < lastDim; d++)
            {
                long i = rowOffset + d;
                float guided = pUncond[i] + guidance * (pCond[i] - pUncond[i]);
                pZ[i] += delta * (guided * ratio);
            }
        }
    }
}
