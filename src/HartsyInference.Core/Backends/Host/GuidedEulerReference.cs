using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the guided Euler contract (<c>docs/BACKEND_OPS.md</c>): the guided prediction of
/// <see cref="GuidanceForm"/>, rescaled by the <see cref="GuidanceNorm"/> ratio (norms summed in F64), added to the latent
/// times delta, in place. The <see cref="IBackend"/> default and the oracle every kernel is checked against.</summary>
public static unsafe class GuidedEulerReference
{
    /// <summary>The guided value of one element under <paramref name="form"/>.</summary>
    public static float Blend(GuidanceForm form, float guidance, float cond, float uncond)
        => form == GuidanceForm.Lerp ? guidance * cond + (1.0f - guidance) * uncond : uncond + guidance * (cond - uncond);

    /// <summary>The rescale of a guided prediction whose squared norm is <paramref name="guidedSq"/> against a conditional one of
    /// <paramref name="condSq"/>: <c>‖cond‖ / (‖guided‖ + eps)</c>, 0 when that denominator is 0, clamped to the norm's bounds.</summary>
    public static float Ratio(double condSq, double guidedSq, in GuidanceNorm norm)
    {
        double denom = Math.Sqrt(guidedSq) + norm.Eps;
        double ratio = denom > 0.0 ? Math.Sqrt(condSq) / denom : 0.0;
        return (float)Math.Clamp(ratio, norm.Min, norm.Max);
    }

    public static void Apply(Tensor z, Tensor cond, Tensor uncond, in GuidedEulerSpec spec)
    {
        GuidedEulerGeometry g = GuidedEulerContract.Validate(z, cond, uncond, spec);
        if (spec.Delta == 0f) return;
        float* pZ = (float*)z.DataPointer;
        float* pCond = (float*)cond.DataPointer;
        float* pUncond = (float*)uncond.DataPointer;
        float guidance = spec.Guidance, delta = spec.Delta;
        GuidanceForm form = spec.Form;
        switch (spec.Norm.Kind)
        {
            case GuidanceNormKind.None:
                for (long i = 0; i < g.Count; i++)
                    pZ[i] = pZ[i] + Blend(form, guidance, pCond[i], pUncond[i]) * delta;
                return;
            case GuidanceNormKind.Global:
                Update(pZ, pCond, pUncond, 0, g.Count, spec);
                return;
            default:
                for (long row = 0; row < g.Rows; row++)
                    Update(pZ, pCond, pUncond, row * g.Width, g.Width, spec);
                return;
        }
    }

    /// <summary>Rescales and applies the guided prediction over <paramref name="count"/> elements from <paramref name="start"/>.</summary>
    private static void Update(float* pZ, float* pCond, float* pUncond, long start, long count, in GuidedEulerSpec spec)
    {
        double condSq = 0.0, guidedSq = 0.0;
        long end = start + count;
        for (long i = start; i < end; i++)
        {
            float c = pCond[i];
            float guided = Blend(spec.Form, spec.Guidance, c, pUncond[i]);
            condSq += (double)c * c;
            guidedSq += (double)guided * guided;
        }
        float scale = Ratio(condSq, guidedSq, spec.Norm);
        for (long i = start; i < end; i++)
        {
            float guided = Blend(spec.Form, spec.Guidance, pCond[i], pUncond[i]);
            guided *= scale;
            pZ[i] += guided * spec.Delta;
        }
    }
}
