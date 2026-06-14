using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>ACE-Step's guidance family (<c>acestep/apg_guidance.py</c>): standard CFG, <b>APG</b> (Adaptive Projected
/// Guidance — momentum-buffered cond−uncond delta, norm-thresholded, decomposed into components parallel/orthogonal
/// to the conditional prediction; default η=0 keeps only the orthogonal part), and <b>CFG-Zero★</b> (per-step optimal
/// scaling α = ⟨cond,uncond⟩/‖uncond‖², optional zero-init on the first step). Useful beyond ACE-Step — APG is being
/// adopted by Flux/SD3 community workflows.</summary>
public static unsafe class AceStepGuidance
{
    /// <summary>Per-trajectory running momentum state for APG (<c>avg = m·avg + diff</c>, default m = −0.75).
    /// Create one per conditioning stream per generation; do not reuse across songs.</summary>
    public sealed class MomentumBuffer : IDisposable
    {
        private readonly float _momentum;
        private Tensor? _running;

        public MomentumBuffer(float momentum = -0.75f) => _momentum = momentum;

        /// <summary>Updates the buffer with the current diff and overwrites <paramref name="diff"/> with the running value.</summary>
        public void UpdateInPlace(Tensor diff)
        {
            long n = diff.Shape.ElementCount;
            float* dp = (float*)diff.DataPointer;
            if (_running is null)
            {
                _running = new Tensor(diff.Shape, DType.F32);
                Buffer.MemoryCopy(dp, (float*)_running.DataPointer, n * 4, n * 4);
                return;
            }
            float* rp = (float*)_running.DataPointer;
            for (long i = 0; i < n; i++)
            {
                rp[i] = _momentum * rp[i] + dp[i];
                dp[i] = rp[i];
            }
        }

        public void Dispose()
        {
            _running?.Dispose();
            _running = null;
        }
    }

    /// <summary>Standard CFG in place on <paramref name="cond"/>: <c>uncond + g·(cond − uncond)</c>.</summary>
    public static void Cfg(Tensor cond, Tensor uncond, float scale)
    {
        long n = cond.Shape.ElementCount;
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++) cp[i] = up[i] + scale * (cp[i] - up[i]);
    }

    /// <summary>APG in place on <paramref name="cond"/>: momentum-buffered diff, norm-thresholded, then
    /// <c>cond + (g−1)·(diff_orth + η·diff_par)</c> with the decomposition along the conditional prediction.</summary>
    public static void Apg(Tensor cond, Tensor uncond, float scale, MomentumBuffer? momentum,
        float normThreshold = 2.5f, float eta = 0.0f)
    {
        long n = cond.Shape.ElementCount;
        Tensor diff = new Tensor(cond.Shape, DType.F32);
        float* dp = (float*)diff.DataPointer;
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++) dp[i] = cp[i] - up[i];

        momentum?.UpdateInPlace(diff);

        // Norm threshold: diff ← diff · threshold / max(‖diff‖, threshold).
        double diffNormSq = 0;
        for (long i = 0; i < n; i++) diffNormSq += (double)dp[i] * dp[i];
        float diffNorm = (float)Math.Sqrt(diffNormSq);
        if (normThreshold > 0 && diffNorm > normThreshold)
        {
            float rescale = normThreshold / diffNorm;
            for (long i = 0; i < n; i++) dp[i] *= rescale;
        }

        // Parallel/orthogonal decomposition along cond.
        double condNormSq = 0, dot = 0;
        for (long i = 0; i < n; i++)
        {
            condNormSq += (double)cp[i] * cp[i];
            dot += (double)dp[i] * cp[i];
        }
        float parCoef = (float)(dot / Math.Max(condNormSq, 1e-16));
        float g = scale - 1f;
        for (long i = 0; i < n; i++)
        {
            float par = parCoef * cp[i];
            float orth = dp[i] - par;
            cp[i] += g * (orth + eta * par);
        }
        diff.Dispose();
    }

    /// <summary>CFG-Zero★ in place on <paramref name="cond"/>: rescales the unconditional branch by
    /// <c>α = ⟨cond,uncond⟩/‖uncond‖²</c> before CFG; with <paramref name="zeroInit"/> the prediction is zeroed
    /// (the recommended first-step behavior).</summary>
    public static void CfgZeroStar(Tensor cond, Tensor uncond, float scale, bool zeroInit = false)
    {
        long n = cond.Shape.ElementCount;
        float* cp = (float*)cond.DataPointer;
        if (zeroInit)
        {
            new Span<float>(cp, (int)Math.Min(int.MaxValue, n)).Clear();
            return;
        }
        float* up = (float*)uncond.DataPointer;
        double dot = 0, uncondNormSq = 0;
        for (long i = 0; i < n; i++)
        {
            dot += (double)cp[i] * up[i];
            uncondNormSq += (double)up[i] * up[i];
        }
        float alpha = (float)(dot / Math.Max(uncondNormSq, 1e-16));
        for (long i = 0; i < n; i++)
        {
            float scaledUncond = up[i] * alpha;
            cp[i] = scaledUncond + scale * (cp[i] - scaledUncond);
        }
    }
}
