namespace HartsyInference.Core.Rope;

/// <summary>Shared RoPE inverse-frequency builder. Computes the base <c>inv_freq[k] = 1/theta^(2k/D)</c> and
/// applies a <see cref="RopeScaling"/> transform, returning the scaled frequencies plus the attention mscale
/// to bake into cos/sin. Each caller (LLM decoder, diffusion text encoders, Cosmos/Anima) keeps its own cos/sin
/// <i>layout</i> but obtains frequencies + mscale here, so the scaling math lives once.
///
/// <para>YaRN matches HuggingFace <c>_compute_yarn_parameters</c> exactly (ported from the GPT-OSS encoder,
/// validated &lt;1e-8); Llama3 matches <c>_compute_llama3_parameters</c>. Computation is in double precision;
/// callers cast to float.</para></summary>
public static class RopeFrequencyBuilder
{
    /// <summary>Base inverse frequencies <c>inv_freq[k] = 1 / theta^(2k/dim)</c>, length <c>dim/2</c>.</summary>
    public static double[] ComputeInvFreq(int dim, double theta)
    {
        int half = dim / 2;
        double[] invFreq = new double[half];
        for (int k = 0; k < half; k++) invFreq[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / dim);
        return invFreq;
    }

    /// <summary>Builds scaled inverse frequencies + the attention mscale (1.0 unless YaRN/LongRope).
    /// <paramref name="currentSeqLen"/> drives DynamicNtk / LongRope short-vs-long selection.</summary>
    public static (double[] InvFreq, double AttentionScaling) Build(int dim, double theta, RopeScaling? scaling, int currentSeqLen)
    {
        RopeScaling s = scaling ?? RopeScaling.None;
        int half = dim / 2;

        // DynamicNtk rescales the base before computing frequencies.
        double effectiveTheta = theta;
        if (s.Type == RopeScalingType.DynamicNtk && s.OriginalContextLength > 0 && currentSeqLen > s.OriginalContextLength)
        {
            double ratio = s.Factor * currentSeqLen / s.OriginalContextLength - (s.Factor - 1.0);
            effectiveTheta = theta * Math.Pow(ratio, (double)dim / (dim - 2));
        }

        double[] invFreq = ComputeInvFreq(dim, effectiveTheta);

        // Explicit per-frequency multiplier (GGUF rope_freqs.weight) wins over the formula — reproduces llama.cpp.
        if (s.InvFreqFactors is { Count: > 0 } factors)
        {
            int n = Math.Min(half, factors.Count);
            for (int k = 0; k < n; k++) invFreq[k] *= factors[k];
            return (invFreq, InferMscale(s));
        }

        switch (s.Type)
        {
            case RopeScalingType.None:
            case RopeScalingType.DynamicNtk:   // already folded into effectiveTheta
                return (invFreq, 1.0);

            case RopeScalingType.Linear:
                for (int k = 0; k < half; k++) invFreq[k] /= s.Factor;
                return (invFreq, 1.0);

            case RopeScalingType.Llama3:
                ApplyLlama3(invFreq, s);
                return (invFreq, 1.0);

            case RopeScalingType.Yarn:
                double mscale = ApplyYarn(invFreq, dim, theta, s);
                return (invFreq, mscale);

            case RopeScalingType.LongRope:
                IReadOnlyList<double>? table = currentSeqLen > s.OriginalContextLength ? s.LongFactor : s.ShortFactor;
                if (table is not null)
                    for (int k = 0; k < half && k < table.Count; k++) invFreq[k] /= table[k];
                return (invFreq, InferMscale(s));

            default:
                return (invFreq, 1.0);
        }
    }

    /// <summary>Llama-3 piecewise wavelength rescale (HF <c>_compute_llama3_parameters</c>), in place.</summary>
    private static void ApplyLlama3(double[] invFreq, RopeScaling s)
    {
        double origCtx = s.OriginalContextLength;
        if (origCtx <= 0) return;
        double lowWavelen = origCtx / s.LowFreqFactor;
        double highWavelen = origCtx / s.HighFreqFactor;
        for (int k = 0; k < invFreq.Length; k++)
        {
            double wavelen = 2.0 * Math.PI / invFreq[k];
            if (wavelen > lowWavelen)
            {
                invFreq[k] /= s.Factor;                       // low frequency: interpolate
            }
            else if (wavelen >= highWavelen)
            {
                double smooth = (origCtx / wavelen - s.LowFreqFactor) / (s.HighFreqFactor - s.LowFreqFactor);
                invFreq[k] = (1.0 - smooth) * (invFreq[k] / s.Factor) + smooth * invFreq[k];
            }
            // else high frequency: keep original (extrapolate)
        }
    }

    /// <summary>YaRN NTK-by-parts blend (HF <c>_compute_yarn_parameters</c>, truncate=false), in place; returns mscale.</summary>
    private static double ApplyYarn(double[] invFreq, int dim, double theta, RopeScaling s)
    {
        double factor = s.Factor;
        double origMaxPos = s.OriginalContextLength;
        double logBase = Math.Log(theta);

        double FindCorrectionDim(double numRotations) =>
            dim * Math.Log(origMaxPos / (numRotations * 2.0 * Math.PI)) / (2.0 * logBase);

        double low = Math.Max(FindCorrectionDim(s.BetaFast), 0.0);
        double high = Math.Min(FindCorrectionDim(s.BetaSlow), dim - 1);
        double rampDenom = high == low ? 0.001 : high - low;

        for (int k = 0; k < invFreq.Length; k++)
        {
            double ramp = (k - low) / rampDenom;
            if (ramp < 0.0) ramp = 0.0;
            else if (ramp > 1.0) ramp = 1.0;
            double extrapolationFactor = 1.0 - ramp;               // 1 → keep original, 0 → fully interpolated
            double extrapolation = invFreq[k];
            double interpolation = invFreq[k] / factor;
            invFreq[k] = interpolation * (1.0 - extrapolationFactor) + extrapolation * extrapolationFactor;
        }
        return InferMscale(s);
    }

    /// <summary>Attention mscale: explicit override if set, else YaRN's <c>0.1·ln(factor)+1</c> (for factor &gt; 1), else 1.</summary>
    private static double InferMscale(RopeScaling s)
    {
        if (!double.IsNaN(s.AttentionFactor)) return s.AttentionFactor;
        if (s.Type == RopeScalingType.Yarn && s.Factor > 1.0) return 0.1 * Math.Log(s.Factor) + 1.0;
        return 1.0;
    }

    /// <summary>Cosmos/Anima per-axis NTK theta factor: <c>scale^(dim/(dim-2))</c> (1.0 when dim ≤ 2 or scale ≈ 1).</summary>
    public static double NtkThetaFactor(double scale, int dim)
    {
        if (dim <= 2 || Math.Abs(scale - 1.0) < 1e-12) return 1.0;
        return Math.Pow(scale, (double)dim / (dim - 2));
    }
}
