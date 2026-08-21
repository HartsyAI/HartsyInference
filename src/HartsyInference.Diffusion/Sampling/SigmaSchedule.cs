using HartsyInference.Core.Logging;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>The sigma-schedule half of sampling: given the sigma grid a model family already produces for N steps,
/// re-spaces it. This is ComfyUI's scheduler dropdown (<c>normal</c> / <c>karras</c> / <c>exponential</c> / …), and it
/// is deliberately a <em>transform over the family's own sigmas</em> rather than a replacement for them.
///
/// <para><b>Why a transform.</b> Each family earns its sigma range from its own config — Flux's resolution-dependent
/// dynamic shift, Qwen-Image's <c>shift_terminal</c>, LTX-2's terminal stretch, SDXL's alphas-cumprod table. Rebuilding
/// a schedule from scratch per name would silently discard all of that and mis-sample every family with a non-default
/// range. Taking <c>sigmaMax = sigmas[0]</c> and <c>sigmaMin</c> from the last non-zero entry keeps the family's own
/// endpoints and changes only the spacing between them, which is exactly what the ComfyUI names mean.</para>
///
/// <para>Every method returns <c>steps + 1</c> values ending at 0.0 — the trailing zero is the terminal sigma the Euler
/// update needs for its final <c>dt</c>, matching the convention <see cref="Schedulers.FlowMatchEulerDiscreteScheduler"/>
/// and <see cref="Schedulers.EulerDiscreteScheduler"/> already use.</para></summary>
public static class SigmaSchedule
{
    /// <summary>Applies the named schedule to <paramref name="baseSigmas"/>. An unrecognized name throws rather than
    /// falling back: a silently substituted schedule produces a different image with no signal, which is the failure
    /// this whole surface exists to remove.</summary>
    /// <param name="name">Schedule name; null/empty means <c>normal</c> (the family's own spacing, unchanged).</param>
    /// <param name="baseSigmas">The family's own sigma array, length <c>steps + 1</c>, descending, ending at 0.</param>
    public static float[] Apply(string? name, float[] baseSigmas)
    {
        ArgumentNullException.ThrowIfNull(baseSigmas);
        if (baseSigmas.Length < 2)
        {
            throw new ArgumentException(
                $"A sigma schedule needs at least one step (2 entries incl. the terminal zero); got {baseSigmas.Length}.",
                nameof(baseSigmas));
        }
        string key = (name ?? "").Trim().ToLowerInvariant();
        if (key.Length == 0 || key is "normal" or "default")
        {
            return baseSigmas;
        }

        int steps = baseSigmas.Length - 1;
        (float sigmaMin, float sigmaMax) = Bounds(baseSigmas);
        float[] result = key switch
        {
            "karras" => Karras(sigmaMin, sigmaMax, steps),
            "exponential" => Exponential(sigmaMin, sigmaMax, steps),
            "sgm_uniform" => SgmUniform(baseSigmas, steps),
            "simple" => Simple(baseSigmas, steps),
            "ddim_uniform" => DdimUniform(baseSigmas, steps),
            "beta" => Beta(baseSigmas, steps),
            "kl_optimal" => KlOptimal(sigmaMin, sigmaMax, steps),
            "linear_quadratic" => LinearQuadratic(sigmaMax, steps),
            _ => throw new NotSupportedException(
                $"Unknown sigma schedule '{name}'. Available: {string.Join(", ", Names)}."),
        };
        Logs.Verbose($"[Sampling] Schedule '{key}': sigma {result[0]:F4} -> {result[^2]:F4} over {steps} steps "
            + $"(family range {sigmaMax:F4} -> {sigmaMin:F4}).");
        return result;
    }

    /// <summary>Every schedule name <see cref="Apply"/> accepts, for error messages and capability reporting.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "kl_optimal",
        "linear_quadratic",
    ];

    /// <summary>Whether <paramref name="name"/> is a schedule this class can apply.</summary>
    public static bool IsKnown(string? name)
    {
        string key = (name ?? "").Trim().ToLowerInvariant();
        return key.Length == 0 || key == "default" || Names.Contains(key, StringComparer.Ordinal);
    }

    /// <summary>The family's own endpoints: <c>sigmaMax</c> is the first entry, <c>sigmaMin</c> the last non-zero one.
    /// Reading sigmaMin off the terminal zero instead would collapse every log-space schedule to −∞.</summary>
    private static (float Min, float Max) Bounds(float[] sigmas)
    {
        float max = sigmas[0];
        float min = 0f;
        for (int i = sigmas.Length - 1; i >= 0; i--)
        {
            if (sigmas[i] > 0f)
            {
                min = sigmas[i];
                break;
            }
        }
        if (min <= 0f || max <= min)
        {
            throw new ArgumentException(
                $"Sigma array is not a usable descending schedule (max={max}, min={min}); cannot re-space it.");
        }
        return (min, max);
    }

    /// <summary>Karras et al. 2022 rho-spacing: <c>(σmax^(1/ρ) + i/(n−1)·(σmin^(1/ρ) − σmax^(1/ρ)))^ρ</c>.</summary>
    public static float[] Karras(float sigmaMin, float sigmaMax, int steps, float rho = 7.0f)
    {
        float[] sigmas = new float[steps + 1];
        double invRho = 1.0 / rho;
        double minInv = Math.Pow(sigmaMin, invRho);
        double maxInv = Math.Pow(sigmaMax, invRho);
        for (int i = 0; i < steps; i++)
        {
            double t = steps == 1 ? 0.0 : (double)i / (steps - 1);
            sigmas[i] = (float)Math.Pow(maxInv + (t * (minInv - maxInv)), rho);
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary>Geometric (log-linear) spacing between the endpoints.</summary>
    public static float[] Exponential(float sigmaMin, float sigmaMax, int steps)
    {
        float[] sigmas = new float[steps + 1];
        double logMin = Math.Log(sigmaMin);
        double logMax = Math.Log(sigmaMax);
        for (int i = 0; i < steps; i++)
        {
            double t = steps == 1 ? 0.0 : (double)i / (steps - 1);
            sigmas[i] = (float)Math.Exp(logMax + (t * (logMin - logMax)));
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary>ComfyUI <c>sgm_uniform</c>: <c>linspace(sigma_max, sigma_min, steps + 1)[:-1]</c> — it lays out
    /// <c>steps + 1</c> points across the family's range and then DROPS the last, so the schedule stops one point short
    /// of sigma_min before the terminal zero. That deliberate shortfall is the whole difference from <c>simple</c>, and
    /// it is what SGM-trained checkpoints expect.</summary>
    private static float[] SgmUniform(float[] baseSigmas, int steps) => Resample(baseSigmas, steps, denominator: steps);

    /// <summary>ComfyUI <c>simple</c>: evenly spaced across the family's own curve, reaching sigma_min on the last
    /// non-terminal step.</summary>
    private static float[] Simple(float[] baseSigmas, int steps) => Resample(baseSigmas, steps, denominator: steps - 1);

    /// <summary>ComfyUI <c>ddim_uniform</c>: DDIM's evenly-spaced timestep subset, which over a monotone sigma curve is
    /// the same index resampling as <c>simple</c>.</summary>
    private static float[] DdimUniform(float[] baseSigmas, int steps) => Resample(baseSigmas, steps, denominator: steps - 1);

    /// <summary>Reads the family's curve at <paramref name="steps"/> evenly-spaced positions, linearly interpolating
    /// between neighbours so a request for more steps than the source array holds stays smooth rather than aliasing.
    ///
    /// <para>Only the NON-terminal entries are sampled (<c>baseSigmas.Length − 1</c> of them). Including the terminal
    /// zero in the interpolation range is what a first cut of this did, and it silently placed a 0 at the last
    /// non-terminal step — an infinite Euler <c>dt</c> ratio and a dead final step, caught by the descending-and-
    /// positive invariant in <c>SigmaScheduleTests</c>.</para>
    ///
    /// <para><paramref name="denominator"/> selects how far across the range the last sampled point lands:
    /// <c>steps − 1</c> reaches the end (simple/ddim_uniform), <c>steps</c> stops one short (sgm_uniform).</para></summary>
    private static float[] Resample(float[] baseSigmas, int steps, int denominator)
    {
        int usable = baseSigmas.Length - 1;
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            double pos = denominator <= 0 ? 0.0 : (double)i / denominator * (usable - 1);
            int lo = (int)Math.Floor(pos);
            int hi = Math.Min(lo + 1, usable - 1);
            double frac = pos - lo;
            sigmas[i] = (float)((baseSigmas[lo] * (1.0 - frac)) + (baseSigmas[hi] * frac));
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary>ComfyUI <c>beta</c>: samples the family's curve at the Beta(α=0.6, β=0.6) quantiles, which clusters
    /// steps at BOTH ends of the schedule instead of one. Uses a bisection inverse-CDF because the incomplete beta
    /// function has no closed form and this runs once per generation, not per step.</summary>
    private static float[] Beta(float[] baseSigmas, int steps, double alpha = 0.6, double beta = 0.6)
    {
        int usable = baseSigmas.Length - 1;
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            double p = steps == 1 ? 0.0 : (double)i / (steps - 1);
            double q = BetaQuantile(p, alpha, beta);
            double pos = q * (usable - 1);
            int lo = (int)Math.Floor(pos);
            int hi = Math.Min(lo + 1, usable - 1);
            double frac = pos - lo;
            sigmas[i] = (float)((baseSigmas[lo] * (1.0 - frac)) + (baseSigmas[hi] * frac));
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary>Inverse Beta CDF by bisection — 60 iterations puts the result well inside F32 resolution.</summary>
    private static double BetaQuantile(double p, double alpha, double beta)
    {
        if (p <= 0.0) return 0.0;
        if (p >= 1.0) return 1.0;
        double lo = 0.0, hi = 1.0;
        for (int i = 0; i < 60; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (RegularizedIncompleteBeta(mid, alpha, beta) < p)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

    /// <summary>Regularized incomplete beta I_x(a,b) via the standard continued fraction, with the symmetry reflection
    /// that keeps the fraction in its fast-converging half.</summary>
    private static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double lnBeta = LogGamma(a) + LogGamma(b) - LogGamma(a + b);
        double front = Math.Exp((Math.Log(x) * a) + (Math.Log(1.0 - x) * b) - lnBeta);
        return x < (a + 1.0) / (a + b + 2.0)
            ? front * BetaContinuedFraction(x, a, b) / a
            : 1.0 - (Math.Exp((Math.Log(1.0 - x) * b) + (Math.Log(x) * a) - lnBeta) * BetaContinuedFraction(1.0 - x, b, a) / b);
    }

    /// <summary>Lentz's algorithm for the beta continued fraction.</summary>
    private static double BetaContinuedFraction(double x, double a, double b)
    {
        const double Tiny = 1e-30;
        double f = 1.0, c = 1.0, d = 0.0;
        for (int i = 0; i <= 200; i++)
        {
            int m = i / 2;
            double numerator = i == 0
                ? 1.0
                : (i % 2 == 0
                    ? m * (b - m) * x / ((a + (2.0 * m) - 1.0) * (a + (2.0 * m)))
                    : -((a + m) * (a + b + m) * x) / ((a + (2.0 * m)) * (a + (2.0 * m) + 1.0)));
            d = 1.0 + (numerator * d);
            if (Math.Abs(d) < Tiny) d = Tiny;
            d = 1.0 / d;
            c = 1.0 + (numerator / c);
            if (Math.Abs(c) < Tiny) c = Tiny;
            double delta = c * d;
            f *= delta;
            if (Math.Abs(1.0 - delta) < 1e-12)
            {
                break;
            }
        }
        return f;
    }

    /// <summary>Lanczos log-gamma, enough precision for a quantile solved to 1e-12.</summary>
    private static double LogGamma(double x)
    {
        double[] g = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (x < 0.5)
        {
            return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        }
        x -= 1.0;
        double a = 0.99999999999980993;
        double t = x + 7.5;
        for (int i = 0; i < g.Length; i++)
        {
            a += g[i] / (x + i + 1.0);
        }
        return (0.5 * Math.Log(2 * Math.PI)) + ((x + 0.5) * Math.Log(t)) - t + Math.Log(a);
    }

    /// <summary>ComfyUI <c>kl_optimal</c> (Align Your Steps): arctan-spaced between the endpoints, which minimizes the
    /// KL divergence of the discretized reverse process rather than spacing sigma itself.</summary>
    public static float[] KlOptimal(float sigmaMin, float sigmaMax, int steps)
    {
        float[] sigmas = new float[steps + 1];
        double aMin = Math.Atan(sigmaMin);
        double aMax = Math.Atan(sigmaMax);
        for (int i = 0; i < steps; i++)
        {
            double t = steps == 1 ? 0.0 : (double)i / (steps - 1);
            sigmas[i] = (float)Math.Tan(aMax + (t * (aMin - aMax)));
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    /// <summary>ComfyUI <c>linear_quadratic</c>: linear over the first half of the schedule, quadratic over the second.
    /// Built for the distilled few-step video models, where the tail needs finer resolution than the head.</summary>
    public static float[] LinearQuadratic(float sigmaMax, int steps, float threshold = 0.025f, float linearSteps = 0.5f)
    {
        float[] sigmas = new float[steps + 1];
        int linear = Math.Max(1, (int)(steps * linearSteps));
        double linearSlope = threshold / linear;
        double quadStart = linear * linearSlope;
        double quadRange = 1.0 - quadStart;
        int quadSteps = Math.Max(1, steps - linear);
        for (int i = 0; i < steps; i++)
        {
            double frac;
            if (i < linear)
            {
                frac = i * linearSlope;
            }
            else
            {
                double q = (double)(i - linear) / quadSteps;
                frac = quadStart + (quadRange * q * q);
            }
            sigmas[i] = (float)(sigmaMax * (1.0 - frac));
        }
        sigmas[steps] = 0f;
        return sigmas;
    }
}
