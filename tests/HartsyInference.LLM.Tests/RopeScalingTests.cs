using HartsyInference.Core.Rope;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Parity for the shared <see cref="RopeFrequencyBuilder"/> against independent reference math for each
/// scaling variant. The Yarn case reproduces the formula ported from the diffusion GPT-OSS encoder (the
/// regression guard that the extraction stayed faithful).</summary>
public sealed class RopeScalingTests
{
    private const int Dim = 64;
    private const double Theta = 500_000.0;

    private static double[] BaseInvFreq(int dim, double theta)
    {
        double[] f = new double[dim / 2];
        for (int k = 0; k < f.Length; k++) f[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / dim);
        return f;
    }

    private static void AssertClose(double[] a, double[] b, double tol, string label)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= tol * (1 + Math.Abs(b[i])), $"{label}[{i}]: {a[i]} vs {b[i]}");
    }

    [Fact]
    public void None_IsBaseInvFreq()
    {
        (double[] inv, double m) = RopeFrequencyBuilder.Build(Dim, Theta, RopeScaling.None, 128);
        AssertClose(inv, BaseInvFreq(Dim, Theta), 1e-12, "none");
        Assert.Equal(1.0, m);
    }

    [Fact]
    public void Linear_DividesAllFrequencies()
    {
        const double factor = 4.0;
        (double[] inv, double m) = RopeFrequencyBuilder.Build(Dim, Theta, new RopeScaling { Type = RopeScalingType.Linear, Factor = factor }, 128);
        double[] expect = BaseInvFreq(Dim, Theta);
        for (int i = 0; i < expect.Length; i++) expect[i] /= factor;
        AssertClose(inv, expect, 1e-12, "linear");
        Assert.Equal(1.0, m);
    }

    [Fact]
    public void Llama3_MatchesPiecewiseReference()
    {
        const double factor = 8.0, low = 1.0, high = 4.0, origCtx = 8192.0;
        RopeScaling s = new() { Type = RopeScalingType.Llama3, Factor = factor, LowFreqFactor = low, HighFreqFactor = high, OriginalContextLength = origCtx };
        (double[] inv, double m) = RopeFrequencyBuilder.Build(Dim, Theta, s, 16384);

        // Independent reference (HF _compute_llama3_parameters).
        double[] expect = BaseInvFreq(Dim, Theta);
        double lowWave = origCtx / low, highWave = origCtx / high;
        for (int k = 0; k < expect.Length; k++)
        {
            double wave = 2.0 * Math.PI / expect[k];
            if (wave > lowWave) expect[k] /= factor;
            else if (wave >= highWave)
            {
                double sm = (origCtx / wave - low) / (high - low);
                expect[k] = (1.0 - sm) * (expect[k] / factor) + sm * expect[k];
            }
        }
        AssertClose(inv, expect, 1e-12, "llama3");
        Assert.Equal(1.0, m);
    }

    [Fact]
    public void Yarn_MatchesPortedFormula_AndMscale()
    {
        const double factor = 32.0, betaFast = 32.0, betaSlow = 1.0, origMax = 4096.0;
        RopeScaling s = new() { Type = RopeScalingType.Yarn, Factor = factor, BetaFast = betaFast, BetaSlow = betaSlow, OriginalContextLength = origMax };
        (double[] inv, double m) = RopeFrequencyBuilder.Build(Dim, Theta, s, 131072);

        // Reference: the GPT-OSS encoder YaRN math (find_correction_dim + ramp blend), recomputed here.
        double[] expect = BaseInvFreq(Dim, Theta);
        double logBase = Math.Log(Theta);
        double Corr(double rot) => Dim * Math.Log(origMax / (rot * 2.0 * Math.PI)) / (2.0 * logBase);
        double lo = Math.Max(Corr(betaFast), 0.0), hi = Math.Min(Corr(betaSlow), Dim - 1);
        double denom = hi == lo ? 0.001 : hi - lo;
        for (int k = 0; k < expect.Length; k++)
        {
            double ramp = Math.Clamp((k - lo) / denom, 0.0, 1.0);
            double extra = 1.0 - ramp;
            expect[k] = (expect[k] / factor) * (1.0 - extra) + expect[k] * extra;
        }
        AssertClose(inv, expect, 1e-12, "yarn");
        Assert.Equal(0.1 * Math.Log(factor) + 1.0, m, 12);
    }

    [Fact]
    public void InvFreqFactors_DivideBase_MatchingGgmlFreqFactors()
    {
        // GGUF rope_freqs.weight stores ggml "freq_factors": divisors (~1 on high frequencies rising to the
        // rope-scaling factor, e.g. 32 on low ones for Llama-3.2). ggml applies theta = theta_base /
        // freq_factor, so the builder must DIVIDE the base inv_freq by them (not multiply).
        float[] factors = new float[Dim / 2];
        for (int i = 0; i < factors.Length; i++) factors[i] = i < 16 ? 1f : Math.Min(32f, 1f + (i - 15) * 4f);
        (double[] inv, _) = RopeFrequencyBuilder.Build(Dim, Theta, new RopeScaling { Type = RopeScalingType.Llama3, InvFreqFactors = factors }, 128);
        double[] expect = BaseInvFreq(Dim, Theta);
        for (int i = 0; i < expect.Length; i++) expect[i] /= factors[i];
        AssertClose(inv, expect, 1e-6, "factors");
    }

    [Fact]
    public void DynamicNtk_RescalesOnlyAboveOriginalContext()
    {
        RopeScaling s = new() { Type = RopeScalingType.DynamicNtk, Factor = 4.0, OriginalContextLength = 4096 };
        (double[] below, _) = RopeFrequencyBuilder.Build(Dim, Theta, s, 1024);
        AssertClose(below, BaseInvFreq(Dim, Theta), 1e-12, "dynamic-below");

        (double[] above, _) = RopeFrequencyBuilder.Build(Dim, Theta, s, 16384);
        Assert.True(Math.Abs(above[1] - BaseInvFreq(Dim, Theta)[1]) > 1e-9, "dynamic above original context should rescale");
    }
}
