using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Reference guidance: which branches a step evaluates, the combine they produce, and the corrections. The flow shift is 3, as
/// Anima's. The fixture checks against the reference adapter's own outputs run when a dump is provided.</summary>
public sealed class ReferenceGuidanceTests
{
    private const double Shift = 3.0;
    private static readonly TensorShape Shape = new(1, 2, 3, 4);

    [Theory]
    // cfg > 1, a reference scale other than 1, inside the window: the base is the negative without the reference.
    [InlineData(4.0f, 2.0f, true, ReferenceOutside.KeepReference, new[] { ReferenceBranch.PositiveWithReference, ReferenceBranch.NegativeWithReference, ReferenceBranch.NegativeWithoutReference })]
    // cfg <= 1: no negative, so the base is the positive without the reference.
    [InlineData(1.0f, 2.0f, true, ReferenceOutside.KeepReference, new[] { ReferenceBranch.PositiveWithReference, ReferenceBranch.PositiveWithoutReference })]
    // reference scale 1 costs no base call.
    [InlineData(4.0f, 1.0f, true, ReferenceOutside.KeepReference, new[] { ReferenceBranch.PositiveWithReference, ReferenceBranch.NegativeWithReference })]
    [InlineData(1.0f, 1.0f, true, ReferenceOutside.KeepReference, new[] { ReferenceBranch.PositiveWithReference })]
    // outside the window, the reference stays in both branches with no extra guidance.
    [InlineData(4.0f, 2.0f, false, ReferenceOutside.KeepReference, new[] { ReferenceBranch.PositiveWithReference, ReferenceBranch.NegativeWithReference })]
    [InlineData(1.0f, 2.0f, false, ReferenceOutside.KeepReference, new[] { ReferenceBranch.PositiveWithReference })]
    // outside the window with DropReference, plain CFG on the base model.
    [InlineData(4.0f, 2.0f, false, ReferenceOutside.DropReference, new[] { ReferenceBranch.PositiveWithoutReference, ReferenceBranch.NegativeWithoutReference })]
    [InlineData(1.0f, 2.0f, false, ReferenceOutside.DropReference, new[] { ReferenceBranch.PositiveWithoutReference })]
    public void Predict_EvaluatesTheBranchesThatTheCombineNeeds(float cfg, float scale, bool inside, ReferenceOutside outside,
        ReferenceBranch[] expected)
    {
        using CpuBackend cpu = new();
        ReferenceGuidanceSettings settings = WindowSettings(scale, outside);
        float sigma = SigmaFor(settings, inside);
        using Tensor x = Filled(0.5f);
        List<ReferenceBranch> evaluated = [];

        using DenoisePrediction prediction = ReferenceGuidance.Predict(cpu, settings, x, sigma, Shift, cfg, branch =>
        {
            evaluated.Add(branch);
            return Filled(1f);
        });

        Assert.Equal(expected, evaluated);
    }

    [Fact]
    public void Predict_CombinesAsBasePlusScaledReferenceAndCfg()
    {
        // The node's formula: base + scale·(negative with reference − base) + cfg·(positive with reference − negative with reference).
        // With constant branches the result is a constant that the formula gives directly: 1 + 2·(4 − 1) + 3·(10 − 4) = 25.
        using CpuBackend cpu = new();
        ReferenceGuidanceSettings settings = new(2.0f, ReferenceCorrection.None);
        using Tensor x = Filled(0f);

        using DenoisePrediction prediction = ReferenceGuidance.Predict(cpu, settings, x, 0.5f, Shift, 3.0f, branch => branch switch
        {
            ReferenceBranch.PositiveWithReference => Filled(10f),
            ReferenceBranch.NegativeWithReference => Filled(4f),
            ReferenceBranch.NegativeWithoutReference => Filled(1f),
            _ => throw new InvalidOperationException($"unexpected branch {branch}"),
        });

        AssertConstant(prediction.Cond, 25f);
        Assert.Same(prediction.Cond, prediction.Uncond);
        Assert.Equal(1.0f, prediction.Guidance);
    }

    [Fact]
    public void Predict_WithCfgAtOne_CombinesPositiveWithReferenceOverItsBase()
    {
        // No negative branch: the base is the positive without the reference, and the mid branch is the positive with it.
        // base + 2·(positive − base) = 1 + 2·(10 − 1) = 19.
        using CpuBackend cpu = new();
        ReferenceGuidanceSettings settings = new(2.0f, ReferenceCorrection.None);
        using Tensor x = Filled(0f);

        using DenoisePrediction prediction = ReferenceGuidance.Predict(cpu, settings, x, 0.5f, Shift, 1.0f, branch => branch switch
        {
            ReferenceBranch.PositiveWithReference => Filled(10f),
            ReferenceBranch.PositiveWithoutReference => Filled(1f),
            _ => throw new InvalidOperationException($"unexpected branch {branch}"),
        });

        AssertConstant(prediction.Cond, 19f);
    }

    [Fact]
    public void Apg_RemovesTheComponentAlongThePositivePrediction()
    {
        // The corrected x0 is guided − α·pos, so its inner product with the positive prediction is ‖pos‖² (the update's parallel part is gone).
        using CpuBackend cpu = new();
        Tensor guided = Noise(7, 2.0f);
        Tensor positive = Noise(11, 1.0f);
        using Tensor corrected = new(Shape, DType.F32);

        ReferenceGuidance.Correct(cpu, ReferenceCorrection.Apg, 0.5f, guided, positive, corrected);

        double dot = Dot(Span(corrected), Span(positive));
        double positiveNormSquared = Dot(Span(positive), Span(positive));
        Assert.Equal(positiveNormSquared, dot, precision: 2);
        guided.Dispose();
        positive.Dispose();
    }

    [Fact]
    public void StdRescale_WithPhiOne_GivesThePositiveStandardDeviation()
    {
        // With phi 1 the output is (std(pos)/std(guided))·guided, so its standard deviation is std(pos).
        using CpuBackend cpu = new();
        Tensor guided = Noise(3, 2.0f);
        Tensor positive = Noise(5, 0.5f);
        using Tensor corrected = new(Shape, DType.F32);

        ReferenceGuidance.Correct(cpu, ReferenceCorrection.StdRescale, 1.0f, guided, positive, corrected);

        Assert.Equal(StandardDeviation(Span(positive)), StandardDeviation(Span(corrected)), precision: 4);
        guided.Dispose();
        positive.Dispose();
    }

    [Fact]
    public void Correct_NoneIsTheIdentity()
    {
        using CpuBackend cpu = new();
        Tensor guided = Noise(9, 1.0f);
        Tensor positive = Noise(13, 1.0f);
        using Tensor corrected = new(Shape, DType.F32);

        ReferenceGuidance.Correct(cpu, ReferenceCorrection.None, 0.5f, guided, positive, corrected);

        Assert.Equal(Span(guided).ToArray(), Span(corrected).ToArray());
        guided.Dispose();
        positive.Dispose();
    }

    [Fact]
    public void Settings_RefuseOutOfRangeValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferenceGuidanceSettings(-1f, ReferenceCorrection.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferenceGuidanceSettings(1f, ReferenceCorrection.StdRescale, rescalePhi: 1.5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferenceGuidanceSettings(1f, ReferenceCorrection.None, start: 0.8f, end: 0.2f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferenceGuidanceSettings(float.NaN, ReferenceCorrection.None));
    }

    [Fact]
    public void Window_IsEverywhereWhenUnbounded_AndTracksTheSchedulePercentages()
    {
        ReferenceGuidanceSettings everywhere = new(1f, ReferenceCorrection.None);
        Assert.True(everywhere.InWindow(0.01f, Shift));
        Assert.True(everywhere.InWindow(1f, Shift));

        ReferenceGuidanceSettings early = new(1f, ReferenceCorrection.None, end: 0.5f);
        double boundary = SamplerOptions.FlowPercentToSigma(0.5, Shift);
        Assert.True(early.InWindow((float)boundary, Shift));
        Assert.False(early.InWindow((float)(boundary / 2), Shift));
    }

    /// <summary>Settings whose window ends at 50% of the denoising, so the boundary is the sigma at 50%.</summary>
    private static ReferenceGuidanceSettings WindowSettings(float scale, ReferenceOutside outside) =>
        new(scale, ReferenceCorrection.None, end: 0.5f, outside: outside);

    /// <summary>A sigma inside the window (between its boundary and 1) or outside it (below the boundary).</summary>
    private static float SigmaFor(ReferenceGuidanceSettings settings, bool inside)
    {
        double boundary = SamplerOptions.FlowPercentToSigma(0.5, Shift);
        return (float)(inside ? (boundary + 1.0) / 2 : boundary / 2);
    }

    private static Tensor Filled(float value)
    {
        Tensor t = new(Shape, DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static Tensor Noise(int seed, float scale)
    {
        Random rng = new(seed);
        Tensor t = new(Shape, DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        }
        return t;
    }

    private static void AssertConstant(Tensor tensor, float expected)
    {
        foreach (float value in Span(tensor))
        {
            Assert.Equal(expected, value, precision: 3);
        }
    }

    private static ReadOnlySpan<float> Span(Tensor tensor) => tensor.AsReadOnlySpan<float>();

    private static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }
        return sum;
    }

    private static double StandardDeviation(ReadOnlySpan<float> values)
    {
        double mean = 0;
        foreach (float v in values)
        {
            mean += v;
        }
        mean /= values.Length;
        double squares = 0;
        foreach (float v in values)
        {
            squares += (v - mean) * (v - mean);
        }
        return Math.Sqrt(squares / (values.Length - 1));
    }
}
