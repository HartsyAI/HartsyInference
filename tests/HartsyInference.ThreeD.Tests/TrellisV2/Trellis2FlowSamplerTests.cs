using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2FlowSamplerTests
{
    private static Trellis2SamplerParams Params(
        int steps, float strength, float rescale = 0f, double lo = 0.0, double hi = 1.0, double rescaleT = 1.0) => new()
    {
        Steps = steps,
        GuidanceStrength = strength,
        GuidanceRescale = rescale,
        IntervalLow = lo,
        IntervalHigh = hi,
        RescaleT = rescaleT,
    };

    [Fact]
    public void Schedule_WarpsALinearGridByRescaleT()
    {
        // t' = 3t / (1 + 2t) over t = 1, 0.75, 0.5, 0.25, 0.
        double[] schedule = Trellis2FlowSampler.Schedule(4, 3.0);

        Assert.Equal(new[] { 1.0, 0.9, 0.75, 0.5, 0.0 }, schedule.Select(v => Math.Round(v, 12)));
    }

    [Fact]
    public void Schedule_ShippedStructureSettings_StartAtOneEndAtZeroAndDecrease()
    {
        double[] schedule = Trellis2FlowSampler.Schedule(12, 5.0);

        Assert.Equal(13, schedule.Length);
        Assert.Equal(1.0, schedule[0]);
        Assert.Equal(0.0, schedule[12]);
        // 5 · (11/12) / (1 + 4 · 11/12) = 0.98214...
        Assert.Equal(0.9821428571, schedule[1], 9);
        for (int i = 1; i < schedule.Length; i++)
            Assert.True(schedule[i] < schedule[i - 1]);
    }

    [Fact]
    public void Schedule_RescaleOfOneIsLinear()
    {
        double[] schedule = Trellis2FlowSampler.Schedule(5, 1.0);

        for (int i = 0; i <= 5; i++)
            Assert.Equal(1.0 - i / 5.0, schedule[i], 12);
    }

    [Fact]
    public void Sample_GivesTheModelOneThousandTimesTWithinTheGuidanceInterval()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 3), DType.F32);
        List<(float T, bool Conditional)> calls = [];

        using Tensor result = Trellis2FlowSampler.Sample(backend, noise, Params(2, 3f, lo: 0.0, hi: 1.0), (x, t, conditional) =>
        {
            calls.Add((t, conditional));
            return new Tensor(x.Shape, DType.F32);
        });

        // Two steps at t = 1 and 0.5, each guided, so each calls the model for the conditional and the unconditional branch.
        Assert.Equal([(1000f, true), (1000f, false), (500f, true), (500f, false)], calls);
    }

    [Fact]
    public void Sample_OutsideTheIntervalRunsTheConditionalBranchAlone()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 3), DType.F32);
        List<(float T, bool Conditional)> calls = [];

        using Tensor result = Trellis2FlowSampler.Sample(backend, noise, Params(2, 3f, lo: 0.6, hi: 1.0), (x, t, conditional) =>
        {
            calls.Add((t, conditional));
            return new Tensor(x.Shape, DType.F32);
        });

        // t = 1 is inside [0.6, 1]; t = 0.5 is not.
        Assert.Equal([(1000f, true), (1000f, false), (500f, true)], calls);
    }

    [Fact]
    public void Sample_AppliesClassifierFreeGuidanceAsStrengthTimesPosPlusOneMinusStrengthTimesNeg()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 2), DType.F32);
        noise.AsSpan<float>()[0] = 2f;
        noise.AsSpan<float>()[1] = -1f;
        const float pos = 0.5f, neg = 0.25f, strength = 7.5f;

        using Tensor result = Trellis2FlowSampler.Sample(backend, noise, Params(1, strength), (x, t, conditional) =>
        {
            Tensor v = new(x.Shape, DType.F32);
            v.AsSpan<float>().Fill(conditional ? pos : neg);
            return v;
        });

        // One step from t = 1 to 0: x + (s·pos + (1 − s)·neg) · (0 − 1).
        float velocity = strength * pos + (1 - strength) * neg;
        ReadOnlySpan<float> values = result.AsReadOnlySpan<float>();
        Assert.Equal(2f - velocity, values[0], 5);
        Assert.Equal(-1f - velocity, values[1], 5);
        Assert.Equal(2f, noise.AsReadOnlySpan<float>()[0]);
    }

    [Fact]
    public void Sample_StrengthOneIsTheConditionalPredictionAlone()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 2), DType.F32);
        int calls = 0;

        using Tensor result = Trellis2FlowSampler.Sample(backend, noise, Params(1, 1f), (x, t, conditional) =>
        {
            calls++;
            Assert.True(conditional);
            Tensor v = new(x.Shape, DType.F32);
            v.AsSpan<float>().Fill(0.4f);
            return v;
        });

        Assert.Equal(1, calls);
        Assert.Equal(-0.4f, result.AsReadOnlySpan<float>()[0], 6);
    }

    [Fact]
    public void RescaleGuidance_WithFullWeightMatchesTheConditionalX0Spread()
    {
        double sigmaMin = 1e-5, t = 0.8;
        float[] x = [1f, 0f, -1f, 2f, 0.5f, -0.25f];
        float[] pos = [0.3f, -0.2f, 0.1f, 0.4f, -0.5f, 0.2f];
        float[] neg = [0.1f, 0.2f, -0.3f, 0.0f, 0.3f, -0.1f];
        float[] guided = (float[])pos.Clone();

        Trellis2FlowSampler.RescaleGuidance(x, guided, neg, 6f, 1f, t, sigmaMin);

        // After the blend with weight 1, the x0 recovered from the new prediction has the conditional x0's std.
        double StdOfX0(float[] prediction) => Std(x.Select((v, i) => (1 - sigmaMin) * v - (sigmaMin + (1 - sigmaMin) * t) * prediction[i]).ToArray());
        Assert.Equal(StdOfX0(pos), StdOfX0(guided), 5);
    }

    [Fact]
    public void RescaleGuidance_WithZeroWeightKeepsThePlainGuidedPrediction()
    {
        float[] x = [1f, 0f, -1f, 2f];
        float[] pos = [0.3f, -0.2f, 0.1f, 0.4f];
        float[] neg = [0.1f, 0.2f, -0.3f, 0.0f];
        float[] guided = (float[])pos.Clone();

        Trellis2FlowSampler.RescaleGuidance(x, guided, neg, 4f, 0f, 0.5, 1e-5);

        for (int i = 0; i < x.Length; i++)
            Assert.Equal(4f * pos[i] + (1 - 4f) * neg[i], guided[i], 5);
    }

    [Fact]
    public void RescaleGuidance_MatchesAHandComputedPartialBlend()
    {
        // x = [1, -1], pos = [0, 0], neg = [1, -1], s = 2, t = 0.5, sigma_min = 0 (limit), rescale = 0.5.
        // a = 1, b = t = 0.5. x0_pos = x = [1, -1], std 1. pred = 2·pos − neg = [-1, 1]; x0_cfg = x − 0.5·pred = [1.5, -1.5], std 1.5.
        // ratio = 2/3; x0 = x0_cfg · (0.5 · 2/3 + 0.5) = x0_cfg · 5/6 = [1.25, -1.25]; pred' = (x − x0) / b = [-0.5, 0.5].
        float[] x = [1f, -1f];
        float[] pos = [0f, 0f];
        float[] neg = [1f, -1f];

        Trellis2FlowSampler.RescaleGuidance(x, pos, neg, 2f, 0.5f, 0.5, 1e-12);

        Assert.Equal(-0.5f, pos[0], 5);
        Assert.Equal(0.5f, pos[1], 5);
    }

    [Fact]
    public void Sample_WithRescaleMovesTheLatentByTheRescaledPrediction()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 2), DType.F32);
        noise.AsSpan<float>()[0] = 1f;
        noise.AsSpan<float>()[1] = -1f;

        using Tensor result = Trellis2FlowSampler.Sample(
            backend,
            noise,
            new Trellis2SamplerParams
            {
                Steps = 1, GuidanceStrength = 2f, GuidanceRescale = 0.5f, IntervalLow = 0.0, IntervalHigh = 1.0, RescaleT = 1.0, SigmaMin = 1e-12,
            },
            (x, t, conditional) =>
            {
                Tensor v = new(x.Shape, DType.F32);
                if (!conditional)
                {
                    v.AsSpan<float>()[0] = 1f;
                    v.AsSpan<float>()[1] = -1f;
                }

                return v;
            });

        // t = 1, sigma_min ~ 0: x0_pos = x = [1, -1] (std 1), pred = 2·pos − neg = [-1, 1], x0_cfg = x − pred = [2, -2] (std 2), ratio 1/2,
        // x0 = x0_cfg · (0.5 · 0.5 + 0.5) = [1.5, -1.5], pred' = x − x0 = [-0.5, 0.5], and the step to t = 0 gives x − pred' = [1.5, -1.5].
        ReadOnlySpan<float> latent = result.AsReadOnlySpan<float>();
        Assert.Equal(1.5f, latent[0], 4);
        Assert.Equal(-1.5f, latent[1], 4);
        Assert.Equal(-latent[0], latent[1], 5);
    }

    [Fact]
    public void Sample_ReleasesTheLatentWhenTheStepCallbackCancels()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 2), DType.F32);
        Tensor? seen = null;

        Assert.Throws<OperationCanceledException>(() => Trellis2FlowSampler.Sample(
            backend,
            noise,
            Params(3, 1f),
            (x, t, conditional) =>
            {
                seen = x;
                return new Tensor(x.Shape, DType.F32);
            },
            step => throw new OperationCanceledException()));

        Assert.NotNull(seen);
        Assert.NotSame(noise, seen);
        Trellis2TestData.AssertDisposed(seen!);
    }

    [Fact]
    public void Sample_RejectsAModelThatReturnsTheWrongSize()
    {
        using CpuBackend backend = new();
        using Tensor noise = new(new TensorShape(1, 4), DType.F32);

        Assert.Throws<InvalidOperationException>(() => Trellis2FlowSampler.Sample(
            backend, noise, Params(1, 1f), (x, t, conditional) => new Tensor(new TensorShape(1, 3), DType.F32)));
    }

    [Theory]
    [InlineData(0, 1f, 1.0)]
    [InlineData(4, 1f, 0.0)]
    [InlineData(4, float.NaN, 1.0)]
    public void Parameters_AreValidated(int steps, float strength, double rescaleT)
    {
        Trellis2SamplerParams bad = Params(steps, strength, rescaleT: rescaleT);

        Assert.Throws<ArgumentException>(bad.Validate);
    }

    private static double Std(double[] values)
    {
        double mean = values.Average();
        return Math.Sqrt(values.Select(v => (v - mean) * (v - mean)).Average());
    }
}
