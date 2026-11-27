using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamFlowSamplerTests
{
    private static Dictionary<string, Tensor> Latents(float shape, float pose)
    {
        Tensor a = new(new TensorShape(2, 2), DType.F32);
        Tensor b = new(new TensorShape(1, 3), DType.F32);
        a.AsSpan<float>().Fill(shape);
        b.AsSpan<float>().Fill(pose);
        return new Dictionary<string, Tensor> { ["shape"] = a, ["translation"] = b };
    }

    private static Dictionary<string, Tensor> Constant(IReadOnlyDictionary<string, Tensor> like, float value)
    {
        Dictionary<string, Tensor> result = [];
        foreach (KeyValuePair<string, Tensor> entry in like)
        {
            Tensor t = new(entry.Value.Shape, DType.F32);
            t.AsSpan<float>().Fill(value);
            result[entry.Key] = t;
        }

        return result;
    }

    private static void Dispose(Dictionary<string, Tensor> tensors)
    {
        foreach (Tensor t in tensors.Values)
            t.Dispose();
    }

    [Fact]
    public void Schedule_ShippedStructureSettings_AscendFromZeroToOneAndWarpTowardsTheNoiseEnd()
    {
        double[] t = SamFlowSampler.Schedule(25, 3.0);

        Assert.Equal(26, t.Length);
        Assert.Equal(0.0, t[0]);
        Assert.Equal(1.0, t[25]);
        // u = 1/25: 0.04 / (1 + 2 · 0.96)
        Assert.Equal(0.04 / 2.92, t[1], 12);
        for (int i = 1; i < t.Length; i++)
            Assert.True(t[i] > t[i - 1]);
    }

    [Fact]
    public void Schedule_RescaleOfOneIsLinear()
    {
        double[] t = SamFlowSampler.Schedule(5, 1.0);

        for (int i = 0; i <= 5; i++)
            Assert.Equal(i / 5.0, t[i], 12);
    }

    [Fact]
    public void IsGuided_StructureDefaults_CoverTheFirst19Of25Steps()
    {
        double[] t = SamFlowSampler.Schedule(25, 3.0);

        int guided = Enumerable.Range(0, 25).Count(k => SamFlowSampler.IsGuided(t[k], SamSamplerParams.StructureDefault));

        Assert.Equal(19, guided);
        Assert.True(SamFlowSampler.IsGuided(t[18], SamSamplerParams.StructureDefault));
        Assert.False(SamFlowSampler.IsGuided(t[19], SamSamplerParams.StructureDefault));
    }

    [Fact]
    public void IsGuided_SlatDefaults_CoverTheFirst13Of25Steps()
    {
        double[] t = SamFlowSampler.Schedule(25, 1.0);

        int guided = Enumerable.Range(0, 25).Count(k => SamFlowSampler.IsGuided(t[k], SamSamplerParams.SlatDefault));

        Assert.Equal(13, guided);
    }

    [Fact]
    public void IsGuided_StrengthZeroIsNeverGuided()
    {
        SamSamplerParams p = SamSamplerParams.SlatDefault with { Strength = 0f };

        Assert.False(SamFlowSampler.IsGuided(0.0, p));
    }

    [Fact]
    public void Sample_GuidedStepFormsOnePlusSCondMinusSUncondOnEveryLatent()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> noise = Latents(0f, 1f);
        SamSamplerParams p = new() { Steps = 1, RescaleT = 1.0, Strength = 2f };

        // cond = 3, uncond = 1: (1 + 2) · 3 − 2 · 1 = 7 over dt = 1.
        Dictionary<string, Tensor> result = SamFlowSampler.Sample(backend, noise, p, (x, t, d, uncond) => Constant(x, uncond ? 1f : 3f));

        Assert.All(result["shape"].AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(7f, v, 5));
        Assert.All(result["translation"].AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(8f, v, 5));
        Dispose(result);
        Dispose(noise);
    }

    [Fact]
    public void Sample_PassesScaledTimeAndShortcutStepAndSkipsTheUnconditionalPassOutsideTheInterval()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> noise = Latents(0f, 0f);
        SamSamplerParams p = new() { Steps = 2, RescaleT = 1.0, Strength = 1f, IntervalLow = 0, IntervalHigh = 400 };
        List<(float T, float D, bool Uncond)> calls = [];

        Dictionary<string, Tensor> result = SamFlowSampler.Sample(backend, noise, p, (x, t, d, uncond) =>
        {
            calls.Add((t, d, uncond));
            return Constant(x, 0f);
        });

        // t = 0 is guided (cond then uncond); t = 0.5 scales to 500 > 400 and runs the conditional pass only.
        Assert.Equal([(0f, 0f, false), (0f, 0f, true), (500f, 0f, false)], calls);
        Dispose(result);
        Dispose(noise);
    }

    [Fact]
    public void Sample_DoesNotModifyTheNoiseAndFailsOnAMissingVelocity()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> noise = Latents(2f, 3f);
        SamSamplerParams p = new() { Steps = 1, RescaleT = 1.0, Strength = 0f };

        Assert.Throws<InvalidOperationException>(() => SamFlowSampler.Sample(backend, noise, p, (x, t, d, uncond) =>
            Constant(new Dictionary<string, Tensor> { ["shape"] = x["shape"] }, 0f)));

        Assert.All(noise["shape"].AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(2f, v));
        Dispose(noise);
    }

    [Fact]
    public void Validate_RejectsEmptyIntervalAndNonPositiveSteps()
    {
        Assert.Throws<ArgumentException>(() => (SamSamplerParams.StructureDefault with { Steps = 0 }).Validate());
        Assert.Throws<ArgumentException>(() => (SamSamplerParams.StructureDefault with { IntervalLow = 600 }).Validate());
        SamSamplerParams.StructureDefault.Validate();
    }
}
