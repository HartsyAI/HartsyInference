using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane: the zero-terminal-SNR schedule, trailing timesteps, and both samplers against an oracle v-prediction model whose data estimate never changes
/// (a correct solver must land on that data).</summary>
public sealed class HunyuanPaintSamplerTests
{
    [Fact]
    public void TrailingTimesteps_FollowDiffusersRounding()
    {
        Assert.Equal([999, 932, 866, 799, 732, 666, 599, 532, 466, 399, 332, 266, 199, 132, 66], HunyuanPaintSchedule.TrailingTimesteps(15));
        Assert.Equal(30, HunyuanPaintSchedule.TrailingTimesteps(30).Length);
        Assert.Equal(999, HunyuanPaintSchedule.TrailingTimesteps(30)[0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => HunyuanPaintSchedule.TrailingTimesteps(0));
    }

    [Fact]
    public void Schedule_HasZeroTerminalSnrAndKeepsTheFirstStep()
    {
        HunyuanPaintSchedule schedule = new();

        Assert.Equal(1 - 0.00085, schedule.AlphasCumprod[0], 6);
        Assert.Equal(Math.Pow(2, -24), schedule.AlphasCumprod[^1]);
        Assert.Equal(4096.0, schedule.Sigmas[^1], 2);
        Assert.True(schedule.AlphasCumprod.Zip(schedule.AlphasCumprod.Skip(1), (a, b) => a > b).All(v => v));
        Assert.InRange(new HunyuanPaintSchedule(rescaleZeroTerminalSnr: false).AlphasCumprod[^1], 1e-3, 1e-2);
    }

    [Theory]
    [InlineData(15)]
    [InlineData(5)]
    [InlineData(1)]
    public void UniPc_WithAnOracleModelEndsOnTheData(int steps)
    {
        const double data = 0.37, noise = -1.2;
        HunyuanPaintUniPcSampler sampler = new(new HunyuanPaintSchedule(), steps);
        float[] sample = [(float)noise, (float)noise];

        for (int step = 0; step < steps; step++)
        {
            double sigma = Sigma(sampler, step);
            double alpha = 1 / Math.Sqrt(sigma * sigma + 1), sigmaT = sigma * alpha;
            double epsilon = (sample[0] - alpha * data) / sigmaT;
            float v = (float)(alpha * epsilon - sigmaT * data);
            sample = sampler.Step([v, v], sample, step, null);
        }

        Assert.Equal(data, sample[0], 3);
        Assert.Equal(data, sample[1], 3);
    }

    [Fact]
    public void EulerAncestral_WithAnOracleModelEndsOnTheDataAndScalesItsInput()
    {
        const double data = -0.25, noise = 0.8;
        HunyuanPaintEulerAncestralSampler sampler = new(new HunyuanPaintSchedule(), 12);
        double top = Sigma(sampler, 0);
        Assert.Equal(top, sampler.InitialNoiseScale, 6);
        float[] sample = [(float)(noise * sampler.InitialNoiseScale)];

        for (int step = 0; step < 12; step++)
        {
            float[] input = sampler.ScaleModelInput(sample, step);
            double sigma = Sigma(sampler, step), alpha = 1 / Math.Sqrt(sigma * sigma + 1), sigmaT = sigma * alpha;
            double epsilon = (input[0] - alpha * data) / sigmaT;
            float v = (float)(alpha * epsilon - sigmaT * data);
            sample = sampler.Step([v], sample, step, [0f]);
        }

        Assert.Equal(data, sample[0], 3);
        Assert.Throws<ArgumentException>(() => sampler.Step([0f], [0f], 0, null));
    }

    // The sigma a sampler uses at a step, recovered from its public schedule.
    private static double Sigma(IHunyuanPaintSampler sampler, int step) => new HunyuanPaintSchedule().Sigmas[sampler.Timesteps[step]];
}
