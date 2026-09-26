using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Noise sources, the discard-penultimate sigma rule and the CFG++ refusal.</summary>
public sealed class SamplerInfrastructureTests
{
    private static readonly TensorShape Shape = new TensorShape(1, 4, 32, 32);

    /// <summary>Brownian increments over disjoint intervals are unit-variance and independent; over nested intervals
    /// they add up exactly, which is what makes <c>dpmpp_sde</c>'s two overlapping draws correlated.</summary>
    [Fact]
    public void BrownianIncrements_AreUnitVarianceAndAdditive()
    {
        using BrownianNoiseSource source = new(Shape, 42, 0.03f, 14.6f);
        float[] ab = Read(source.Sample(0, 0, 10f, 5f));
        float[] bc = Read(source.Sample(1, 0, 5f, 1f));
        float[] ac = Read(source.Sample(2, 0, 10f, 1f));
        Assert.InRange(Variance(ab), 0.9, 1.1);
        Assert.InRange(Variance(bc), 0.9, 1.1);
        Assert.InRange(Math.Abs(Correlation(ab, bc)), 0.0, 0.05);
        for (int k = 0; k < ac.Length; k++)
        {
            // W(1) − W(10) = (W(5) − W(10)) + (W(1) − W(5)), each normalised by its own interval length.
            double expected = ((ab[k] * Math.Sqrt(5.0)) + (bc[k] * Math.Sqrt(4.0))) / Math.Sqrt(9.0);
            Assert.True(Math.Abs(expected - ac[k]) < 1e-4, $"element {k}: {ac[k]} vs additive {expected}.");
        }
    }

    /// <summary>Consecutive-interval draws down a real SDXL schedule, the pattern the multistep SDE samplers use, are
    /// each unit-variance and mutually uncorrelated.</summary>
    [Fact]
    public void BrownianIncrements_DownASchedule_AreIndependentUnitNormals()
    {
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(20);
        float[] sigmas = scheduler.Sigmas();
        using BrownianNoiseSource source = BrownianNoiseSource.ForSchedule(Shape, 42, sigmas);
        float[]? previous = null;
        for (int i = 0; i < sigmas.Length - 2; i++)
        {
            float[] draw = Read(source.Sample(i, 0, sigmas[i], sigmas[i + 1]));
            Assert.InRange(Variance(draw), 0.9, 1.1);
            if (previous is not null)
            {
                Assert.InRange(Math.Abs(Correlation(previous, draw)), 0.0, 0.05);
            }
            previous = draw;
        }
    }

    /// <summary>The same seed and query order reproduce the path; a different seed does not.</summary>
    [Fact]
    public void BrownianSource_IsDeterministicPerSeed()
    {
        float[] Draw(int seed)
        {
            using BrownianNoiseSource source = new(Shape, seed, 0.03f, 14.6f);
            return Read(source.Sample(0, 0, 14.6f, 7f));
        }
        Assert.Equal(Draw(5), Draw(5));
        Assert.NotEqual(Draw(5), Draw(6));
    }

    /// <summary>Sub-draws within a step are distinct streams; draw 0 keeps the pre-existing per-step seed.</summary>
    [Fact]
    public void StepSeed_SubDrawZeroMatchesTheLegacySeedAndOthersDiffer()
    {
        Assert.Equal(SamplerOps.StepSeed(11, 3), SamplerOps.StepSeed(11, 3, 0));
        Assert.NotEqual(SamplerOps.StepSeed(11, 3, 0), SamplerOps.StepSeed(11, 3, 1));
        Assert.NotEqual(SamplerOps.StepSeed(11, 3, 1), SamplerOps.StepSeed(11, 3, 2));
    }

    /// <summary>uni_pc/dpm_2 keep the step count and both endpoints but take ComfyUI's steps+1 grid minus its
    /// penultimate sigma; every other sampler gets the schedule untouched, and img2img opts out.</summary>
    [Fact]
    public void BuildSigmas_DiscardsThePenultimateSigmaOnlyForTheListedSamplers()
    {
        float[] base10 = new FlowMatchEulerDiscreteScheduler(3.0f) is { } s ? Schedule(s, 10) : [];
        float[] uni = SamplerRegistry.BuildSigmas("uni_pc", null, base10);
        Assert.Equal(base10.Length, uni.Length);
        Assert.Equal(base10[0], uni[0]);
        Assert.Equal(0f, uni[^1]);
        Assert.True(uni[^2] > base10[^2], "The last non-zero sigma must come from the finer grid's third-from-last entry.");
        Assert.Same(base10, SamplerRegistry.BuildSigmas("dpmpp_2m", null, base10));
        Assert.Same(base10, SamplerRegistry.BuildSigmas("uni_pc", null, base10, startsFromNoisedInit: true));
        for (int k = 1; k < uni.Length; k++)
        {
            Assert.True(uni[k] < uni[k - 1], "The resampled schedule must stay strictly descending.");
        }
    }

    /// <summary>ComfyUI's discrete <c>percent_to_sigma</c> at the ends and in the middle of SD's training schedule.</summary>
    [Fact]
    public void EulerDiscreteScheduler_SigmaAtPercentMatchesTheTrainingTable()
    {
        EulerDiscreteScheduler scheduler = new();
        Assert.Equal(0.0, scheduler.SigmaAtPercent(1.0));
        Assert.True(scheduler.SigmaAtPercent(0.0) > 1e8);
        double early = scheduler.SigmaAtPercent(0.2);
        double late = scheduler.SigmaAtPercent(0.8);
        Assert.InRange(early, 3.0, 8.0);
        Assert.InRange(late, 0.1, 0.6);
    }

    /// <summary>Off-schedule sigmas (every second-order and predictor–corrector evaluation) map back to the timestep the
    /// schedule would give; they used to all map to timestep 0 because the lookup assumed descending training sigmas.</summary>
    [Fact]
    public void EulerDiscreteScheduler_OffScheduleSigmaMapsToItsTimestep()
    {
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(20);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 0; i < 20; i++)
        {
            float t = scheduler.TimestepForSigma(scheduler.Sigma(i), -1);
            Assert.True(MathF.Abs(t - timesteps[i]) < 0.5f, $"sigma[{i}] mapped to t={t}, schedule says {timesteps[i]}.");
        }
        float mid = MathF.Sqrt(scheduler.Sigma(5) * scheduler.Sigma(6));
        float tMid = scheduler.TimestepForSigma(mid, 5);
        Assert.InRange(tMid, timesteps[6], timesteps[5]);
    }

    /// <summary>The Karras variant runs from the largest training sigma down, not up.</summary>
    [Fact]
    public void EulerDiscreteScheduler_KarrasScheduleDescends()
    {
        EulerDiscreteScheduler scheduler = new(useKarrasSigmas: true);
        scheduler.SetTimesteps(10);
        float[] sigmas = scheduler.Sigmas();
        Assert.True(sigmas[0] > 14f, $"first Karras sigma {sigmas[0]} should be the training maximum.");
        for (int k = 1; k < sigmas.Length; k++)
        {
            Assert.True(sigmas[k] < sigmas[k - 1]);
        }
        Assert.True(scheduler.Timesteps[0] > 990f);
    }

    /// <summary>The flow-shift estimate recovers the shift of a shifted-linear schedule.</summary>
    [Fact]
    public void EstimateFlowShift_RecoversTheScheduleShift()
    {
        float[] sigmas = Schedule(new FlowMatchEulerDiscreteScheduler(3.16f), 20);
        Assert.InRange(SamplerOptions.EstimateFlowShift(sigmas), 3.15, 3.17);
    }

    /// <summary>CFG++ cannot run without an unconditional branch and says so instead of degrading to Euler.</summary>
    [Fact]
    public void EulerCfgPlusPlus_RefusesAGuidanceFreePair()
    {
        IBackend backend = new CpuBackend();
        TensorShape shape = new TensorShape(1, 4, 4, 4);
        using Tensor z = new Tensor(shape, DType.F32);
        ISampler sampler = SamplerRegistry.Create("euler_cfg_pp", [14.6f, 5f, 1f, 0f], 0);
        sampler.Reset(shape);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => sampler.Step(backend, z, new AliasedPredictor(), 0));
        Assert.Contains("unconditional", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>The <c>_gpu</c> names ComfyUI workflows carry resolve to the same solvers.</summary>
    [Theory]
    [InlineData("dpmpp_sde_gpu", "dpmpp_sde")]
    [InlineData("dpmpp_2m_sde_gpu", "dpmpp_2m_sde")]
    [InlineData("dpmpp_3m_sde_gpu", "dpmpp_3m_sde")]
    public void GpuAliases_ResolveToTheSameSampler(string alias, string canonical)
    {
        Assert.True(SamplerRegistry.IsKnown(alias));
        Assert.Equal(canonical, SamplerRegistry.Create(alias, [14.6f, 1f, 0f], 0).Name);
    }

    private static float[] Schedule(FlowMatchEulerDiscreteScheduler scheduler, int steps)
    {
        scheduler.SetTimesteps(steps);
        return scheduler.Sigmas();
    }

    private static unsafe float[] Read(Tensor t)
    {
        using (t)
        {
            return new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();
        }
    }

    private static double Variance(float[] v)
    {
        double mean = v.Average(x => (double)x);
        return v.Sum(x => (x - mean) * (x - mean)) / v.Length;
    }

    private static double Correlation(float[] a, float[] b)
    {
        double ma = a.Average(x => (double)x);
        double mb = b.Average(x => (double)x);
        double cov = 0.0;
        double va = 0.0;
        double vb = 0.0;
        for (int k = 0; k < a.Length; k++)
        {
            cov += (a[k] - ma) * (b[k] - mb);
            va += (a[k] - ma) * (a[k] - ma);
            vb += (b[k] - mb) * (b[k] - mb);
        }
        return cov / Math.Sqrt(va * vb);
    }

    private sealed class AliasedPredictor : IDenoisePredictor
    {
        public PredictionType Prediction => PredictionType.Epsilon;

        public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            Tensor eps = new Tensor(x.Shape, DType.F32);
            return new DenoisePrediction(eps, eps);
        }
    }
}
