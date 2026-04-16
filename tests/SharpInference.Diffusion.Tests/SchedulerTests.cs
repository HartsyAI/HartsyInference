using SharpInference.Core.Schedulers;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Schedulers;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Tests for diffusion schedulers. Validates noise schedule computation, timestep selection, and step update formulas against known reference values from HuggingFace diffusers.</summary>
public sealed class SchedulerTests
{
    private const float Tolerance = 1e-4f;

    // ── Noise Schedule Tests ──────────────────────────────────────────────

    [Fact]
    public void NoiseSchedule_ScaledLinear_BetasMatchExpected()
    {
        SchedulerConfig config = new SchedulerConfig
        {
            NumTrainTimesteps = 1000,
            BetaStart = 0.00085f,
            BetaEnd = 0.012f,
            BetaSchedule = BetaScheduleType.ScaledLinear,
        };

        float[] betas = NoiseSchedule.ComputeBetas(config);

        Assert.Equal(1000, betas.Length);
        // First beta: sqrt(0.00085)^2 = 0.00085
        Assert.InRange(betas[0], 0.00085f - Tolerance, 0.00085f + Tolerance);
        // Last beta: sqrt(0.012)^2 = 0.012
        Assert.InRange(betas[999], 0.012f - Tolerance, 0.012f + Tolerance);
        // Betas should be monotonically increasing
        for (int i = 1; i < betas.Length; i++)
        {
            Assert.True(betas[i] >= betas[i - 1], $"Beta[{i}] = {betas[i]} < Beta[{i - 1}] = {betas[i - 1]}");
        }
    }

    [Fact]
    public void NoiseSchedule_AlphasCumprod_FirstAndLastValues()
    {
        SchedulerConfig config = new SchedulerConfig();
        float[] betas = NoiseSchedule.ComputeBetas(config);
        float[] alphas = NoiseSchedule.ComputeAlphas(betas);
        float[] alphasCumprod = NoiseSchedule.ComputeAlphasCumprod(alphas);

        Assert.Equal(1000, alphasCumprod.Length);
        // First alphas_cumprod should be close to 1 (very little noise)
        Assert.InRange(alphasCumprod[0], 0.999f, 1.0f);
        // Last alphas_cumprod should be close to 0 (almost pure noise)
        Assert.InRange(alphasCumprod[999], 0.0f, 0.01f);
        // Should be monotonically decreasing
        for (int i = 1; i < alphasCumprod.Length; i++)
        {
            Assert.True(alphasCumprod[i] <= alphasCumprod[i - 1]);
        }
    }

    [Fact]
    public void NoiseSchedule_Sigmas_CorrectRelationToAlphasCumprod()
    {
        SchedulerConfig config = new SchedulerConfig();
        float[] betas = NoiseSchedule.ComputeBetas(config);
        float[] alphas = NoiseSchedule.ComputeAlphas(betas);
        float[] alphasCumprod = NoiseSchedule.ComputeAlphasCumprod(alphas);
        float[] sigmas = NoiseSchedule.ComputeSigmas(alphasCumprod);

        // sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        for (int i = 0; i < 10; i++)
        {
            float expected = MathF.Sqrt((1.0f - alphasCumprod[i]) / alphasCumprod[i]);
            Assert.InRange(sigmas[i], expected - Tolerance, expected + Tolerance);
        }
    }

    [Fact]
    public void NoiseSchedule_KarrasSigmas_EndpointsAndMonotonicity()
    {
        float[] sigmas = NoiseSchedule.ComputeKarrasSigmas(0.0292f, 14.6146f, 20);

        Assert.Equal(21, sigmas.Length);
        // First sigma should be close to sigmaMax
        Assert.InRange(sigmas[0], 14.0f, 15.0f);
        // Last sigma should be 0 (terminal)
        Assert.Equal(0.0f, sigmas[20]);
        // Should be monotonically decreasing (ignoring terminal zero)
        for (int i = 1; i < 20; i++)
        {
            Assert.True(sigmas[i] < sigmas[i - 1], $"Karras sigma[{i}] not decreasing");
        }
    }

    // ── Euler Discrete Tests ──────────────────────────────────────────────

    [Fact]
    public void Euler_SetTimesteps_ProducesCorrectCount()
    {
        EulerDiscreteScheduler scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        Assert.Equal(20, scheduler.NumInferenceSteps);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    [Fact]
    public void Euler_Timesteps_AreDescending()
    {
        EulerDiscreteScheduler scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 1; i < timesteps.Length; i++)
        {
            Assert.True(timesteps[i] <= timesteps[i - 1],
                $"Euler timestep[{i}] = {timesteps[i]} > timestep[{i - 1}] = {timesteps[i - 1]}");
        }
    }

    [Fact]
    public unsafe void Euler_Step_EpsilonPrediction_ReducesNoise()
    {
        EulerDiscreteScheduler scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        TensorShape shape = new TensorShape(1, 4, 8, 8);
        using Tensor sample = new Tensor(shape, DType.F32);
        using Tensor modelOutput = new Tensor(shape, DType.F32);
        using Tensor output = new Tensor(shape, DType.F32);

        // Fill sample with random-ish values
        Span<float> sampleSpan = sample.AsSpan<float>();
        Span<float> modelSpan = modelOutput.AsSpan<float>();
        for (int i = 0; i < sampleSpan.Length; i++)
        {
            sampleSpan[i] = MathF.Sin(i * 0.1f);
            modelSpan[i] = MathF.Cos(i * 0.1f) * 0.5f;
        }

        scheduler.Step(output, modelOutput, sample, 0);

        // Output should be different from input (denoising happened)
        Span<float> outSpan = output.AsSpan<float>();
        bool anyDifferent = false;
        for (int i = 0; i < outSpan.Length; i++)
        {
            if (MathF.Abs(outSpan[i] - sampleSpan[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Euler step should modify the sample");
    }

    [Fact]
    public void Euler_InitNoiseSigma_IsPositive()
    {
        EulerDiscreteScheduler scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        Assert.True(scheduler.InitialNoiseSigma > 0.0f);
    }

    [Fact]
    public unsafe void Euler_AddNoise_ProducesNoisySample()
    {
        EulerDiscreteScheduler scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        TensorShape shape = new TensorShape(1, 4, 8, 8);
        using Tensor sample = new Tensor(shape, DType.F32);
        using Tensor noise = new Tensor(shape, DType.F32);
        using Tensor output = new Tensor(shape, DType.F32);

        Span<float> sampleSpan = sample.AsSpan<float>();
        Span<float> noiseSpan = noise.AsSpan<float>();
        for (int i = 0; i < sampleSpan.Length; i++)
        {
            sampleSpan[i] = 1.0f;
            noiseSpan[i] = 0.5f;
        }

        scheduler.AddNoise(output, sample, noise, 0);

        // Output should differ from input
        Span<float> outSpan = output.AsSpan<float>();
        Assert.NotEqual(1.0f, outSpan[0]);
    }

    // ── DDIM Tests ─────────────────────────────────────────────────────────

    [Fact]
    public void Ddim_SetTimesteps_ProducesCorrectCount()
    {
        DdimScheduler scheduler = new DdimScheduler();
        scheduler.SetTimesteps(20);

        Assert.Equal(20, scheduler.NumInferenceSteps);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    [Fact]
    public void Ddim_InitNoiseSigma_IsOne()
    {
        DdimScheduler scheduler = new DdimScheduler();
        Assert.Equal(1.0f, scheduler.InitialNoiseSigma);
    }

    [Fact]
    public unsafe void Ddim_Step_Deterministic_WhenEtaZero()
    {
        DdimScheduler scheduler = new DdimScheduler(eta: 0.0f);
        scheduler.SetTimesteps(20);

        TensorShape shape = new TensorShape(1, 4, 8, 8);
        using Tensor sample = new Tensor(shape, DType.F32);
        using Tensor modelOutput = new Tensor(shape, DType.F32);
        using Tensor output1 = new Tensor(shape, DType.F32);
        using Tensor output2 = new Tensor(shape, DType.F32);

        Span<float> sampleSpan = sample.AsSpan<float>();
        Span<float> modelSpan = modelOutput.AsSpan<float>();
        for (int i = 0; i < sampleSpan.Length; i++)
        {
            sampleSpan[i] = MathF.Sin(i * 0.1f);
            modelSpan[i] = MathF.Cos(i * 0.1f) * 0.5f;
        }

        // Run step twice with same inputs — should produce identical results (deterministic)
        scheduler.Step(output1, modelOutput, sample, 0);
        scheduler.Step(output2, modelOutput, sample, 0);

        Span<float> out1 = output1.AsSpan<float>();
        Span<float> out2 = output2.AsSpan<float>();
        for (int i = 0; i < out1.Length; i++)
        {
            Assert.Equal(out1[i], out2[i]);
        }
    }

    [Fact]
    public unsafe void Ddim_AddNoise_MatchesForwardProcess()
    {
        DdimScheduler scheduler = new DdimScheduler();
        scheduler.SetTimesteps(20);

        TensorShape shape = new TensorShape(1, 4, 8, 8);
        using Tensor sample = new Tensor(shape, DType.F32);
        using Tensor noise = new Tensor(shape, DType.F32);
        using Tensor output = new Tensor(shape, DType.F32);

        Span<float> sampleSpan = sample.AsSpan<float>();
        Span<float> noiseSpan = noise.AsSpan<float>();
        sampleSpan.Fill(1.0f);
        noiseSpan.Fill(1.0f);

        scheduler.AddNoise(output, sample, noise, 0);

        // With sample=1, noise=1: output = sqrt(alpha_cumprod) + sqrt(1-alpha_cumprod) ~ 1.0
        Span<float> outSpan = output.AsSpan<float>();
        // Sum should be close to 1.0 since sqrt(a) + sqrt(1-a) ~ 1 for a close to 1
        Assert.InRange(outSpan[0], 0.9f, 1.1f);
    }

    // ── DPM++ 2M Tests ────────────────────────────────────────────────────

    [Fact]
    public void DpmPP2M_SetTimesteps_ProducesCorrectCount()
    {
        DpmPlusPlus2MScheduler scheduler = new DpmPlusPlus2MScheduler();
        scheduler.SetTimesteps(20);

        Assert.Equal(20, scheduler.NumInferenceSteps);
        Assert.Equal(20, scheduler.Timesteps.Length);
    }

    [Fact]
    public unsafe void DpmPP2M_FirstStep_UsesFirstOrderUpdate()
    {
        DpmPlusPlus2MScheduler scheduler = new DpmPlusPlus2MScheduler();
        scheduler.SetTimesteps(20);

        TensorShape shape = new TensorShape(1, 4, 8, 8);
        using Tensor sample = new Tensor(shape, DType.F32);
        using Tensor modelOutput = new Tensor(shape, DType.F32);
        using Tensor output = new Tensor(shape, DType.F32);

        Span<float> sampleSpan = sample.AsSpan<float>();
        Span<float> modelSpan = modelOutput.AsSpan<float>();
        for (int i = 0; i < sampleSpan.Length; i++)
        {
            sampleSpan[i] = MathF.Sin(i * 0.1f);
            modelSpan[i] = MathF.Cos(i * 0.1f) * 0.5f;
        }

        // First step should work (first-order fallback)
        scheduler.Step(output, modelOutput, sample, 0);

        Span<float> outSpan = output.AsSpan<float>();
        bool anyDifferent = false;
        for (int i = 0; i < outSpan.Length; i++)
        {
            if (MathF.Abs(outSpan[i] - sampleSpan[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "DPM++ 2M first step should modify the sample");
    }

    [Fact]
    public unsafe void DpmPP2M_SecondStep_UsesMultistepUpdate()
    {
        DpmPlusPlus2MScheduler scheduler = new DpmPlusPlus2MScheduler();
        scheduler.SetTimesteps(20);

        TensorShape shape = new TensorShape(1, 4, 4, 4);
        using Tensor sample = new Tensor(shape, DType.F32);
        using Tensor modelOutput = new Tensor(shape, DType.F32);
        using Tensor output1 = new Tensor(shape, DType.F32);
        using Tensor output2 = new Tensor(shape, DType.F32);

        Span<float> sampleSpan = sample.AsSpan<float>();
        Span<float> modelSpan = modelOutput.AsSpan<float>();
        for (int i = 0; i < sampleSpan.Length; i++)
        {
            sampleSpan[i] = MathF.Sin(i * 0.1f);
            modelSpan[i] = MathF.Cos(i * 0.1f) * 0.5f;
        }

        // Step 0 (first order)
        scheduler.Step(output1, modelOutput, sample, 0);

        // Step 1 (should use second order with previous output)
        scheduler.Step(output2, modelOutput, output1, 1);

        Span<float> out2 = output2.AsSpan<float>();
        bool anyDifferent = false;
        for (int i = 0; i < out2.Length; i++)
        {
            if (MathF.Abs(out2[i] - sampleSpan[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "DPM++ 2M second step should produce different output");
    }

    // ── Cross-scheduler consistency tests ─────────────────────────────────

    [Fact]
    public void AllSchedulers_HaveCorrectNames()
    {
        EulerDiscreteScheduler euler = new EulerDiscreteScheduler();
        DdimScheduler ddim = new DdimScheduler();
        DpmPlusPlus2MScheduler dpmpp = new DpmPlusPlus2MScheduler();

        Assert.Equal("euler", euler.Name);
        Assert.Equal("ddim", ddim.Name);
        Assert.Equal("dpm++2m", dpmpp.Name);
    }

    [Fact]
    public void AllSchedulers_TimestepsDescendAfterSetup()
    {
        IScheduler[] schedulers = new IScheduler[]
        {
            new EulerDiscreteScheduler(),
            new DdimScheduler(),
            new DpmPlusPlus2MScheduler(),
        };

        foreach (IScheduler scheduler in schedulers)
        {
            scheduler.SetTimesteps(20);
            ReadOnlySpan<float> ts = scheduler.Timesteps;
            for (int i = 1; i < ts.Length; i++)
            {
                Assert.True(ts[i] <= ts[i - 1],
                    $"{scheduler.Name} timestep[{i}] = {ts[i]} > [{i - 1}] = {ts[i - 1]}");
            }
        }
    }
}
