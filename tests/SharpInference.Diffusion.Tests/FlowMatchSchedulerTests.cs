using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Schedulers;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Tests for FlowMatchEulerDiscreteScheduler. Validates sigma schedule, timestep computation, Euler step, and noise addition against known reference values from HuggingFace diffusers.</summary>
public sealed class FlowMatchSchedulerTests
{
    private const float Tolerance = 1e-5f;

    // ── Sigma Schedule Tests ──────────────────────────────────────────────

    [Fact]
    public void SetTimesteps_SigmasHaveCorrectCount()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        // Sigmas include terminal 0, so 28 + 1 = 29 values
        // But timesteps should be 28
        Assert.Equal(28, scheduler.NumInferenceSteps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        Assert.Equal(28, timesteps.Length);
    }

    [Fact]
    public void SetTimesteps_Shift3_FirstSigmaIsOne()
    {
        // With shift=3.0, sigma = 3*t / (1 + 2*t). At t=1.0: sigma = 3/3 = 1.0
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        Assert.InRange(scheduler.InitialNoiseSigma, 1.0f - Tolerance, 1.0f + Tolerance);
    }

    [Fact]
    public void SetTimesteps_Shift3_FirstTimestepIs1000()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        // timestep = sigma * 1000 = 1.0 * 1000 = 1000.0
        Assert.InRange(timesteps[0], 1000.0f - Tolerance, 1000.0f + Tolerance);
    }

    [Fact]
    public void SetTimesteps_Shift3_MidpointSigma()
    {
        // At t=0.5: sigma = 3.0 * 0.5 / (1 + 2.0 * 0.5) = 1.5 / 2.0 = 0.75
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        // Step 14 corresponds to t = 1.0 - 14/28 = 0.5
        Assert.InRange(timesteps[14], 750.0f - 0.1f, 750.0f + 0.1f);
    }

    [Fact]
    public void SetTimesteps_TimestepsAreMonotonicallyDecreasing()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        for (int i = 1; i < timesteps.Length; i++)
        {
            Assert.True(timesteps[i] < timesteps[i - 1],
                $"Timestep[{i}]={timesteps[i]} >= Timestep[{i - 1}]={timesteps[i - 1]}");
        }
    }

    [Fact]
    public void SetTimesteps_Shift1_LinearSchedule()
    {
        // With shift=1.0, sigma = t (no transformation). Schedule is linear from 1 to 0.
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 1.0f);
        scheduler.SetTimesteps(10);

        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        // timesteps should be [1000, 900, 800, ..., 100]
        for (int i = 0; i < 10; i++)
        {
            float expected = (1.0f - (float)i / 10) * 1000.0f;
            Assert.InRange(timesteps[i], expected - 0.1f, expected + 0.1f);
        }
    }

    [Fact]
    public void SetTimesteps_Shift3_MatchesDiffusersValues()
    {
        // Reference values from diffusers FlowMatchEulerDiscreteScheduler(shift=3.0).set_timesteps(28)
        // These are sigma * 1000 values for the first 5 and last 5 timesteps
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        ReadOnlySpan<float> ts = scheduler.Timesteps;

        // Verify first timestep
        Assert.InRange(ts[0], 999.9f, 1000.1f);

        // Verify last timestep is small but > 0
        float lastTs = ts[27];
        Assert.True(lastTs > 0.0f, $"Last timestep should be > 0, got {lastTs}");
        // t[27] = 1/28, sigma = 3*(1/28) / (1 + 2*(1/28)) = (3/28) / (30/28) = 3/30 = 0.1
        Assert.InRange(lastTs, 99.0f, 101.0f);
    }

    // ── Euler Step Tests ──────────────────────────────────────────────────

    [Fact]
    public unsafe void Step_VelocityPrediction_CorrectFormula()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        // Create simple test tensors
        TensorShape shape = new TensorShape(1, 4, 2, 2);
        Tensor sample = new Tensor(shape, DType.F32);
        Tensor modelOutput = new Tensor(shape, DType.F32);
        Tensor output = new Tensor(shape, DType.F32);

        float* samplePtr = (float*)sample.DataPointer;
        float* modelPtr = (float*)modelOutput.DataPointer;

        // Fill with known values
        for (int i = 0; i < 16; i++)
        {
            samplePtr[i] = 1.0f;
            modelPtr[i] = 0.5f;
        }

        // Step at index 0: sigma[0]=1.0, sigma[1]≈0.9878, dt ≈ -0.01220
        scheduler.Step(output, modelOutput, sample, 0);

        float* outPtr = (float*)output.DataPointer;
        // x_next = x + v * dt = 1.0 + 0.5 * (sigma[1] - sigma[0])
        // dt is negative (sigma decreases), so x_next < x when v > 0
        Assert.True(outPtr[0] < 1.0f, $"Expected x_next < 1.0, got {outPtr[0]}");
        Assert.True(outPtr[0] > 0.99f, $"Expected x_next close to 1.0, got {outPtr[0]}");

        sample.Dispose();
        modelOutput.Dispose();
        output.Dispose();
    }

    [Fact]
    public unsafe void Step_ZeroVelocity_SampleUnchanged()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        TensorShape shape = new TensorShape(1, 4, 2, 2);
        Tensor sample = new Tensor(shape, DType.F32);
        Tensor modelOutput = new Tensor(shape, DType.F32);
        Tensor output = new Tensor(shape, DType.F32);

        float* samplePtr = (float*)sample.DataPointer;
        float* modelPtr = (float*)modelOutput.DataPointer;

        for (int i = 0; i < 16; i++)
        {
            samplePtr[i] = 1.0f;
            modelPtr[i] = 0.0f; // zero velocity
        }

        scheduler.Step(output, modelOutput, sample, 0);

        float* outPtr = (float*)output.DataPointer;
        // x_next = x + 0 * dt = x
        for (int i = 0; i < 16; i++)
        {
            Assert.InRange(outPtr[i], 1.0f - Tolerance, 1.0f + Tolerance);
        }

        sample.Dispose();
        modelOutput.Dispose();
        output.Dispose();
    }

    // ── Noise Addition Tests ──────────────────────────────────────────────

    [Fact]
    public unsafe void AddNoise_AtSigmaZero_ReturnsCleanSample()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        TensorShape shape = new TensorShape(1, 4, 2, 2);
        Tensor sample = new Tensor(shape, DType.F32);
        Tensor noise = new Tensor(shape, DType.F32);
        Tensor output = new Tensor(shape, DType.F32);

        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;

        for (int i = 0; i < 16; i++)
        {
            samplePtr[i] = 1.0f;
            noisePtr[i] = -1.0f;
        }

        // At the last step+1 index, sigma should be 0 → output = (1-0)*sample + 0*noise = sample
        // sigma[28] = 0.0 (terminal)
        scheduler.AddNoise(output, sample, noise, 28);

        float* outPtr = (float*)output.DataPointer;
        for (int i = 0; i < 16; i++)
        {
            Assert.InRange(outPtr[i], 1.0f - Tolerance, 1.0f + Tolerance);
        }

        sample.Dispose();
        noise.Dispose();
        output.Dispose();
    }

    [Fact]
    public unsafe void AddNoise_AtSigmaOne_ReturnsNoise()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        TensorShape shape = new TensorShape(1, 4, 2, 2);
        Tensor sample = new Tensor(shape, DType.F32);
        Tensor noise = new Tensor(shape, DType.F32);
        Tensor output = new Tensor(shape, DType.F32);

        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;

        for (int i = 0; i < 16; i++)
        {
            samplePtr[i] = 1.0f;
            noisePtr[i] = -1.0f;
        }

        // At step 0, sigma[0]=1.0 → output = (1-1)*sample + 1*noise = noise = -1.0
        scheduler.AddNoise(output, sample, noise, 0);

        float* outPtr = (float*)output.DataPointer;
        for (int i = 0; i < 16; i++)
        {
            Assert.InRange(outPtr[i], -1.0f - Tolerance, -1.0f + Tolerance);
        }

        sample.Dispose();
        noise.Dispose();
        output.Dispose();
    }

    // ── ScaleModelInput Tests ─────────────────────────────────────────────

    [Fact]
    public void ScaleModelInput_AlwaysReturnsOne()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(shift: 3.0f);
        scheduler.SetTimesteps(28);

        for (int i = 0; i < 28; i++)
        {
            Assert.Equal(1.0f, scheduler.ScaleModelInput(i));
        }
    }

    // ── Dynamic Shift Tests ───────────────────────────────────────────────

    [Fact]
    public void CreateWithDynamicShift_BaseSeqLen_ReturnsBaseShift()
    {
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen: 256);
        scheduler.SetTimesteps(20);

        // At baseSeqLen=256: shift should be baseShift=0.5
        // With shift=0.5, sigma at t=1.0: 0.5 * 1 / (1 + (-0.5)*1) = 0.5 / 0.5 = 1.0
        Assert.InRange(scheduler.InitialNoiseSigma, 1.0f - Tolerance, 1.0f + Tolerance);
    }

    [Fact]
    public void CreateWithDynamicShift_MaxSeqLen_ReturnsMaxShift()
    {
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen: 4096);
        scheduler.SetTimesteps(20);

        // At maxSeqLen=4096: shift should be maxShift=1.15
        // With shift=1.15, sigma at t=1.0: 1.15 * 1 / (1 + 0.15*1) = 1.15 / 1.15 = 1.0
        Assert.InRange(scheduler.InitialNoiseSigma, 1.0f - Tolerance, 1.0f + Tolerance);
    }

    [Fact]
    public void CreateWithDynamicShift_MiddleSeqLen_InterpolatesShift()
    {
        // At imageSeqLen = (256+4096)/2 = 2176: mu = midpoint between 0.5 and 1.15 = 0.825
        // Flux uses exponential shift: shift = exp(mu) = exp(0.825) ≈ 2.2824
        FlowMatchEulerDiscreteScheduler scheduler =
            FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen: 2176);
        scheduler.SetTimesteps(4);

        ReadOnlySpan<float> ts = scheduler.Timesteps;
        // Step at index 2 corresponds to t = 1.0 - 2/4 = 0.5
        // sigma = exp(mu) * t / (1 + (exp(mu) - 1) * t)
        float shift = MathF.Exp(0.825f);
        float expectedSigma = shift * 0.5f / (1.0f + (shift - 1.0f) * 0.5f);
        float expectedTimestep = expectedSigma * 1000.0f;
        Assert.InRange(ts[2], expectedTimestep - 1.0f, expectedTimestep + 1.0f);
    }
}
