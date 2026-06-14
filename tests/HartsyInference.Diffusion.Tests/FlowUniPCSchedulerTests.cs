using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for the FlowUniPC multistep predictor–corrector: the shifted sigma grid and an analytic
/// constant-velocity trajectory the full predictor/corrector chain must integrate exactly.</summary>
public unsafe class FlowUniPCSchedulerTests
{
    [Fact]
    public void SetTimesteps_AppliesShiftAndTerminalZero()
    {
        FlowUniPCMultistepScheduler sched = new();
        sched.SetTimesteps(4, shift: 5.0f);

        Assert.Equal(4, sched.NumInferenceSteps);
        Assert.Equal(5, sched.Sigmas.Length);
        Assert.Equal(1.0f, sched.Sigmas[0], 5);              // shift fixes σ=1
        Assert.Equal(0.0f, sched.Sigmas[^1]);                // final_sigmas_type = zero
        for (int i = 1; i < sched.Sigmas.Length; i++)
            Assert.True(sched.Sigmas[i] < sched.Sigmas[i - 1], "sigmas must decrease");

        // σ = shift·s / (1 + (shift−1)·s) at s = 0.75 (second grid point of linspace(1, 1/1000, 5)).
        float s = 1.0f + (0.001f - 1.0f) * 1 / 4;
        float expected = 5f * s / (1f + 4f * s);
        Assert.Equal(expected, sched.Sigmas[1], 4);
        Assert.Equal(sched.Sigmas[0] * 1000f, sched.Timesteps[0], 3);
    }

    [Theory]
    [InlineData(3)]    // Matrix-Game distilled step count
    [InlineData(10)]
    [InlineData(50)]   // Matrix-Game base step count
    public void Step_ConstantVelocity_RecoversX0Exactly(int steps)
    {
        // Flow matching with constant velocity v = noise − x0 has x_t = x0 + σ·v; starting from pure noise at σ=1,
        // the converted x0-prediction is exact at every step, so any-order UniPC must land on x0.
        const float x0 = 2.0f, noise = -1.0f;
        const float v = noise - x0;

        FlowUniPCMultistepScheduler sched = new();
        sched.SetTimesteps(steps, shift: 5.0f);

        Tensor sample = new Tensor(new TensorShape(1), DType.F32);
        Tensor velocity = new Tensor(new TensorShape(1), DType.F32);
        *(float*)sample.DataPointer = noise;
        *(float*)velocity.DataPointer = v;

        for (int k = 0; k < steps; k++)
        {
            // Re-derive the exact velocity at the current sample (it is constant for this trajectory).
            *(float*)velocity.DataPointer = v;
            sched.Step(sample, velocity);
        }

        float result = *(float*)sample.DataPointer;
        Assert.True(MathF.Abs(result - x0) < 1e-3f, $"expected x0={x0}, got {result}");
        sample.Dispose();
        velocity.Dispose();
    }

    [Fact]
    public void Step_RequiresSetTimesteps_AndEndsTrajectory()
    {
        FlowUniPCMultistepScheduler sched = new();
        Tensor sample = new Tensor(new TensorShape(1), DType.F32);
        Tensor vel = new Tensor(new TensorShape(1), DType.F32);
        Assert.Throws<InvalidOperationException>(() => sched.Step(sample, vel));

        sched.SetTimesteps(2, shift: 5.0f);
        sched.Step(sample, vel);
        sched.Step(sample, vel);
        Assert.Throws<InvalidOperationException>(() => sched.Step(sample, vel));

        sched.SetTimesteps(2, shift: 5.0f);   // fresh trajectory per segment
        sched.Step(sample, vel);
        sample.Dispose(); vel.Dispose();
    }
}
