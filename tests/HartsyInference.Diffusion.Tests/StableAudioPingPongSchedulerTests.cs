using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Validates the sigma grid against a hand-computed reference of <c>stable_audio_tools.inference.
/// sampling.sample_rf</c>'s logSNR-linspace schedule (Python: <c>logsnr=linspace(-6,2,steps+1);
/// t=sigmoid(-logsnr); t[0]=1; t[-1]=0</c>) and the ping-pong step formula against a hand-worked example.</summary>
public sealed unsafe class StableAudioPingPongSchedulerTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void SetTimesteps_8Steps_MatchesPythonReference()
    {
        float[] expected =
        [
            1.0f, 0.9933071733f, 0.9820137620f, 0.9525741339f,
            0.8807970285f, 0.7310585976f, 0.5f, 0.2689414322f, 0.0f,
        ];

        StableAudioPingPongScheduler scheduler = new();
        scheduler.SetTimesteps(8);

        ReadOnlySpan<float> sigmas = scheduler.Sigmas;
        Assert.Equal(9, sigmas.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(sigmas[i], expected[i] - Tolerance, expected[i] + Tolerance);
    }

    [Fact]
    public void Step_MatchesHandWorkedPingPongFormula()
    {
        StableAudioPingPongScheduler scheduler = new();
        scheduler.SetTimesteps(8);
        float t0 = scheduler.Sigmas[0], t1 = scheduler.Sigmas[1];

        Tensor sample = MakeTensor([2f, -1f, 0.5f]);
        Tensor velocity = MakeTensor([0.5f, 0.25f, -0.5f]);
        Tensor noise = MakeTensor([1f, 1f, 1f]);

        scheduler.Step(sample, velocity, noise, stepIndex: 0);

        float* sp = (float*)sample.DataPointer;
        float x0_0 = 2f - t0 * 0.5f, expected0 = (1f - t1) * x0_0 + t1 * 1f;
        float x0_1 = -1f - t0 * 0.25f, expected1 = (1f - t1) * x0_1 + t1 * 1f;
        float x0_2 = 0.5f - t0 * -0.5f, expected2 = (1f - t1) * x0_2 + t1 * 1f;

        Assert.InRange(sp[0], expected0 - Tolerance, expected0 + Tolerance);
        Assert.InRange(sp[1], expected1 - Tolerance, expected1 + Tolerance);
        Assert.InRange(sp[2], expected2 - Tolerance, expected2 + Tolerance);
    }

    [Fact]
    public void Step_TerminalStep_LandsExactlyOnCleanEstimate()
    {
        StableAudioPingPongScheduler scheduler = new();
        scheduler.SetTimesteps(8);

        Tensor sample = MakeTensor([2f]);
        Tensor velocity = MakeTensor([0.5f]);
        Tensor noise = MakeTensor([999f]);

        scheduler.Step(sample, velocity, noise, stepIndex: 7);

        float* sp = (float*)sample.DataPointer;
        float expected = 2f - scheduler.Sigmas[7] * 0.5f;
        Assert.InRange(sp[0], expected - Tolerance, expected + Tolerance);
    }

    private static Tensor MakeTensor(float[] values)
    {
        Tensor t = new(new TensorShape(values.Length), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < values.Length; i++) p[i] = values[i];
        return t;
    }
}
