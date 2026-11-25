using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>Self-consistency of the two solvers, with the answers in closed form. These are not parity with upstream; that is the parity step,
/// against the Python sampler. The renoise noise is recorded so the call order and the per-stream inputs are checked, which parity depends on.</summary>
public sealed class DuoSamplingTests
{
    [Fact]
    public void EulerWithAConstantVelocityAddsItOverTheWholeFlow()
    {
        // The flow runs from 0 to 1, so the steps sum to a displacement of exactly one velocity, whatever the shift.
        float[] x0Video = [0.5f, -1f];
        float[] x0Audio = [0.25f];
        (float[] video, float[] audio) = DuoSampling.Sample(x0Video, x0Audio, steps: 4, shift: 2.0, DuoSolver.Euler,
            (v, a, t) => ([2f, 3f], [4f]));
        Assert.Equal(2.5f, video[0], 5);
        Assert.Equal(2f, video[1], 5);
        Assert.Equal(4.25f, audio[0], 5);
    }

    [Fact]
    public void ConsistencyWithOneStepIsTheNoisyStatePlusThePrediction()
    {
        (float[] video, float[] audio) = DuoSampling.Sample([1f], [2f], steps: 1, shift: 1.0, DuoSolver.ConsistencyRenoise,
            (v, a, t) => ([3f], [5f]));
        Assert.Equal(4f, video[0], 5);
        Assert.Equal(7f, audio[0], 5);
    }

    [Fact]
    public void ConsistencyRenoiseWithZeroNoiseScalesByTheFlowTime()
    {
        // With shift 1 the flow is [0, 0.5, 1]. Step 0 predicts x0 + v and re-noises to t = 0.5 with zero noise: 0.5 * (x0 + v).
        // Step 1 at t = 0.5 predicts 0.5 * (x0 + v) + 0.5 * v, which is 0.5 * x0 + v.
        (float[] video, _) = DuoSampling.Sample([1f], [0f], steps: 2, shift: 1.0, DuoSolver.ConsistencyRenoise,
            (v, a, t) => ([2f], [0f]), (step, stream) => [0f]);
        Assert.Equal(0.5f * 1f + 2f, video[0], 5);
    }

    [Fact]
    public void ConsistencyRenoiseCallsRunInStepAndStreamOrderAndTheOutputsAreClosedForm()
    {
        // Three steps on shift 1: the flow is [0, 1/3, 2/3, 1]. Renoise runs for steps 0 and 1 only, never on the last step.
        // Video: x = 1, velocity 2, noise 3 at step 0 and 5 at step 1. Audio: x = 0, velocity 4, noise -1 and -2.
        // Step by step, with x0 = x + (1 - t) * v and x' = x0 * t' + n * (1 - t'):
        //   video: x0 = 3, x = 3;   x0 = 3 + (2/3) * 2 = 13/3, x = 26/9 + 5/3 = 41/9;   x0 = 41/9 + (1/3) * 2 = 47/9.
        //   audio: x0 = 4, x = 2/3; x0 = 2/3 + (2/3) * 4 = 10/3, x = 20/9 - 2/3 = 14/9; x0 = 14/9 + (1/3) * 4 = 26/9.
        List<(int Step, int Stream)> calls = [];
        float[] Noise(int step, int stream)
        {
            calls.Add((step, stream));
            return stream == 0 ? new[] { 3f + 2f * step } : new[] { -1f - step };
        }

        (float[] video, float[] audio) = DuoSampling.Sample([1f], [0f], steps: 3, shift: 1.0, DuoSolver.ConsistencyRenoise,
            (v, a, t) => ([2f], [4f]), Noise);

        Assert.Equal(new[] { (0, 0), (0, 1), (1, 0), (1, 1) }, calls);
        Assert.Equal(47f / 9f, video[0], 5);
        Assert.Equal(26f / 9f, audio[0], 5);
    }

    [Fact]
    public void ConsistencyRenoiseRefusesANoiseArrayOfTheWrongLength()
    {
        ArgumentException error = Assert.Throws<ArgumentException>(() => DuoSampling.Sample([1f, 2f], [1f], steps: 2, shift: 1.0,
            DuoSolver.ConsistencyRenoise, (v, a, t) => ([1f, 1f], [1f]), (step, stream) => new float[1]));
        // The video stream has two values, so the one-value noise is named by step and stream at the call site.
        Assert.Contains("renoise(0, 0) has 1 values; the stream has 2", error.Message);
    }

    [Fact]
    public void ConsistencyRequiresANoiseSourceForMoreThanOneStep()
    {
        Assert.Throws<ArgumentNullException>(() => DuoSampling.Sample([1f], [1f], steps: 2, shift: 1.0, DuoSolver.ConsistencyRenoise,
            (v, a, t) => ([1f], [1f])));
    }

    [Fact]
    public void ConsistencyOnTheDistillGridRecoversTheCleanSample()
    {
        // The loop test in DuoRopeAndSamplerTests, driven through DuoSampling.Sample on the real distill grid (4 steps, shift 5).
        // With the perfect velocity clean - noise and constant noise, every step's x0 is the clean sample, so the output is too.
        float[] clean = [0.5f, -0.5f];
        float[] noise = [1f, 1f];
        float[] velocity = [clean[0] - noise[0], clean[1] - noise[1]];
        (float[] video, _) = DuoSampling.Sample(noise, noise, DuoSchedule.DistillSteps, DuoSchedule.DistillShift,
            DuoSolver.ConsistencyRenoise, (v, a, t) => (velocity, velocity), (step, stream) => noise);
        Assert.Equal(clean[0], video[0], 1e-5f);
        Assert.Equal(clean[1], video[1], 1e-5f);
    }

    [Fact]
    public void ANonfinitePredictionIsRefused()
    {
        Assert.Throws<InvalidOperationException>(() => DuoSampling.Sample([1f], [1f], steps: 2, shift: 1.0, DuoSolver.Euler,
            (v, a, t) => ([float.NaN], [1f])));
    }
}
