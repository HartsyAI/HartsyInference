using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>Self-consistency of the two solvers on constant velocities, where the answers are known in closed form. The flow times are the
/// existing ModelAssets grid (covered by its own tests). These are not parity with upstream; that is the parity step, against the Python sampler.</summary>
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
    public void ConsistencyRequiresANoiseSourceForMoreThanOneStep()
    {
        Assert.Throws<ArgumentNullException>(() => DuoSampling.Sample([1f], [1f], steps: 2, shift: 1.0, DuoSolver.ConsistencyRenoise,
            (v, a, t) => ([1f], [1f])));
    }

    [Fact]
    public void ANonfinitePredictionIsRefused()
    {
        Assert.Throws<InvalidOperationException>(() => DuoSampling.Sample([1f], [1f], steps: 2, shift: 1.0, DuoSolver.Euler,
            (v, a, t) => ([float.NaN], [1f])));
    }
}
