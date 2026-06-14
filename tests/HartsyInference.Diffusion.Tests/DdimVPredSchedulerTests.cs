using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Analytic gates for the sigmoid-β v-prediction DDIM (Oasis): schedule shape, the DDIM index ladder, and
/// exactness of the v-param update on a known (x₀, ε) trajectory.</summary>
public unsafe class DdimVPredSchedulerTests
{
    [Fact]
    public void SigmoidBetaSchedule_ProducesMonotoneAlphasCumprod()
    {
        DdimVPredScheduler sched = new();
        ReadOnlySpan<float> alphas = sched.AlphasCumprod;
        Assert.Equal(1000, alphas.Length);
        Assert.True(alphas[0] > 0.999f, $"ᾱ₀ = {alphas[0]} should start ≈ 1");
        Assert.True(alphas[^1] < 1e-3f, $"ᾱ_T = {alphas[^1]} should end ≈ 0");
        for (int i = 1; i < alphas.Length; i++)
            Assert.True(alphas[i] <= alphas[i - 1], $"ᾱ must decrease (index {i})");
        Assert.Equal(1f, sched.AlphaCumprod(-1));   // clean state
    }

    [Fact]
    public void BuildNoiseRange_MatchesReferenceLadder()
    {
        DdimVPredScheduler sched = new();
        int[] range = sched.BuildNoiseRange(10);
        Assert.Equal(11, range.Length);
        Assert.Equal(-1, range[0]);
        Assert.Equal(999, range[10]);
        Assert.Equal(99, range[1]);    // linspace(-1, 999, 11) = [-1, 99, 199, ..., 999]
        Assert.Equal(899, range[9]);
    }

    [Theory]
    [InlineData(999, 499)]
    [InlineData(499, 99)]
    [InlineData(99, -1)]    // terminal step lands exactly on x₀
    public void StepFrame_VParamUpdate_IsExactOnKnownTrajectory(int t, int tNext)
    {
        // x_t = √ᾱ·x₀ + √(1−ᾱ)·ε and v = √ᾱ·ε − √(1−ᾱ)·x₀ — the DDIM v-param step must land exactly on x_{t_next}.
        const float x0 = 1.25f, eps = -0.5f;
        DdimVPredScheduler sched = new();
        float at = sched.AlphaCumprod(t);
        float an = sched.AlphaCumprod(tNext);

        Tensor frame = new Tensor(new TensorShape(1), DType.F32);
        Tensor v = new Tensor(new TensorShape(1), DType.F32);
        *(float*)frame.DataPointer = MathF.Sqrt(at) * x0 + MathF.Sqrt(1 - at) * eps;
        *(float*)v.DataPointer = MathF.Sqrt(at) * eps - MathF.Sqrt(1 - at) * x0;

        sched.StepFrame(frame, v, t, tNext);

        float expected = MathF.Sqrt(an) * x0 + MathF.Sqrt(Math.Max(0, 1 - an)) * eps;
        Assert.True(MathF.Abs(*(float*)frame.DataPointer - expected) < 1e-4f,
            $"t={t}→{tNext}: got {*(float*)frame.DataPointer}, expected {expected}");
        frame.Dispose();
        v.Dispose();
    }
}
