using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for the flow-matching DPM++ 2M midpoint solver Wan-Animate-2 samples with. Every expected value is
/// derived from the reference's own formulas (<c>get_sampling_sigmas</c> and
/// <c>multistep_dpm_solver_second_order_update</c>), not from a recorded run of this code.</summary>
public unsafe class FlowDpmPlusPlus2MSchedulerTests
{
    /// <summary><c>shift·s / (1 + (shift−1)·s)</c> with <c>s = (n−i)/n</c>, which reduces to <c>m(n−i)/(n+(m−1)(n−i))</c>.</summary>
    private static double ExpectedSigma(int i, int steps, double shift)
    {
        double s = (double)(steps - i) / steps;
        return shift * s / (1.0 + (shift - 1.0) * s);
    }

    /// <summary>The grid is <c>linspace(1, 0, steps+1)[:steps]</c> shifted, plus a terminal 0 — it must reach exactly
    /// zero and must NOT be floored at <c>1/num_train_timesteps</c> the way UniPC's is. The two solvers disagree on
    /// the grid as well as the update, so a correct solver over UniPC's grid is still wrong.</summary>
    [Theory]
    [InlineData(10)]
    [InlineData(40)]
    public void SetTimesteps_MatchesGetSamplingSigmas_AndReachesExactlyZero(int steps)
    {
        const float shift = 5f;
        FlowDpmPlusPlus2MScheduler sched = new();
        sched.SetTimesteps(steps, shift);

        Assert.Equal(steps, sched.NumInferenceSteps);
        Assert.Equal(steps + 1, sched.Sigmas.Length);
        Assert.Equal(0f, sched.Sigmas[steps]);
        Assert.Equal(1f, sched.Sigmas[0]);
        for (int i = 0; i < steps; i++)
            Assert.Equal((float)ExpectedSigma(i, steps, shift), sched.Sigmas[i], 6);

        // Closed form of the last non-zero point: shift / (steps + shift − 1). UniPC's grid stops well short of it.
        Assert.Equal(shift / (steps + shift - 1f), sched.Sigmas[steps - 1], 6);
        FlowUniPCMultistepScheduler unipc = new();
        unipc.SetTimesteps(steps, shift);
        Assert.True(unipc.Sigmas[steps - 1] > sched.Sigmas[steps - 1],
            "UniPC's floored grid must not coincide with get_sampling_sigmas — if it does, the grid fix regressed.");
    }

    /// <summary>The reference casts <c>sigma · num_train_timesteps</c> to int64 before the DiT sees it, so the
    /// conditioning timesteps are truncated, not rounded.</summary>
    [Fact]
    public void Timesteps_AreTruncatedToIntegers_LikeTheReferencesInt64Cast()
    {
        const int steps = 40;
        const float shift = 5f;
        FlowDpmPlusPlus2MScheduler sched = new();
        sched.SetTimesteps(steps, shift);

        for (int i = 0; i < steps; i++)
        {
            double expected = Math.Truncate(ExpectedSigma(i, steps, shift) * 1000.0);
            Assert.Equal((float)expected, sched.Timesteps[i]);
        }
        Assert.Equal(1000f, sched.Timesteps[0]);
        // 5/44 · 1000 = 113.636…, so the last step conditions on 113, not 114.
        Assert.Equal(113f, sched.Timesteps[steps - 1]);
    }

    /// <summary>Flow matching with a constant velocity <c>v = noise − x0</c> has <c>x_t = x0 + sigma·v</c>, so the
    /// converted x0-prediction is exact at every step and the solver must land on x0 whatever order it takes. This
    /// also proves the terminal <c>sigma = 0</c> update is finite (<c>lambda</c> is infinite there — a clamp would
    /// leave a residue).</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(10)]
    [InlineData(40)]
    public void Step_ConstantVelocity_LandsOnX0(int steps)
    {
        const float x0 = 2.0f, noise = -1.0f;
        const float v = noise - x0;

        FlowDpmPlusPlus2MScheduler sched = new();
        sched.SetTimesteps(steps, 5.0f);
        using Tensor sample = new Tensor(new TensorShape(1), DType.F32);
        using Tensor velocity = new Tensor(new TensorShape(1), DType.F32);
        *(float*)sample.DataPointer = noise;
        for (int k = 0; k < steps; k++)
        {
            *(float*)velocity.DataPointer = v;
            sched.Step(sample, velocity);
        }
        Assert.Equal(x0, *(float*)sample.DataPointer, 5);
    }

    /// <summary>Step 0 has no history and must be first-order; step 1 onwards is the midpoint update. Both are
    /// checked against the reference expressions evaluated by hand on a scalar trajectory, so a wrong D1 scaling or a
    /// wrong <c>0.5</c> factor fails here.</summary>
    [Fact]
    public void Step_SecondOrderMidpoint_MatchesTheReferenceExpression()
    {
        const int steps = 4;
        const float shift = 5f;
        FlowDpmPlusPlus2MScheduler sched = new();
        sched.SetTimesteps(steps, shift);
        double[] sigma = new double[steps + 1];
        for (int i = 0; i < steps; i++) sigma[i] = (float)ExpectedSigma(i, steps, shift);

        // An arbitrary velocity sequence — the point is the update algebra, not the trajectory.
        float[] velocities = [0.4f, -0.9f, 0.25f, 1.1f];
        using Tensor sample = new Tensor(new TensorShape(1), DType.F32);
        using Tensor velocity = new Tensor(new TensorShape(1), DType.F32);
        *(float*)sample.DataPointer = 0.75f;

        double x = 0.75, m0 = 0, m1 = 0;
        for (int k = 0; k < steps; k++)
        {
            *(float*)velocity.DataPointer = velocities[k];
            sched.Step(sample, velocity);

            m1 = m0;
            m0 = x - sigma[k] * velocities[k];
            double lambdaT = Lambda(sigma[k + 1]), lambdaS0 = Lambda(sigma[k]);
            double h = lambdaT - lambdaS0;
            double alphaT = 1.0 - sigma[k + 1];
            double d0Coeff = alphaT * (Math.Exp(-h) - 1.0);
            double next = sigma[k + 1] / sigma[k] * x - d0Coeff * m0;
            // First-order at step 0 (no history) and at the last step (final_sigmas_type = "zero" forces it at EVERY
            // step count, not only below 15). Everything between is midpoint.
            if (k > 0 && k < steps - 1)
            {
                double r0 = (lambdaS0 - Lambda(sigma[k - 1])) / h;
                next -= 0.5 * d0Coeff * (1.0 / r0) * (m0 - m1);
            }
            x = next;
            Assert.Equal((float)x, *(float*)sample.DataPointer, 5);
        }

        static double Lambda(double s) => Math.Log(1.0 - s) - Math.Log(s);
    }

    /// <summary>The final step is first-order because <c>final_sigmas_type = "zero"</c>, independently of the step
    /// count — at 40 steps too, where the reference's <c>&lt; 15</c> clause does not fire. A second-order final step
    /// would add a <c>0.5·D1</c> term, so feeding a velocity history with a non-zero second difference and landing
    /// exactly on the converted x0 is the discriminating check.</summary>
    [Fact]
    public void Step_FinalStep_IsFirstOrder_AtEveryStepCount()
    {
        foreach (int steps in new[] { 4, 40 })
        {
            FlowDpmPlusPlus2MScheduler sched = new();
            sched.SetTimesteps(steps, 5.0f);
            using Tensor sample = new Tensor(new TensorShape(1), DType.F32);
            using Tensor velocity = new Tensor(new TensorShape(1), DType.F32);
            *(float*)sample.DataPointer = 0.3f;
            for (int k = 0; k < steps - 1; k++)
            {
                *(float*)velocity.DataPointer = 0.2f * k - 0.5f;
                sched.Step(sample, velocity);
            }
            // At sigma_next = 0 a first-order update collapses to x_t = x0 = sample − sigma·v exactly; the extra
            // midpoint term would not vanish.
            float before = *(float*)sample.DataPointer;
            float lastV = 1.75f;
            *(float*)velocity.DataPointer = lastV;
            float sigmaLast = sched.Sigmas[steps - 1];
            sched.Step(sample, velocity);
            Assert.Equal(before - sigmaLast * lastV, *(float*)sample.DataPointer, 5);
        }
    }

    [Fact]
    public void Step_RequiresSetTimesteps_AndEndsTrajectory()
    {
        FlowDpmPlusPlus2MScheduler sched = new();
        using Tensor sample = new Tensor(new TensorShape(1), DType.F32);
        using Tensor vel = new Tensor(new TensorShape(1), DType.F32);
        Assert.Throws<InvalidOperationException>(() => sched.Step(sample, vel));

        sched.SetTimesteps(2, 5.0f);
        sched.Step(sample, vel);
        sched.Step(sample, vel);
        Assert.Throws<InvalidOperationException>(() => sched.Step(sample, vel));

        sched.SetTimesteps(2, 5.0f);
        sched.Step(sample, vel);
    }
}
