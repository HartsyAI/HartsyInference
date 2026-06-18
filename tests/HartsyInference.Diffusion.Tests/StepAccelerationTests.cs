using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for the Feature 3 step-acceleration primitives: the LCM and TCD consistency schedulers, the across-step <see cref="FeatureCache"/>, and the CFG guidance-active gate. All CPU-side and deterministic.</summary>
public sealed unsafe class StepAccelerationTests
{
    private static Tensor Filled(int count, float value)
    {
        Tensor t = new Tensor(new TensorShape(1, count), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < count; i++) p[i] = value;
        return t;
    }

    private static Tensor Ramp(int count, float start, float step)
    {
        Tensor t = new Tensor(new TensorShape(1, count), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < count; i++) p[i] = start + i * step;
        return t;
    }

    private static float[] ToArray(Tensor t)
    {
        int count = (int)t.ElementCount;
        float[] result = new float[count];
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < count; i++) result[i] = p[i];
        return result;
    }

    // ── LCM scheduler ─────────────────────────────────────────────────────

    [Fact]
    public void Lcm_Step_NoLongerThrows_AndProducesFiniteOutput()
    {
        LcmScheduler sched = new LcmScheduler();
        sched.SetTimesteps(4);
        using Tensor model = Ramp(64, 0.0f, 0.01f);
        using Tensor sample = Filled(64, 0.5f);
        using Tensor output = Filled(64, 0.0f);

        sched.Step(output, model, sample, stepIndex: 0);

        foreach (float v in ToArray(output))
            Assert.True(float.IsFinite(v), "LCM step produced a non-finite value.");
    }

    [Fact]
    public void Lcm_LastStep_IsDeterministicCleanEstimate()
    {
        // The final step must not re-noise, so two runs match regardless of seed.
        LcmScheduler a = new LcmScheduler { Seed = 1 };
        LcmScheduler b = new LcmScheduler { Seed = 999 };
        a.SetTimesteps(2);
        b.SetTimesteps(2);
        using Tensor model = Ramp(32, 0.1f, 0.005f);
        using Tensor sample = Filled(32, 0.3f);
        using Tensor outA = Filled(32, 0f);
        using Tensor outB = Filled(32, 0f);

        a.Step(outA, model, sample, stepIndex: 1);
        b.Step(outB, model, sample, stepIndex: 1);

        Assert.Equal(ToArray(outA), ToArray(outB));
    }

    [Fact]
    public void Lcm_Step_IsReproducibleForSameSeed()
    {
        LcmScheduler a = new LcmScheduler { Seed = 42 };
        LcmScheduler b = new LcmScheduler { Seed = 42 };
        a.SetTimesteps(4);
        b.SetTimesteps(4);
        using Tensor model = Ramp(48, 0f, 0.01f);
        using Tensor sample = Filled(48, 0.25f);
        using Tensor outA = Filled(48, 0f);
        using Tensor outB = Filled(48, 0f);

        a.Step(outA, model, sample, stepIndex: 0); // non-final → re-noises
        b.Step(outB, model, sample, stepIndex: 0);

        Assert.Equal(ToArray(outA), ToArray(outB));
    }

    // ── TCD scheduler ─────────────────────────────────────────────────────

    [Fact]
    public void Tcd_EtaZero_IsDeterministic()
    {
        // Eta = 0 means no stochastic re-noise, so the output is seed-independent.
        TcdScheduler a = new TcdScheduler { Eta = 0f, Seed = 1 };
        TcdScheduler b = new TcdScheduler { Eta = 0f, Seed = 7 };
        a.SetTimesteps(4);
        b.SetTimesteps(4);
        using Tensor model = Ramp(40, 0f, 0.02f);
        using Tensor sample = Filled(40, 0.4f);
        using Tensor outA = Filled(40, 0f);
        using Tensor outB = Filled(40, 0f);

        a.Step(outA, model, sample, stepIndex: 0);
        b.Step(outB, model, sample, stepIndex: 0);

        Assert.Equal(ToArray(outA), ToArray(outB));
    }

    [Fact]
    public void Tcd_Step_ProducesFiniteOutput()
    {
        TcdScheduler sched = new TcdScheduler { Eta = 0.3f, Seed = 5 };
        sched.SetTimesteps(8);
        using Tensor model = Ramp(64, 0f, 0.005f);
        using Tensor sample = Filled(64, 0.2f);
        using Tensor output = Filled(64, 0f);

        for (int s = 0; s < 8; s++)
            sched.Step(output, model, output, s);

        foreach (float v in ToArray(output))
            Assert.True(float.IsFinite(v), "TCD step produced a non-finite value.");
    }

    // ── FeatureCache gate ─────────────────────────────────────────────────

    [Fact]
    public void FeatureCache_FirstStep_AlwaysComputes()
    {
        using FeatureCache cache = new FeatureCache(threshold: 0.5f);
        using Tensor proxy = Filled(16, 1.0f);
        Assert.True(cache.ShouldCompute(proxy));
    }

    [Fact]
    public void FeatureCache_StableProxy_ReusesAfterFirstCompute()
    {
        using FeatureCache cache = new FeatureCache(threshold: 0.5f, maxConsecutiveReuse: 10);
        using Tensor input = Filled(16, 2.0f);
        using Tensor output = Filled(16, 3.0f);
        using Tensor proxy = Filled(16, 1.0f);

        Assert.True(cache.ShouldCompute(proxy));   // first step computes
        cache.StoreResidual(input, output);

        // Identical proxy → zero drift → reuse.
        using Tensor proxy2 = Filled(16, 1.0f);
        Assert.False(cache.ShouldCompute(proxy2));

        using Tensor recon = Filled(16, 0f);
        cache.ApplyResidual(input, recon);
        // residual was output - input = 1.0, so recon = input + 1.0 = output.
        foreach (float v in ToArray(recon))
            Assert.Equal(3.0f, v, 4);
    }

    [Fact]
    public void FeatureCache_LargeDrift_ForcesRecompute()
    {
        using FeatureCache cache = new FeatureCache(threshold: 0.1f, maxConsecutiveReuse: 10);
        using Tensor input = Filled(16, 1.0f);
        using Tensor output = Filled(16, 2.0f);
        using Tensor proxy = Filled(16, 1.0f);

        Assert.True(cache.ShouldCompute(proxy));
        cache.StoreResidual(input, output);

        // Proxy doubled → relative-L1 = 1.0 ≫ 0.1 threshold → recompute.
        using Tensor proxyFar = Filled(16, 2.0f);
        Assert.True(cache.ShouldCompute(proxyFar));
    }

    [Fact]
    public void FeatureCache_MaxConsecutiveReuse_IsBounded()
    {
        using FeatureCache cache = new FeatureCache(threshold: 100f, maxConsecutiveReuse: 2);
        using Tensor input = Filled(8, 1.0f);
        using Tensor output = Filled(8, 1.5f);
        using Tensor proxy = Filled(8, 1.0f);

        Assert.True(cache.ShouldCompute(proxy));
        cache.StoreResidual(input, output);

        // Huge threshold would allow infinite reuse, but the cap forces a recompute
        // on the 3rd consult (after 2 reuses).
        using Tensor p1 = Filled(8, 1.0f);
        using Tensor p2 = Filled(8, 1.0f);
        using Tensor p3 = Filled(8, 1.0f);
        Assert.False(cache.ShouldCompute(p1));
        Assert.False(cache.ShouldCompute(p2));
        Assert.True(cache.ShouldCompute(p3));
    }

    // ── CFG guidance gate ─────────────────────────────────────────────────

    [Theory]
    [InlineData(0.0f, false)]
    [InlineData(1.0f, false)]
    [InlineData(1.00001f, false)]
    [InlineData(2.5f, true)]
    [InlineData(7.5f, true)]
    public void Cfg_IsGuidanceActive_GatesUncondPass(float scale, bool expected)
    {
        Assert.Equal(expected, CfgHelper.IsGuidanceActive(scale));
    }
}
