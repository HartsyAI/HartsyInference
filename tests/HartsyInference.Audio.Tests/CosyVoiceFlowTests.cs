using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for the OT-CFM Euler solver — the exactly-specified flow-matching
/// integration + classifier-free-guidance combine, verified with analytic stub estimators. The full
/// UNet1D estimator + flow encoder need <c>flow.pt</c> (covered by the env-gated generation test).</summary>
public sealed unsafe class CosyVoiceFlowTests
{
    /// <summary>Returns a constant velocity per call: <c>condValue</c> when μ has any non-zero element
    /// (the conditional branch), <c>uncondValue</c> when μ is all-zero (the CFG unconditional branch).</summary>
    private sealed class ConstEstimator(int melBins, float condValue, float uncondValue) : ICfmEstimator
    {
        public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond)
        {
            bool muNonZero = false;
            float* mp = (float*)mu.DataPointer;
            for (long i = 0; i < mu.ElementCount; i++) if (mp[i] != 0f) { muNonZero = true; break; }
            float v = muNonZero ? condValue : uncondValue;
            Tensor outT = new(x.Shape, DType.F32);
            float* op = (float*)outT.DataPointer;
            for (long i = 0; i < outT.ElementCount; i++) op[i] = v;
            _ = melBins;
            return outT;
        }
    }

    private static Tensor Filled(int melBins, int t, float value)
    {
        Tensor x = new(new TensorShape(1, melBins, t), DType.F32);
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) p[i] = value;
        return x;
    }

    [Fact]
    public void Solve_NoCfg_IntegratesConstantVelocityToUnitTime()
    {
        using CpuBackend backend = new();
        const int mel = 80, t = 12;
        // x0 recovered by a zero-velocity estimator at the same seed.
        ConditionalCfm zero = new(new ConstEstimator(mel, 0f, 0f), mel);
        ConditionalCfm three = new(new ConstEstimator(mel, 3f, 3f), mel);
        using Tensor mu = Filled(mel, t, 1f);
        using Tensor spk = Filled(mel, 1, 0.5f);
        using Tensor cond = new(new TensorShape(1, mel, t), DType.F32);

        using Tensor x0 = zero.Solve(backend, mu, spk, cond, numSteps: 10, cfgRate: 0f, seed: 7);
        using Tensor xV = three.Solve(backend, mu, spk, cond, numSteps: 10, cfgRate: 0f, seed: 7);

        float* a = (float*)x0.DataPointer;
        float* b = (float*)xV.DataPointer;
        for (long i = 0; i < x0.ElementCount; i++)
            Assert.True(MathF.Abs((b[i] - a[i]) - 3f) < 1e-4f, $"∫ constant v=3 over [0,1] should add 3; got {b[i] - a[i]}");
    }

    [Fact]
    public void Solve_Cfg_AppliesGuidanceCombine()
    {
        using CpuBackend backend = new();
        const int mel = 80, t = 8;
        const float cfg = 0.7f, condV = 2f, uncondV = 1f;
        ConditionalCfm zero = new(new ConstEstimator(mel, 0f, 0f), mel);
        ConditionalCfm est = new(new ConstEstimator(mel, condV, uncondV), mel);
        using Tensor mu = Filled(mel, t, 1f);
        using Tensor spk = Filled(mel, 1, 0f);
        using Tensor cond = new(new TensorShape(1, mel, t), DType.F32);

        using Tensor x0 = zero.Solve(backend, mu, spk, cond, numSteps: 10, cfgRate: 0f, seed: 3);
        using Tensor x = est.Solve(backend, mu, spk, cond, numSteps: 10, cfgRate: cfg, seed: 3);

        float expected = (1f + cfg) * condV - cfg * uncondV;   // 2.7
        float* a = (float*)x0.DataPointer;
        float* b = (float*)x.DataPointer;
        for (long i = 0; i < x.ElementCount; i++)
            Assert.True(MathF.Abs((b[i] - a[i]) - expected) < 1e-4f, $"CFG combine should add {expected}; got {b[i] - a[i]}");
    }

    [Fact]
    public void Solve_IsDeterministic_ForFixedSeed()
    {
        using CpuBackend backend = new();
        const int mel = 80, t = 10;
        ConditionalCfm est = new(new ConstEstimator(mel, 1.3f, 0.4f), mel);
        using Tensor mu = Filled(mel, t, 1f);
        using Tensor spk = Filled(mel, 1, 0f);
        using Tensor cond = new(new TensorShape(1, mel, t), DType.F32);
        using Tensor a = est.Solve(backend, mu, spk, cond, 10, 0.7f, seed: 99);
        using Tensor b = est.Solve(backend, mu, spk, cond, 10, 0.7f, seed: 99);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        for (long i = 0; i < a.ElementCount; i++) Assert.Equal(ap[i], bp[i]);
    }
}
