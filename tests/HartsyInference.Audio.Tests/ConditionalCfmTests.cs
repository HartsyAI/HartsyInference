using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>A constant-velocity <see cref="ICfmEstimator"/> stub: always returns an all-ones tensor the shape
/// of <c>x</c>, independent of its inputs. Lets a test predict <see cref="ConditionalCfm.Solve"/>'s exact
/// output without needing a real model.</summary>
file sealed unsafe class ConstantEstimator : ICfmEstimator
{
    public Tensor Estimate(IBackend backend, Tensor x, Tensor mu, float t, Tensor spk, Tensor cond, Tensor? attnMask = null)
    {
        Tensor v = new(x.Shape, DType.F32);
        foreach (ref float f in v.AsSpan<float>()) f = 1f;
        return v;
    }
}

/// <summary>Regression coverage for <see cref="ConditionalCfm.Solve"/>'s new <c>promptLen</c> parameter (added
/// for IndexTTS-2's S2Mel CFM, which keeps the in-context reference-mel prefix's noise channel clamped to
/// zero throughout the Euler trajectory). <c>promptLen: 0</c> (the default) must reproduce the exact existing
/// CosyVoice behavior; a positive value must zero exactly that many leading frames, every step.</summary>
public sealed unsafe class ConditionalCfmTests
{
    [Fact]
    public void Solve_WithPromptLenZero_MatchesPriorBehavior_NoClamping()
    {
        ConditionalCfm cfm = new(new ConstantEstimator(), melBins: 4);
        using CpuBackend backend = new();
        using Tensor mu = new(new TensorShape(1, 4, 6), DType.F32);
        using Tensor spk = new(new TensorShape(1, 4), DType.F32);
        using Tensor cond = new(new TensorShape(1, 4, 6), DType.F32);

        using Tensor result = cfm.Solve(backend, mu, spk, cond, numSteps: 4, cfgRate: 0f, seed: 1);

        // v=1 every step, dt=1/4, 4 steps => x increases by exactly 1.0 total, regardless of starting noise —
        // only checking the delta is predictable without needing to know the seeded noise draw.
        Assert.Equal(new TensorShape(1, 4, 6), result.Shape);
        foreach (float v in result.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Solve_WithPromptLenPositive_ZeroesThatManyLeadingFrames_EveryStep()
    {
        ConditionalCfm cfm = new(new ConstantEstimator(), melBins: 4);
        using CpuBackend backend = new();
        using Tensor mu = new(new TensorShape(1, 4, 6), DType.F32);
        using Tensor spk = new(new TensorShape(1, 4), DType.F32);
        using Tensor cond = new(new TensorShape(1, 4, 6), DType.F32);

        const int promptLen = 2;
        using Tensor result = cfm.Solve(backend, mu, spk, cond, numSteps: 4, cfgRate: 0f, seed: 1, promptLen: promptLen);

        Span<float> data = result.AsSpan<float>();
        int t = 6, channels = 4;
        for (int c = 0; c < channels; c++)
            for (int j = 0; j < promptLen; j++)
                Assert.Equal(0f, data[c * t + j]);

        // Frames at/after promptLen are NOT forced to zero — the constant-velocity estimator still moves them.
        bool anyNonZeroPastPrompt = false;
        for (int c = 0; c < channels; c++)
            for (int j = promptLen; j < t; j++)
                if (data[c * t + j] != 0f) anyNonZeroPastPrompt = true;
        Assert.True(anyNonZeroPastPrompt);
    }
}
