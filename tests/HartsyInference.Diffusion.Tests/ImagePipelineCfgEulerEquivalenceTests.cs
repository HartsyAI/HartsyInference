using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Locks the three image-pipeline substitutions from host CFG + Euler into the backend's
/// in-place fused operation. HiDream and Kandinsky use the standard shifted scheduler's negative
/// delta; F-Lite uses the same CFG formula with its custom positive <c>t - tNext</c> delta.</summary>
public sealed unsafe class ImagePipelineCfgEulerEquivalenceTests
{
    [Theory]
    [InlineData(3.0f, 28, 0, 5.0f)]
    [InlineData(3.0f, 28, 13, 1.0f)]
    [InlineData(5.0f, 50, 0, 3.5f)]
    [InlineData(5.0f, 50, 49, 7.0f)]
    public void ShiftedFlowMatch_FusedCfgEuler_MatchesLegacyTwoStageMath(
        float shift, int steps, int stepIndex, float guidance)
    {
        FlowMatchEulerDiscreteScheduler scheduler = new(shift);
        scheduler.SetTimesteps(steps);
        using Tensor sample = Values(0.25f, -1.5f, 2.0f, 10.0f, -0.001f, 4.25f);
        using Tensor cond = Values(-0.5f, 3.0f, 0.125f, -2.0f, 9.0f, 1.25f);
        using Tensor uncond = Values(1.5f, -2.0f, 4.0f, 0.5f, -3.0f, 8.0f);
        using Tensor guided = CfgHelper.ApplyCfg(uncond, cond, guidance);
        using Tensor expected = new(sample.Shape, DType.F32);
        scheduler.Step(expected, guided, sample, stepIndex);

        using Tensor actual = Clone(sample);
        nint latentAddress = (nint)actual.DataPointer;
        using IBackend backend = new CpuBackend();
        backend.CfgEulerStep(actual, cond, uncond, guidance, scheduler.Dt(stepIndex));

        Assert.True(latentAddress == (nint)actual.DataPointer, "CfgEulerStep replaced the latent storage instead of updating it in place.");
        AssertClose(expected, actual, 2e-6f);
    }

    [Theory]
    [InlineData(30, 0, 4.0f, 6.0f)]
    [InlineData(30, 14, 2.0f, 6.0f)]
    [InlineData(30, 29, 1.0f, 1.0f)]
    public void FLitePositiveDelta_FusedCfgEuler_MatchesLegacyAccumulator(
        int steps, int stepIndex, float alpha, float guidance)
    {
        float t = ShiftedTime(stepIndex, steps, alpha);
        float tNext = ShiftedTime(stepIndex + 1, steps, alpha);
        float delta = t - tNext;
        Assert.True(delta > 0.0f);

        using Tensor sample = Values(-3.0f, 0.0f, 0.25f, 7.0f, -12.0f, 0.0625f);
        using Tensor cond = Values(2.0f, -4.0f, 1.5f, 0.125f, 8.0f, -2.0f);
        using Tensor uncond = Values(-1.0f, 3.0f, -0.75f, 6.0f, -4.0f, 5.0f);
        using Tensor expected = Clone(sample);

        float* expectedPtr = (float*)expected.DataPointer;
        float* condPtr = (float*)cond.DataPointer;
        float* uncondPtr = (float*)uncond.DataPointer;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            float velocity = uncondPtr[i] + guidance * (condPtr[i] - uncondPtr[i]);
            expectedPtr[i] += delta * velocity;
        }

        using Tensor actual = Clone(sample);
        using IBackend backend = new CpuBackend();
        backend.CfgEulerStep(actual, cond, uncond, guidance, delta);

        AssertClose(expected, actual, 2e-6f);
    }

    [Fact]
    public void FusedStep_DoesNotMutateOrDisposeCallerOwnedPredictions()
    {
        using Tensor latent = Values(1.0f, 2.0f, 3.0f);
        using Tensor cond = Values(4.0f, 5.0f, 6.0f);
        using Tensor uncond = Values(-1.0f, -2.0f, -3.0f);
        float[] condBefore = ToArray(cond);
        float[] uncondBefore = ToArray(uncond);

        using IBackend backend = new CpuBackend();
        backend.CfgEulerStep(latent, cond, uncond, 4.5f, -0.125f);

        Assert.Equal(condBefore, ToArray(cond));
        Assert.Equal(uncondBefore, ToArray(uncond));
    }

    private static float ShiftedTime(int stepIndex, int steps, float alpha)
    {
        float tNorm = (steps - stepIndex) / (float)steps;
        return tNorm * alpha / (1.0f + (alpha - 1.0f) * tNorm);
    }

    private static Tensor Values(params float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(new Span<float>((void*)tensor.DataPointer, values.Length));
        return tensor;
    }

    private static Tensor Clone(Tensor source)
    {
        Tensor clone = new(source.Shape, source.DType);
        Buffer.MemoryCopy(source.DataPointer, clone.DataPointer,
            source.ElementCount * sizeof(float), source.ElementCount * sizeof(float));
        return clone;
    }

    private static float[] ToArray(Tensor tensor)
    {
        float[] values = new float[tensor.ElementCount];
        new ReadOnlySpan<float>((void*)tensor.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        float* expectedPtr = (float*)expected.DataPointer;
        float* actualPtr = (float*)actual.DataPointer;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            Assert.True(float.IsFinite(expectedPtr[i]) && float.IsFinite(actualPtr[i]),
                $"Non-finite value at {i}: expected={expectedPtr[i]:R}, actual={actualPtr[i]:R}");
            float error = MathF.Abs(expectedPtr[i] - actualPtr[i]);
            Assert.True(float.IsFinite(error) && error <= tolerance,
                $"Mismatch at {i}: expected={expectedPtr[i]:R}, actual={actualPtr[i]:R}, error={error:R}");
        }
    }
}
