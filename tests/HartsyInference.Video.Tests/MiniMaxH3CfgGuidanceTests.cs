using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Sampling;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>The shared CFG combine and fused Euler step that the MiniMax-H3 denoise loop calls for guided steps.</summary>
public sealed class MiniMaxH3CfgGuidanceTests
{
    [Fact]
    public void FusedEulerStepAtGuidanceOneEqualsTheSingleVelocityStep()
    {
        IBackend backend = new CpuBackend();
        TensorShape shape = new TensorShape(2, 3);
        using Tensor cond = TensorFrom([0.5f, -1f, 1.5f, -2f, 2.5f, -3f], shape);
        using Tensor uncond = TensorFrom([9f, 9f, 9f, 9f, 9f, 9f], shape);
        using Tensor single = TensorFrom([1f, 2f, 3f, 4f, 5f, 6f], shape);
        using Tensor guidedOne = TensorFrom([1f, 2f, 3f, 4f, 5f, 6f], shape);

        backend.CfgEulerStep(single, cond, cond, 1f, -0.25f);
        backend.CfgEulerStep(guidedOne, cond, uncond, 1f, -0.25f);

        Assert.Equal(Snapshot(single), Snapshot(guidedOne));
    }

    [Fact]
    public void FusedEulerStepAboveOneRunsTheUnconditionalBranch()
    {
        IBackend backend = new CpuBackend();
        TensorShape shape = new TensorShape(1, 2);
        using Tensor cond = TensorFrom([2f, 4f], shape);
        using Tensor uncond = TensorFrom([1f, 1f], shape);
        using Tensor z = TensorFrom([10f, 10f], shape);

        // v = uncond + 5·(cond − uncond) = [6, 16]; z += v·(−0.5)
        backend.CfgEulerStep(z, cond, uncond, 5f, -0.5f);

        AssertClose([7f, 2f], Snapshot(z), 1e-5f, "CFG 5 Euler step");
    }

    [Fact]
    public void CombineCfgMatchesTheGuidedFormulaAndSkipsAtOne()
    {
        IBackend backend = new CpuBackend();
        TensorShape shape = new TensorShape(1, 2);
        using Tensor cond = TensorFrom([2f, 4f], shape);
        using Tensor uncond = TensorFrom([1f, 1f], shape);
        using Tensor output = new Tensor(shape, DType.F32);

        SamplerMath.CombineCfg(backend, output, new DenoisePrediction(cond, uncond, 5f, ownsTensors: false));
        AssertClose([6f, 16f], Snapshot(output), 1e-5f, "CFG 5 combine");

        SamplerMath.CombineCfg(backend, output, new DenoisePrediction(cond, uncond, 1f, ownsTensors: false));
        AssertClose([2f, 4f], Snapshot(output), 1e-6f, "CFG 1 combine returns cond");
    }

    [Fact]
    public void NegativePromptPresentsAsPlainTextWithNoVisionSegments()
    {
        // The unconditional branch encodes the negative prompt with no reference conditions: its presentation is the
        // raw text ids only, tagged as text, with no vision blocks.
        MiniMaxH3TextEncoding.Encoded negative = MiniMaxH3TextEncoding.Build(
            static text => text.Select(static c => (int)c).ToArray(), "blurry");

        Assert.Equal([(int)'b', (int)'l', (int)'u', (int)'r', (int)'r', (int)'y'], negative.TokenIds);
        Assert.All(negative.ModalityTags, static tag => Assert.Equal(MiniMaxH3TextEncoding.TextTag, tag));
        Assert.Empty(negative.VisionBlockTokenCounts);
    }

    private static Tensor TensorFrom(float[] values, TensorShape shape)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static float[] Snapshot(Tensor tensor) => tensor.AsReadOnlySpan<float>().ToArray();

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"{label}: index {i} expected {expected[i]} got {actual[i]}");
        }
    }
}
