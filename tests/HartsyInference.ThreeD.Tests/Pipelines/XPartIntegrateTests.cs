using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.ThreeD.Tests.Pipelines;

public sealed class XPartIntegrateTests
{
    private static Tensor Filled(float value)
    {
        Tensor t = new(new TensorShape(2, 3), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    [Fact]
    public void Integrate_WithAConstantVelocityAddsVelocityTimesTheSumOfTheStepWidths_AndSkipsTheZeroWidthLastStep()
    {
        using CpuBackend backend = new();
        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(5);
        List<float> seen = [];

        using Tensor result = XPartPipeline.Integrate(backend, Filled(1f), sigmas, (x, t) =>
        {
            seen.Add(t);
            return Filled(2f);
        });

        Assert.Equal([0f, 0.25f, 0.5f, 0.75f], seen);
        // The widths sum to 1 (sigma 0 to 1): x = 1 + 2 * 1.
        Assert.All(result.AsSpan<float>().ToArray(), v => Assert.Equal(3f, v, 5));
    }

    [Fact]
    public void Integrate_CountsEveryStepIncludingTheSkippedOne()
    {
        using CpuBackend backend = new();
        int advanced = 0;

        using Tensor result = XPartPipeline.Integrate(backend, Filled(0f), Hunyuan3DFlowSchedule.Sigmas(4), (x, t) => Filled(0f), () => advanced++);

        Assert.Equal(4, advanced);
    }

    [Fact]
    public void Integrate_DisposesTheLatentsWhenTheVelocityThrows()
    {
        using CpuBackend backend = new();
        Tensor latents = Filled(1f);
        int calls = 0;

        Assert.Throws<InvalidOperationException>(() => XPartPipeline.Integrate(backend, latents, Hunyuan3DFlowSchedule.Sigmas(4), (x, t) =>
        {
            if (++calls == 3)
                throw new InvalidOperationException("boom");
            return Filled(1f);
        }));

        Assert.Equal(3, calls);
        // The tensor that was current when the step failed is disposed; the first one was disposed by the first step.
        Assert.Throws<ObjectDisposedException>(() => latents.AsSpan<float>());
    }

    [Fact]
    public void Integrate_ReturnsTheInputWhenNoStepHasWidth()
    {
        using CpuBackend backend = new();
        Tensor latents = Filled(1f);

        using Tensor result = XPartPipeline.Integrate(backend, latents, [1f, 1f], (x, t) => throw new InvalidOperationException("no forward expected"));

        Assert.Same(latents, result);
    }
}
