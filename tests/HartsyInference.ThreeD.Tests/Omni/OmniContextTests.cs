using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

namespace HartsyInference.ThreeD.Tests.Omni;

public sealed class OmniContextTests
{
    private static Tensor Filled(long tokens, long width, float start)
    {
        Tensor t = new(new TensorShape(1, tokens, width), DType.F32);
        float v = start;
        foreach (ref float x in t.AsSpan<float>())
            x = v++;
        return t;
    }

    [Fact]
    public void Build_PutsControlAndSignalTokensBehindTheDinoTokensInBothBranchesAndZerosOnlyTheDinoPartOfTheUnconditionalOne()
    {
        using CpuBackend backend = new();
        using Tensor image = Filled(3, 4, 1), control = Filled(2, 4, 100);

        (Tensor cond, Tensor uncond) = OmniContext.Build(backend, image, control);
        using Tensor c = cond, u = uncond;

        Assert.Equal(new TensorShape(1, 5, 4), c.Shape);
        Assert.Equal(new TensorShape(1, 5, 4), u.Shape);
        float[] expected = [.. image.AsSpan<float>().ToArray(), .. control.AsSpan<float>().ToArray()];
        Assert.Equal(expected, c.AsSpan<float>().ToArray());
        float[] unconditional = u.AsSpan<float>().ToArray();
        Assert.All(unconditional[..12], v => Assert.Equal(0f, v));
        Assert.Equal(control.AsSpan<float>().ToArray(), unconditional[12..]);
    }

    [Fact]
    public void Build_RejectsMismatchedWidthsAndBatches()
    {
        using CpuBackend backend = new();
        using Tensor image = Filled(3, 4, 1), wide = Filled(2, 8, 0);
        using Tensor batched = new(new TensorShape(2, 3, 4), DType.F32);

        Assert.Throws<ArgumentException>(() => OmniContext.Build(backend, image, wide));
        Assert.Throws<ArgumentException>(() => OmniContext.Build(backend, batched, image));
        using Tensor half = new(image.Shape, DType.F16);
        Assert.Throws<ArgumentException>(() => OmniContext.Build(backend, half, image));
    }
}
