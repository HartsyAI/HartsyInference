using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

public sealed class Pixal3DDeviceMemoryTests
{
    [Fact]
    public void ReleaseAll_OnTheCpuBackendLeavesTheConditioningUnchanged()
    {
        Tensor global = new(new TensorShape(1, 5, 4), DType.F32), proj = new(new TensorShape(1, 3, 6), DType.F32);
        for (int i = 0; i < global.ElementCount; i++)
            global.AsSpan<float>()[i] = i + 1;
        for (int i = 0; i < proj.ElementCount; i++)
            proj.AsSpan<float>()[i] = -i - 0.5f;
        using Pixal3DCondition cond = new(global, proj);
        using Pixal3DCondition negative = cond.ZerosLike();
        using CpuBackend backend = new();

        Pixal3DDeviceMemory.ReleaseAll(backend, cond, negative);

        Assert.Equal(1f, cond.Global.AsSpan<float>()[0]);
        Assert.Equal(20f, cond.Global.AsSpan<float>()[19]);
        Assert.Equal(-0.5f, cond.Proj.AsSpan<float>()[0]);
        Assert.Equal(-17.5f, cond.Proj.AsSpan<float>()[17]);
        Assert.All(negative.Proj.AsSpan<float>().ToArray(), x => Assert.Equal(0f, x));
    }

    [Fact]
    public void ReleaseAll_RejectsNullArguments()
    {
        using CpuBackend backend = new();
        Assert.Throws<ArgumentNullException>(() => Pixal3DDeviceMemory.ReleaseAll(null!, []));
        Assert.Throws<ArgumentNullException>(() => Pixal3DDeviceMemory.ReleaseAll(backend, null!));
    }
}
