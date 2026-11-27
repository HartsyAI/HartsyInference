using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The shared stream window owner: parts concatenate in order along an axis, slices and empty parts behave, and an upload refuses a
/// wrong count.</summary>
public sealed class DuoDeviceWindowTests
{
    [Fact]
    public void ConcatJoinsPartsInOrderOnTheChosenAxis()
    {
        // [1, 2, n, 3] with n steps; the window joins three parts of 1, 2 and 1 steps on axis 2.
        Tensor a = Filled(1f, 1, 2, 1, 3), b = Filled(2f, 1, 2, 2, 3), c = Filled(3f, 1, 2, 1, 3);
        using Tensor window = DuoDeviceWindow.Concat(TestBackends.Cpu, 2,
        [
            new DuoDeviceWindow.Part(a, 0, 1),
            new DuoDeviceWindow.Part(b, 0, 2),
            new DuoDeviceWindow.Part(c, 0, 1),
        ]);
        a.Dispose();
        b.Dispose();
        c.Dispose();

        Assert.Equal(new TensorShape(1, 2, 4, 3), window.Shape);
        float[] values = window.AsSpan<float>().ToArray();
        for (int head = 0; head < 2; head++)
        {
            float[] expected = [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3];
            for (int i = 0; i < 12; i++)
            {
                Assert.Equal(expected[i], values[head * 12 + i]);
            }
        }
    }

    [Fact]
    public void EmptyPartsAreSkippedAndAllEmptyIsRefused()
    {
        Tensor a = Filled(5f, 1, 2, 2, 3);
        using Tensor window = DuoDeviceWindow.Concat(TestBackends.Cpu, 2, [new DuoDeviceWindow.Part(a, 0, 0), new DuoDeviceWindow.Part(a, 0, 2)]);
        Assert.Equal(new TensorShape(1, 2, 2, 3), window.Shape);
        Assert.Throws<ArgumentException>(() => DuoDeviceWindow.Concat(TestBackends.Cpu, 2, [new DuoDeviceWindow.Part(a, 0, 0)]));
        a.Dispose();
    }

    [Fact]
    public void SliceTakesTheStepsFromTheStart()
    {
        Tensor source = Filled(0f, 1, 2, 4, 1);
        float[] ramp = [0, 1, 2, 3, 10, 11, 12, 13];
        ramp.AsSpan().CopyTo(source.AsSpan<float>());
        using Tensor piece = DuoDeviceWindow.Slice(TestBackends.Cpu, source, 2, 1, 2);
        source.Dispose();

        Assert.Equal(new TensorShape(1, 2, 2, 1), piece.Shape);
        Assert.Equal([1f, 2f, 11f, 12f], piece.AsSpan<float>().ToArray());
    }

    [Fact]
    public void AnUploadOfTheWrongCountIsRefused()
    {
        Assert.Throws<ArgumentException>(() => DuoDeviceWindow.Upload([2, 3], new float[5]));
    }

    private static Tensor Filled(float value, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }
}
