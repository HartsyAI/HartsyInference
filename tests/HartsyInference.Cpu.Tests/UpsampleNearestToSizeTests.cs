using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary>The interface default of <see cref="IBackend.UpsampleNearest2DToSize"/> (CpuBackend inherits it) against
/// the nearest-neighbour rule <c>interpolate(size=…, mode="nearest")</c> applies — source row <c>floor(oh·in/out)</c>
/// — at the one-short sizes a UNet up path produces for odd latent sizes, and at the exact double.</summary>
public sealed unsafe class UpsampleNearestToSizeTests
{
    [Theory]
    [InlineData(23, 40, 45, 80)]   // SDXL 1280×720: 45-row skip, 23-row activation
    [InlineData(12, 7, 23, 13)]
    [InlineData(6, 5, 12, 10)]     // exact double
    public void MatchesInterpolateNearest(int inH, int inW, int outH, int outW)
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(2, 3, inH, inW), DType.F32);
        for (long i = 0; i < input.ElementCount; i++) ((float*)input.DataPointer)[i] = i;
        using Tensor output = new(new TensorShape(2, 3, outH, outW), DType.F32);
        backend.UpsampleNearest2DToSize(output, input, 2);

        float* src = (float*)input.DataPointer, dst = (float*)output.DataPointer;
        for (int plane = 0; plane < 6; plane++)
            for (int y = 0; y < outH; y++)
                for (int x = 0; x < outW; x++)
                {
                    int sy = (int)Math.Floor((double)y * inH / outH), sx = (int)Math.Floor((double)x * inW / outW);
                    Assert.Equal(src[(plane * inH + sy) * inW + sx], dst[(plane * outH + y) * outW + x]);
                }
    }

    [Theory]
    [InlineData(47, 80)]   // overshoots the double
    [InlineData(44, 80)]   // drops a whole input row
    public void RejectsSizesOutsideTheOneShortRange(int outH, int outW)
    {
        IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(1, 1, 23, 40), DType.F32);
        using Tensor output = new(new TensorShape(1, 1, outH, outW), DType.F32);
        Assert.Throws<ArgumentException>(() => backend.UpsampleNearest2DToSize(output, input, 2));
    }
}
