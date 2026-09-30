using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>A block-scaled weight splits by whole 128-row tiles of its swizzled scales — what a fused QKV split needs — and refuses anything finer.</summary>
public sealed class QuantWeightInfoBlockScaleSliceTests
{
    [Fact]
    public unsafe void TileAlignedSlice_KeepsEveryRowWithItsOwnScales()
    {
        const int rows = 384, paddedCols = 8;
        using Tensor scales = new(new TensorShape(rows, paddedCols), DType.U8);
        byte* p = (byte*)scales.DataPointer;
        for (int i = 0; i < rows * paddedCols; i++) p[i] = (byte)(i % 251);
        scales.Fp8ScaleFactor = 0.5f;
        QuantWeightInfo info = new() { Format = "mxfp8", BlockScale = scales };

        QuantWeightInfo k = info.SliceRows(128, 128, "attn.to_k");
        Assert.NotNull(k.BlockScale);
        Assert.Equal(new TensorShape(128, paddedCols), k.BlockScale!.Shape);
        Assert.Equal(0.5f, k.BlockScale.Fp8ScaleFactor);
        byte* q = (byte*)k.BlockScale.DataPointer;
        for (int r = 0; r < 128; r++)
            for (int c = 0; c < paddedCols; c++)
                Assert.Equal(p[Nvfp4ResidentCodec.SwizzledScaleIndex(128 + r, c, paddedCols)], q[Nvfp4ResidentCodec.SwizzledScaleIndex(r, c, paddedCols)]);
    }

    [Theory]
    [InlineData(64, 128)]
    [InlineData(128, 96)]
    public void SliceOffATile_Refuses(int offset, int count)
    {
        using Tensor scales = new(new TensorShape(256, 8), DType.U8);
        QuantWeightInfo info = new() { Format = "mxfp8", BlockScale = scales };
        Assert.Throws<NotSupportedException>(() => info.SliceRows(offset, count, "attn.to_qkv"));
    }
}
