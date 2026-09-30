using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Mxfp8;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers <see cref="Mxfp8Codec.TryAttachResident"/>, the gate that keeps an MXFP8 weight packed at one byte per parameter with its block scales on <see cref="Tensor.QuantInfo"/>. An accept that pairs a weight with scales that do not describe it multiplies every block by another block's scale — plausible output, silently wrong — so every shape refusal is pinned, and the host decode the packed form is measured against is checked once more here.</summary>
public sealed unsafe class Mxfp8ResidentAttachTests
{
    private const int OutFeatures = 128;
    private const int InFeatures = 256;   // 8 blocks of 32 → 2 block-column groups of 4

    private static Tensor Weight(long rows, long cols, DType dtype) => new Tensor(new TensorShape(rows, cols), dtype);

    private static Tensor Scale(long rows, long cols, byte e8m0 = 127)
    {
        Tensor scale = new Tensor(new TensorShape(rows, cols), DType.U8);
        new Span<byte>((void*)scale.DataPointer, (int)(rows * cols)).Fill(e8m0);
        return scale;
    }

    [Fact]
    public void ValidWeight_KeepsItsBytesAndCarriesTheScales()
    {
        using Tensor weight = Weight(OutFeatures, InFeatures, DType.F8E4M3);
        using Tensor scale = Scale(OutFeatures, InFeatures / Mxfp8ResidentCodec.GroupSize);
        Assert.True(Mxfp8Codec.TryAttachResident(weight, scale));
        Assert.Equal(DType.F8E4M3, weight.DType);
        Assert.Equal("mxfp8", weight.QuantInfo?.Format);
        Assert.Same(scale, weight.QuantInfo!.BlockScale);
        Assert.Null(weight.QuantInfo.GlobalScale);
    }

    [Theory]
    [InlineData("F16 weight", "dtype")]
    [InlineData("F32 scale", "scaledtype")]
    [InlineData("rank-3 weight", "rank")]
    [InlineData("K not a multiple of 32", "group")]
    [InlineData("scale narrower than K/32", "narrow")]
    [InlineData("scale columns not a multiple of 4", "cols")]
    [InlineData("scale shorter than N", "rows")]
    public void WrongShapesAreRefusedUntouched(string _, string kind)
    {
        using Tensor weight = kind switch
        {
            "dtype" => Weight(OutFeatures, InFeatures, DType.F16),
            "rank" => new Tensor(new TensorShape(2, OutFeatures, InFeatures), DType.F8E4M3),
            "group" => Weight(OutFeatures, InFeatures + 8, DType.F8E4M3),
            _ => Weight(OutFeatures, InFeatures, DType.F8E4M3),
        };
        using Tensor scale = kind switch
        {
            "scaledtype" => new Tensor(new TensorShape(OutFeatures, InFeatures / 32), DType.F32),
            "narrow" => Scale(OutFeatures, 4),
            "cols" => Scale(OutFeatures, InFeatures / 32 + 1),
            "rows" => Scale(OutFeatures - 1, InFeatures / 32),
            _ => Scale(OutFeatures, InFeatures / 32),
        };
        Assert.False(Mxfp8Codec.TryAttachResident(weight, scale));
        Assert.Null(weight.QuantInfo);
    }

    /// <summary>The host decode: each block's E8M0 exponent scales its 32 elements, read through the blocked layout.</summary>
    [Fact]
    public void HostDecodeAppliesEachBlockItsOwnExponent()
    {
        using Tensor f32 = new Tensor(new TensorShape(OutFeatures, InFeatures), DType.F32);
        new Span<float>((void*)f32.DataPointer, OutFeatures * InFeatures).Fill(1.5f);   // exact in E4M3
        using Tensor weight = f32.CastTo(DType.F8E4M3);
        using Tensor scale = Scale(OutFeatures, InFeatures / 32, 0);
        byte* s = (byte*)scale.DataPointer;
        for (int r = 0; r < OutFeatures; r++)
            for (int b = 0; b < InFeatures / 32; b++)
                s[BlockScaleSwizzle.SwizzledIndex(r, b, InFeatures / 32)] = (byte)(127 + b - 3);   // 2^-3 .. 2^4
        using Tensor bf16 = Mxfp8ResidentCodec.DequantToBf16(weight, scale);
        using Tensor back = bf16.CastTo(DType.F32);
        float* o = (float*)back.DataPointer;
        for (int r = 0; r < OutFeatures; r += 37)
            for (int b = 0; b < InFeatures / 32; b++)
                Assert.Equal(1.5f * MathF.Pow(2f, b - 3), o[r * InFeatures + b * 32 + 17]);
    }
}
