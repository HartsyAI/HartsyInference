using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Nvfp4;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers <see cref="Nvfp4Codec.TryAttachResident"/>, the gate that decides whether an nvfp4 weight stays
/// packed at 0.5 byte/param or is eagerly dequantized. Both answers must be right for the wrong reason to be
/// impossible: an accept that mis-shapes the view runs every GEMM at half the true inner dimension, and a refusal
/// that should have been an accept only costs memory — but a refusal that DOESN'T happen (a
/// <c>pre_quant_scale</c> weight let through) silently drops a per-input-channel scale from the arithmetic.</summary>
public sealed unsafe class Nvfp4ResidentAttachTests
{
    private const int OutFeatures = 128;
    private const int InFeatures = 256;

    private static Tensor Packed(long rows, long packedCols, DType dtype) => new Tensor(new TensorShape(rows, packedCols), dtype);

    private static Tensor BlockScale(long rows, long cols)
    {
        Tensor scale = new Tensor(new TensorShape(rows, cols), DType.F8E4M3);
        byte* p = (byte*)scale.DataPointer;
        for (long i = 0; i < rows * cols; i++) p[i] = 0x40;   // E4M3 2.0
        return scale;
    }

    private static Tensor Global(float value)
    {
        Tensor global = new Tensor(new TensorShape(1), DType.F32);
        ((float*)global.DataPointer)[0] = value;
        return global;
    }

    [Fact]
    public void ValidWeight_IsRelabelledInPlaceWithItsCompanions()
    {
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.U8);
        using Tensor blockScale = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize);
        using Tensor global = Global(0.37f);
        long packedBytes = packed.DType.ComputeByteCount(packed.ElementCount);

        Assert.True(Nvfp4Codec.TryAttachResident(packed, blockScale, global, hasPreQuantScale: false, out Tensor resident));
        using Tensor owned = resident;

        Assert.Equal(DType.F4E2M1, owned.DType);
        Assert.Equal(2, owned.Shape.Rank);
        Assert.Equal((long)OutFeatures, owned.Shape[0]);
        Assert.Equal((long)InFeatures, owned.Shape[1]);
        Assert.Equal(packedBytes, owned.DType.ComputeByteCount(owned.ElementCount));
        Assert.True(packed.DataPointer == owned.DataPointer, "the resident view must borrow the packed bytes, not copy them.");
        Assert.NotNull(owned.QuantInfo);
        Assert.Equal("nvfp4", owned.QuantInfo!.Format);
        Assert.Same(blockScale, owned.QuantInfo.BlockScale);
        Assert.Same(global, owned.QuantInfo.GlobalScale);
    }

    [Fact]
    public void PreQuantScale_IsRefused()
    {
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.U8);
        using Tensor blockScale = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize);
        using Tensor global = Global(1f);
        AssertRefused(packed, blockScale, global, hasPreQuantScale: true);
    }

    [Fact]
    public void NonU8Weight_IsRefused()
    {
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.I8);
        using Tensor blockScale = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize);
        using Tensor global = Global(1f);
        AssertRefused(packed, blockScale, global, hasPreQuantScale: false);
    }

    [Fact]
    public void Rank3ExpertBank_IsRefused()
    {
        using Tensor packed = new Tensor(new TensorShape(2, OutFeatures, InFeatures / 2), DType.U8);
        using Tensor blockScale = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize);
        using Tensor global = Global(1f);
        AssertRefused(packed, blockScale, global, hasPreQuantScale: false);
    }

    [Fact]
    public void NonE4M3BlockScale_IsRefused()
    {
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.U8);
        using Tensor blockScale = new Tensor(new TensorShape(OutFeatures, InFeatures / Nvfp4Codec.GroupSize), DType.F16);
        using Tensor global = Global(1f);
        AssertRefused(packed, blockScale, global, hasPreQuantScale: false);
    }

    [Fact]
    public void BlockScaleTooSmall_IsRefused()
    {
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.U8);
        using Tensor tooFewRows = BlockScale(OutFeatures / 2, InFeatures / Nvfp4Codec.GroupSize);
        using Tensor tooFewCols = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize - 4);
        using Tensor global = Global(1f);
        AssertRefused(packed, tooFewRows, global, hasPreQuantScale: false);
        AssertRefused(packed, tooFewCols, global, hasPreQuantScale: false);
    }

    [Fact]
    public void BlockScaleColumnsNotMultipleOfFour_IsRefused()
    {
        // The swizzle divides the stored last dim by 4; a stride that is not a multiple of 4 would make every
        // scale lookup address the wrong element rather than fail loudly.
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.U8);
        using Tensor blockScale = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize + 1);
        using Tensor global = Global(1f);
        AssertRefused(packed, blockScale, global, hasPreQuantScale: false);
    }

    [Fact]
    public void NonScalarGlobalScale_IsRefused()
    {
        using Tensor packed = Packed(OutFeatures, InFeatures / 2, DType.U8);
        using Tensor blockScale = BlockScale(OutFeatures, InFeatures / Nvfp4Codec.GroupSize);
        using Tensor perExpert = new Tensor(new TensorShape(4), DType.F32);
        AssertRefused(packed, blockScale, perExpert, hasPreQuantScale: false);
    }

    private static void AssertRefused(Tensor packed, Tensor blockScale, Tensor globalScale, bool hasPreQuantScale)
    {
        Assert.False(Nvfp4Codec.TryAttachResident(packed, blockScale, globalScale, hasPreQuantScale, out Tensor resident));
        Assert.Same(packed, resident);
        Assert.Null(resident.QuantInfo);
    }
}
