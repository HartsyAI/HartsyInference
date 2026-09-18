using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Guards the rule that quantization companions follow the bytes. A <see cref="Tensor"/> built over another's
/// memory — a reshape, a dtype relabel, a same-device copy — used to start life with a default
/// <see cref="Tensor.Fp8ScaleFactor"/> of 1.0 and no <see cref="Tensor.QuantInfo"/>, so a converter that reshaped a
/// quantized weight handed the backend raw bytes whose real values are <c>stored · scale</c>. The output is not an
/// error, it is a weight hundreds of times too large, and that is only visible as noise at the end of a generation.</summary>
public sealed unsafe class TensorViewQuantMetadataTests
{
    private static Tensor Fp8Weight(long rows, long columns, float scale = 0.0195f, float inputScale = 0.0625f)
    {
        Tensor weight = new Tensor(new TensorShape(rows, columns), DType.F8E4M3)
        {
            Fp8ScaleFactor = scale,
            Fp8InputScaleFactor = inputScale,
        };
        return weight;
    }

    private static QuantWeightInfo Int8Info(Tensor rowScale, int convRotGroupSize = 256) => new()
    {
        Format = "int8_tensorwise",
        RowScale = rowScale,
        ConvRotGroupSize = convRotGroupSize,
    };

    private static Tensor RowScale(long rows)
    {
        Tensor scale = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> values = scale.AsSpan<float>();
        for (int i = 0; i < rows; i++) values[i] = 0.5f + i;
        return scale;
    }

    [Fact]
    public void Reshape_CarriesBothFp8ScalarsAndRowScales()
    {
        using Tensor weight = Fp8Weight(4, 256);
        using Tensor rowScale = RowScale(4);
        weight.QuantInfo = Int8Info(rowScale);

        using Tensor view = weight.Reshape(new TensorShape(4, 16, 16));

        Assert.Equal(0.0195f, view.Fp8ScaleFactor);
        Assert.Equal(0.0625f, view.Fp8InputScaleFactor);
        Assert.Same(weight.QuantInfo, view.QuantInfo);
    }

    [Fact]
    public void Reshape_ThatRenumbersRows_RefusesAQuantizedWeight()
    {
        using Tensor weight = Fp8Weight(4, 256);
        using Tensor rowScale = RowScale(4);
        weight.QuantInfo = Int8Info(rowScale);

        // The dims-swap RelabelRank2ToPyTorchOrder performs: legitimate for a GGUF block-quantized tensor, which
        // carries no per-row companion, and never correct for one that does.
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => weight.Reshape(new TensorShape(256, 4)));
        Assert.Contains("int8_tensorwise", error.Message);
    }

    [Fact]
    public void Reshape_ThatRenumbersRows_IsFineWithoutPerRowCompanions()
    {
        using Tensor weight = new Tensor(new TensorShape(4, 256), DType.Q4_K);

        using Tensor view = weight.Reshape(new TensorShape(256, 4));

        Assert.Equal(1.0f, view.Fp8ScaleFactor);
        Assert.Null(view.QuantInfo);
    }

    [Fact]
    public void ReinterpretAs_CarriesCompanionsOntoTheRelabelledView()
    {
        using Tensor packed = new Tensor(new TensorShape(6, 8), DType.U8) { Fp8ScaleFactor = 0.25f };

        using Tensor view = packed.ReinterpretAs(DType.F4E2M1, new TensorShape(6, 16));

        Assert.Equal(0.25f, view.Fp8ScaleFactor);
    }

    [Fact]
    public void To_CarriesTheInputScaleAlongWithTheWeightScale()
    {
        using Tensor weight = Fp8Weight(4, 256);
        using Tensor rowScale = RowScale(4);
        weight.QuantInfo = Int8Info(rowScale);

        using Tensor copy = weight.To(DeviceKind.Cpu);

        Assert.Equal(0.0195f, copy.Fp8ScaleFactor);
        // Dropping this silently switched the Linear from a constant activation scale back to a per-call absmax.
        Assert.Equal(0.0625f, copy.Fp8InputScaleFactor);
        Assert.Same(weight.QuantInfo, copy.QuantInfo);
    }

    [Fact]
    public void SliceRows_ViewsTheRequestedRowsAndCarriesThePerTensorScalars()
    {
        using Tensor weight = new Tensor(new TensorShape(6, 4), DType.U8) { Fp8ScaleFactor = 0.5f, Fp8InputScaleFactor = 2f };
        Span<byte> bytes = weight.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)i;

        using Tensor slice = weight.SliceRows(2, 3);

        Assert.Equal(3L, slice.Shape[0]);
        Assert.Equal(4L, slice.Shape[1]);
        Assert.Equal(0.5f, slice.Fp8ScaleFactor);
        Assert.Equal(2f, slice.Fp8InputScaleFactor);
        Assert.True((byte*)weight.DataPointer + 8 == (byte*)slice.DataPointer, "SliceRows copied instead of viewing.");
        ReadOnlySpan<byte> seen = slice.AsReadOnlySpan<byte>();
        for (int i = 0; i < 12; i++) Assert.Equal((byte)(8 + i), seen[i]);
    }

    [Fact]
    public void SliceRows_OutOfRange_Throws()
    {
        using Tensor weight = new Tensor(new TensorShape(6, 4), DType.U8);
        Assert.Throws<HartsyInferenceException>(() => weight.SliceRows(4, 3));
    }

    [Fact]
    public void SliceRows_OfABlockQuantWhoseRowIsNotAWholeNumberOfBlocks_Throws()
    {
        // Q4_K packs 256 elements per block, so a 128-wide row would put the slice offset mid-block.
        using Tensor weight = new Tensor(new TensorShape(4, 128), DType.Q4_K);
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => weight.SliceRows(1, 2));
        Assert.Contains("quant block", error.Message);
    }

    [Fact]
    public void QuantInfoSliceRows_NarrowsAPerRowScaleAndSharesAPerTensorOne()
    {
        using Tensor perRow = RowScale(6);
        QuantWeightInfo sliced = Int8Info(perRow).SliceRows(2, 3, "blocks.0.attn.to_k.weight");
        Assert.NotSame(perRow, sliced.RowScale);
        Assert.Equal(3L, sliced.RowScale!.Shape[0]);
        Assert.Equal(2.5f, sliced.RowScale.AsReadOnlySpan<float>()[0]);
        Assert.Equal(256, sliced.ConvRotGroupSize);

        using Tensor perTensor = new Tensor(new TensorShape(1), DType.F32);
        QuantWeightInfo shared = Int8Info(perTensor).SliceRows(2, 3, "blocks.0.attn.to_k.weight");
        Assert.Same(perTensor, shared.RowScale);
    }

    private static LowRankAdjunct Adjunct(long outFeatures, long inFeatures, long rank, float scale = 0.75f)
    {
        Tensor down = new(new TensorShape(rank, inFeatures), DType.F32);
        Tensor up = new(new TensorShape(outFeatures, rank), DType.F32);
        Span<float> upValues = up.AsSpan<float>();
        for (int i = 0; i < upValues.Length; i++) upValues[i] = i;
        return new LowRankAdjunct { Terms = [new LowRankAdjunctTerm { Down = down, Up = up, Scale = scale }] };
    }

    [Fact]
    public void WithLowRankAdjunct_LeavesTheBaseAloneAndAliasesItsBytes()
    {
        // The whole point: the base object is what caches, resident models and the identity-keyed device cache
        // hold, so a LoRA written onto it would follow into the next request that reuses the entry.
        using Tensor weight = new Tensor(new TensorShape(4, 256), DType.Q4_K) { Fp8ScaleFactor = 0.5f };
        LowRankAdjunct adjunct = Adjunct(4, 256, 2);

        using Tensor patched = weight.WithLowRankAdjunct(adjunct);

        Assert.Null(weight.LowRankAdjunct);
        Assert.Same(adjunct, patched.LowRankAdjunct);
        Assert.NotSame(weight, patched);
        Assert.True(weight.DataPointer == patched.DataPointer, "WithLowRankAdjunct copied instead of aliasing.");
        Assert.Equal(weight.Shape, patched.Shape);
        Assert.Equal(DType.Q4_K, patched.DType);
        Assert.Equal(0.5f, patched.Fp8ScaleFactor);
    }

    [Fact]
    public void Reshape_CarriesTheAdjunctWhenRowsAndColumnsSurvive()
    {
        using Tensor weight = new Tensor(new TensorShape(4, 256), DType.Q4_K);
        using Tensor patched = weight.WithLowRankAdjunct(Adjunct(4, 256, 2));

        using Tensor view = patched.Reshape(new TensorShape(4, 16, 16));

        Assert.Same(patched.LowRankAdjunct, view.LowRankAdjunct);
    }

    [Fact]
    public void Reshape_ThatRenumbersRows_RefusesAnAdjunctWeight()
    {
        using Tensor weight = new Tensor(new TensorShape(4, 256), DType.Q4_K);
        using Tensor patched = weight.WithLowRankAdjunct(Adjunct(4, 256, 2));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => patched.Reshape(new TensorShape(256, 4)));
        Assert.Contains("LoRA-adjunct", error.Message);
    }

    [Fact]
    public void ReinterpretAs_ThatChangesTheInputWidth_RefusesAnAdjunctWeight()
    {
        // nvfp4 relabels U8 [N, K/2] as F4E2M1 [N, K]: rows survive, the inner dimension does not, and the
        // adjunct's down matrix is indexed by exactly that dimension.
        using Tensor packed = new Tensor(new TensorShape(6, 8), DType.U8);
        using Tensor patched = packed.WithLowRankAdjunct(Adjunct(6, 8, 2));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => patched.ReinterpretAs(DType.F4E2M1, new TensorShape(6, 16)));
        Assert.Contains("LoRA-adjunct", error.Message);
    }

    [Fact]
    public void SliceRows_LeavesTheAdjunctToTheCaller()
    {
        // Same rule QuantInfo follows: only the caller knows the window, and a silently carried whole-weight
        // adjunct would add the wrong rows' delta to a chunked projection.
        using Tensor weight = new Tensor(new TensorShape(6, 256), DType.Q4_K);
        using Tensor patched = weight.WithLowRankAdjunct(Adjunct(6, 256, 2));

        using Tensor slice = patched.SliceRows(2, 3);

        Assert.Null(slice.LowRankAdjunct);
    }

    [Fact]
    public void AdjunctSliceRows_NarrowsTheUpMatrixAndSharesTheDownMatrix()
    {
        LowRankAdjunct adjunct = Adjunct(6, 256, 2);
        LowRankAdjunctTerm whole = adjunct.Terms[0];

        LowRankAdjunct window = adjunct.SliceRows(2, 3);
        LowRankAdjunctTerm sliced = window.Terms[0];

        Assert.Same(whole.Down, sliced.Down);
        Assert.Equal(3L, sliced.Up!.Shape[0]);
        Assert.Equal(2L, sliced.Up.Shape[1]);
        // Row 2 of a [6, 2] up matrix filled with its flat index starts at 4.
        Assert.Equal(4f, sliced.Up.AsReadOnlySpan<float>()[0]);
        Assert.Equal(whole.Scale, sliced.Scale);
        // Memoized: the device weight cache is keyed by tensor identity, so the same window must be the same object.
        Assert.Same(window, adjunct.SliceRows(2, 3));
    }

    [Fact]
    public void AdjunctExpandWeights_YieldsEveryFactorTheGemmWillRead()
    {
        using Tensor weight = new Tensor(new TensorShape(6, 256), DType.Q4_K);
        LowRankAdjunct adjunct = Adjunct(6, 256, 2);
        using Tensor patched = weight.WithLowRankAdjunct(adjunct);
        LowRankAdjunct window = adjunct.SliceRows(0, 3);

        List<Tensor> expanded = [.. LowRankAdjunct.ExpandWeights([patched])];

        Assert.Contains(patched, expanded);
        Assert.Contains(adjunct.Terms[0].Down, expanded);
        Assert.Contains(adjunct.Terms[0].Up!, expanded);
        // The windows a chunked projection created must be freed with the weight too, not leaked on the device.
        Assert.Contains(window.Terms[0].Up!, expanded);
    }

    [Fact]
    public void QuantInfoSliceRows_RefusesNvfp4BlockScales()
    {
        using Tensor blockScale = new Tensor(new TensorShape(128, 4), DType.F8E4M3);
        QuantWeightInfo info = new() { Format = "nvfp4", BlockScale = blockScale };

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => info.SliceRows(0, 4, "blocks.0.attn.qkv_proj.weight"));
        Assert.Contains("blocks.0.attn.qkv_proj.weight", error.Message);
    }
}
