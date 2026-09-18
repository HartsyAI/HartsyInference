using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Covers the row window YuE2's AR stack carves out of <c>lm_head</c> for the semantic pass on a quantized
/// checkpoint.</summary>
/// <remarks>The window is a row range, and every packed format indexes its dequant scale by exactly those rows — so a
/// byte copy that leaves the scale behind produces a head of raw int8, and a copy sized from
/// <c>DType.SizeInBytes</c> (0 for every block quant) produces a head of zeros. Both read as the model having learned
/// nothing about pitch; neither fails at load.</remarks>
public sealed unsafe class Yue2SemanticHeadSliceTests
{
    private const int Rows = 16;
    private const int Columns = 32;
    private const int Start = 4;
    private const int Count = 8;

    private static Tensor Int8Head(int group, out Tensor rowScale)
    {
        Tensor head = new Tensor(new TensorShape(Rows, Columns), DType.I8);
        Span<sbyte> bytes = head.AsSpan<sbyte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (sbyte)((i * 37) % 251 - 125);
        rowScale = new Tensor(new TensorShape(Rows, 1), DType.F32);
        Span<float> scales = rowScale.AsSpan<float>();
        for (int i = 0; i < Rows; i++) scales[i] = 0.002f * (i + 1);
        head.QuantInfo = new QuantWeightInfo
        {
            Format = "int8_tensorwise",
            RowScale = rowScale,
            ConvRotGroupSize = group,
        };
        return head;
    }

    private static Tensor Dense()
    {
        Tensor dense = new Tensor(new TensorShape(Rows, Columns), DType.F32);
        Span<float> values = dense.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = 0.01f * ((i * 13) % 41 - 20);
        return dense;
    }

    [Fact]
    public void SliceHeadWindow_DecodesAnInt8ConvRotWindowWithItsOwnRowScales()
    {
        using Tensor head = Int8Head(group: 16, out Tensor rowScale);
        using Tensor scale = rowScale;
        using Tensor window = Yue2ArLm.SliceHeadWindow(head, Start, Count);
        Assert.Equal(DType.BF16, window.DType);
        Assert.Equal(Count, window.Shape[0]);
        Assert.Equal(Columns, window.Shape[1]);

        using Tensor whole = Int8ConvRotCodec.DequantToBf16(head, rowScale, 16);
        ReadOnlySpan<ushort> expected = whole.AsReadOnlySpan<ushort>();
        ReadOnlySpan<ushort> actual = window.AsReadOnlySpan<ushort>();
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[Start * Columns + i], actual[i]);
    }

    [Fact]
    public void SliceHeadWindow_DecodesABlockQuantizedWindow()
    {
        using Tensor dense = Dense();
        using Tensor head = GgufQuantizer.Quantize(dense, DType.Q8_0);
        using Tensor window = Yue2ArLm.SliceHeadWindow(head, Start, Count);
        Assert.Equal(DType.F16, window.DType);

        using Tensor whole = GgufDequantizer.Dequantize(head, DType.F16);
        ReadOnlySpan<Half> expected = whole.AsReadOnlySpan<Half>();
        ReadOnlySpan<Half> actual = window.AsReadOnlySpan<Half>();
        Assert.Equal(Count * Columns, actual.Length);
        bool anyNonZero = false;
        for (int i = 0; i < actual.Length; i++)
        {
            Assert.Equal(expected[Start * Columns + i], actual[i]);
            anyNonZero |= (float)actual[i] != 0f;
        }
        Assert.True(anyNonZero, "a window of zeros would satisfy every other assertion here");
    }

    [Fact]
    public void SliceHeadWindow_CopiesADenseWindowUnchanged()
    {
        using Tensor head = Dense();
        using Tensor window = Yue2ArLm.SliceHeadWindow(head, Start, Count);
        Assert.Equal(DType.F32, window.DType);
        ReadOnlySpan<float> expected = head.AsReadOnlySpan<float>();
        ReadOnlySpan<float> actual = window.AsReadOnlySpan<float>();
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[Start * Columns + i], actual[i]);
    }
}
