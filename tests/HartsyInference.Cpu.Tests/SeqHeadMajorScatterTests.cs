using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary>The host reference of <see cref="IBackend.ScatterSeqHeadMajor(Tensor, Tensor, int, int)"/>, which the
/// GPU backends are held to by <c>CrossBackendOpParityTests</c>: a prefix of each head's rows, between head-major
/// tensors whose sequence extents differ, as a byte copy — the KV cache resize's layout. A wrong per-head stride on
/// either side lands rows in the wrong head without failing anything else.</summary>
public sealed class SeqHeadMajorScatterTests
{
    /// <summary>Each element encodes its own (head, row, column), offset by <paramref name="tag"/>.</summary>
    private static Tensor Coded(int heads, int seq, int hd, float tag)
    {
        Tensor t = new(new TensorShape(1, heads, seq, hd), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int h = 0; h < heads; h++)
        {
            for (int s = 0; s < seq; s++)
            {
                for (int d = 0; d < hd; d++)
                {
                    values[(h * seq + s) * hd + d] = tag + h * 10_000 + s * 100 + d;
                }
            }
        }
        return t;
    }

    [Theory]
    [InlineData(6, 1, 4)]    // output shorter than the input: a shrink, at an offset
    [InlineData(14, 3, 9)]   // output longer than the input: a grow, every input row
    [InlineData(14, 0, 0)]   // nothing to copy
    public void A_Row_Prefix_Lands_At_The_Offset_Of_Every_Head(int outputSeq, int seqOffset, int rows)
    {
        const int heads = 3, inputSeq = 9, hd = 4;
        using IBackend backend = new CpuBackend();
        using Tensor input = Coded(heads, inputSeq, hd, tag: 0.5f);
        using Tensor output = Coded(heads, outputSeq, hd, tag: -1_000_000f);

        backend.ScatterSeqHeadMajor(output, input, seqOffset, rows);

        ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();
        for (int h = 0; h < heads; h++)
        {
            for (int s = 0; s < outputSeq; s++)
            {
                for (int d = 0; d < hd; d++)
                {
                    bool copied = s >= seqOffset && s < seqOffset + rows;
                    float expected = copied ? 0.5f + h * 10_000 + (s - seqOffset) * 100 + d : -1_000_000f + h * 10_000 + s * 100 + d;
                    Assert.Equal(expected, actual[(h * outputSeq + s) * hd + d]);
                }
            }
        }
    }

    [Fact]
    public void Half_Precision_Rows_Are_Copied_Bit_For_Bit()
    {
        using IBackend backend = new CpuBackend();
        using Tensor input = new(new TensorShape(1, 2, 3, 2), DType.F16);
        using Tensor output = new(new TensorShape(1, 2, 5, 2), DType.F16);
        Span<ushort> source = input.AsSpan<ushort>();
        for (int i = 0; i < source.Length; i++)
        {
            source[i] = (ushort)(0x7C01 + i);   // NaN payloads: a float round trip would canonicalize them
        }

        backend.ScatterSeqHeadMajor(output, input, seqOffset: 2, rows: 2);

        ReadOnlySpan<ushort> written = output.AsReadOnlySpan<ushort>();
        for (int h = 0; h < 2; h++)
        {
            for (int i = 0; i < 2 * 2; i++)
            {
                Assert.Equal(source[h * 3 * 2 + i], written[(h * 5 + 2) * 2 + i]);
            }
        }
    }

    [Fact]
    public void The_Three_Argument_Form_Copies_Every_Input_Row()
    {
        using IBackend backend = new CpuBackend();
        using Tensor input = Coded(heads: 2, seq: 3, hd: 2, tag: 7f);
        using Tensor viaAll = Coded(heads: 2, seq: 8, hd: 2, tag: -5f);
        using Tensor viaRows = Coded(heads: 2, seq: 8, hd: 2, tag: -5f);

        backend.ScatterSeqHeadMajor(viaAll, input, 4);
        backend.ScatterSeqHeadMajor(viaRows, input, 4, rows: 3);

        Assert.True(viaAll.AsReadOnlySpan<float>().SequenceEqual(viaRows.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void Out_Of_Range_Or_Mismatched_Arguments_Are_Refused()
    {
        using IBackend backend = new CpuBackend();
        using Tensor input = Coded(heads: 2, seq: 4, hd: 2, tag: 0f);
        using Tensor output = Coded(heads: 2, seq: 6, hd: 2, tag: 0f);
        using Tensor otherHeads = Coded(heads: 3, seq: 6, hd: 2, tag: 0f);
        using Tensor half = new(new TensorShape(1, 2, 6, 2), DType.F16);

        Assert.Throws<ArgumentOutOfRangeException>(() => backend.ScatterSeqHeadMajor(output, input, 0, rows: 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.ScatterSeqHeadMajor(output, input, 3, rows: 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.ScatterSeqHeadMajor(output, input, -1, rows: 1));
        Assert.Throws<ArgumentException>(() => backend.ScatterSeqHeadMajor(otherHeads, input, 0, rows: 1));
        Assert.Throws<ArgumentException>(() => backend.ScatterSeqHeadMajor(half, input, 0, rows: 1));
    }
}
