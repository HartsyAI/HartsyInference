using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the uneven fused-projection split every image converter now routes through. The quantized path is
/// what these pin: <c>DType.SizeInBytes</c> is 0 for every block quant, so byte math derived from it copies nothing at
/// all while the dense path stays byte-perfect, and a per-row <c>int8_tensorwise</c> scale that is not narrowed with
/// the rows leaves every piece but the first running at another layer's dynamic range.</summary>
public sealed unsafe class SplitRowsTests
{
    /// <summary>A Q8_0 weight of <paramref name="rows"/> × <paramref name="columns"/>, each 32-value block an F16 scale plus 32 codes.</summary>
    private static Tensor Q8Weight(int rows, int columns)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, columns), DType.Q8_0);
        Span<byte> bytes = tensor.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i % 251 + 1);
        return tensor;
    }

    private static Tensor Int8Weight(int rows, int columns)
    {
        Tensor weight = new Tensor(new TensorShape(rows, columns), DType.I8);
        Span<sbyte> values = weight.AsSpan<sbyte>();
        for (int i = 0; i < values.Length; i++) values[i] = (sbyte)(i % 127);
        return weight;
    }

    private static Tensor RowScale(int rows)
    {
        Tensor scale = new Tensor(new TensorShape(rows, 1), DType.F32);
        Span<float> values = scale.AsSpan<float>();
        for (int i = 0; i < rows; i++) values[i] = 0.001f * (i + 1);
        return scale;
    }

    private static void DisposeAll(Tensor[] pieces)
    {
        foreach (Tensor piece in pieces) piece.Dispose();
    }

    [Fact]
    public void SplitRows_CopiesEveryBlockQuantByteOfEachPiece()
    {
        using Tensor fused = Q8Weight(rows: 8, columns: 64);
        ReadOnlySpan<byte> source = fused.AsReadOnlySpan<byte>();
        // Two blocks per row × 34 bytes: the size DType.SizeInBytes reports as 0.
        const int rowBytes = 2 * 34;

        Tensor[] pieces = CheckpointConvertUtils.SplitRows(fused, [2, 2, 4],
            ["blocks.0.attn.to_q.weight", "blocks.0.attn.to_k.weight", "blocks.0.proj_mlp.weight"]);
        try
        {
            Assert.Equal(3, pieces.Length);
            int offset = 0;
            foreach ((Tensor piece, int rows) in new[] { (pieces[0], 2), (pieces[1], 2), (pieces[2], 4) })
            {
                Assert.Equal(DType.Q8_0, piece.DType);
                Assert.Equal(rows, (int)piece.Shape[0]);
                Assert.Equal(64, (int)piece.Shape[1]);
                ReadOnlySpan<byte> copied = piece.AsReadOnlySpan<byte>();
                Assert.Equal(rows * rowBytes, copied.Length);
                Assert.True(copied.SequenceEqual(source.Slice(offset, rows * rowBytes)));
                offset += rows * rowBytes;
            }
            Assert.Equal(source.Length, offset);
        }
        finally
        {
            DisposeAll(pieces);
        }
    }

    [Fact]
    public void SplitRows_GivesEachPieceItsOwnRowScales()
    {
        using Tensor fused = Int8Weight(rows: 8, columns: 256);
        using Tensor rowScale = RowScale(8);
        fused.QuantInfo = new QuantWeightInfo
        {
            Format = "int8_tensorwise",
            RowScale = rowScale,
            ConvRotGroupSize = 256,
        };

        Tensor[] pieces = CheckpointConvertUtils.SplitRows(fused, [3, 1, 4], ["q", "k", "mlp"]);
        try
        {
            int firstRow = 0;
            foreach ((Tensor piece, int rows) in new[] { (pieces[0], 3), (pieces[1], 1), (pieces[2], 4) })
            {
                QuantWeightInfo info = Assert.IsType<QuantWeightInfo>(piece.QuantInfo);
                Assert.Equal(rows, (int)info.RowScale!.Shape[0]);
                // ConvRot rotates along the INPUT dimension, which a row split leaves alone.
                Assert.Equal(256, info.ConvRotGroupSize);
                ReadOnlySpan<float> scales = info.RowScale.AsReadOnlySpan<float>();
                for (int row = 0; row < rows; row++)
                    Assert.Equal(0.001f * (firstRow + row + 1), scales[row], 6);
                firstRow += rows;
            }
        }
        finally
        {
            DisposeAll(pieces);
        }
    }

    [Fact]
    public void SplitRows_CarriesThePerTensorFp8ScaleOntoEveryPiece()
    {
        using Tensor fused = new Tensor(new TensorShape(6, 4), DType.F8E4M3)
        {
            Fp8ScaleFactor = 0.0125f,
            Fp8InputScaleFactor = 2.5f,
        };

        Tensor[] pieces = CheckpointConvertUtils.SplitRows(fused, [2, 2, 2], ["q", "k", "v"]);
        try
        {
            foreach (Tensor piece in pieces)
            {
                // The raw fp8 bytes alone are value/scale; dropping the factor runs the projection hundreds of
                // times too large and renders as noise rather than failing.
                Assert.Equal(0.0125f, piece.Fp8ScaleFactor);
                Assert.Equal(2.5f, piece.Fp8InputScaleFactor);
            }
        }
        finally
        {
            DisposeAll(pieces);
        }
    }

    [Fact]
    public void SplitRows_SplitsARankOneBias()
    {
        using Tensor fused = new Tensor(new TensorShape(6), DType.F32);
        Span<float> values = fused.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = i;

        Tensor[] pieces = CheckpointConvertUtils.SplitRows(fused, [2, 4], ["q.bias", "mlp.bias"]);
        try
        {
            Assert.Equal([0f, 1f], pieces[0].AsReadOnlySpan<float>().ToArray());
            Assert.Equal([2f, 3f, 4f, 5f], pieces[1].AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            DisposeAll(pieces);
        }
    }

    [Fact]
    public void SplitRows_RefusesARowLengthThatIsNotAWholeNumberOfBlocks()
    {
        // 48 values per row is not a multiple of Q8_0's 32, so a row boundary falls inside a block and no byte
        // offset can name it. Refusing beats copying a shifted window that decodes to plausible garbage.
        using Tensor fused = new Tensor(new TensorShape(4, 48), DType.Q8_0);
        Assert.Throws<NotSupportedException>(() =>
            CheckpointConvertUtils.SplitRows(fused, [2, 2], ["q.weight", "k.weight"]));
    }

    [Fact]
    public void SplitRows_RefusesRowCountsThatDoNotCoverTheTensor()
    {
        using Tensor fused = new Tensor(new TensorShape(6, 4), DType.F32);
        Assert.Throws<InvalidOperationException>(() =>
            CheckpointConvertUtils.SplitRows(fused, [2, 2], ["q.weight", "k.weight"]));
    }
}
