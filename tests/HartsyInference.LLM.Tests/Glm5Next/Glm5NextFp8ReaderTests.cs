using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Glm5Next;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The glm5_next host readers decode a block-scaled FP8 weight through its 128x128 grid. A plain cast reads the codes without the grid and
/// returns values off by the scale, so these tests set the scale to 2 and assert the doubled values.</summary>
public sealed class Glm5NextFp8ReaderTests
{
    // E4M3 0x38 is exactly 1.0.
    private const byte One = 0x38;

    [Fact]
    public void ScaledFp8Vector_IsDecodedThroughItsBlockGrid()
    {
        float[] values = ReadAsF32(() => Scaled(rows: 2, cols: 128, scale: 2f));

        Assert.Equal(256, values.Length);
        Assert.All(values, v => Assert.Equal(2f, v));
    }

    [Fact]
    public void UnscaledFp8Vector_IsCastPlain()
    {
        float[] values = ReadAsF32(() => Plain(rows: 2, cols: 128));

        Assert.All(values, v => Assert.Equal(1f, v));
    }

    [Fact]
    public void ScaledFp8Vector_DecodesEveryBlockWithItsOwnScale()
    {
        // Two column blocks, scales 2 and 3: the first 128 columns read 2, the next 128 read 3.
        float[] values = ReadAsF32(() =>
        {
            Tensor weight = new(new TensorShape(1, 256), DType.F8E4M3);
            weight.AsSpan<byte>().Fill(One);
            Tensor grid = new(new TensorShape(1, 2), DType.F32);
            grid.AsSpan<float>()[0] = 2f;
            grid.AsSpan<float>()[1] = 3f;
            weight.QuantInfo = new QuantWeightInfo { Format = "fp8_block_scale_inv", Fp8BlockScaleInv = grid, Fp8BlockSize = 128 };
            return weight;
        });

        Assert.All(values[..128], v => Assert.Equal(2f, v));
        Assert.All(values[128..], v => Assert.Equal(3f, v));
    }

    private static float[] ReadAsF32(Func<Tensor> make)
    {
        IErnie5WeightSource source = new OneWeight(make);
        using Tensor f32 = Glm5NextLayer.F32(source, "w", elements: 256);
        return f32.AsReadOnlySpan<float>().ToArray();
    }

    private static Tensor Scaled(int rows, int cols, float scale)
    {
        Tensor weight = Plain(rows, cols);
        int gridRows = (rows + 127) / 128, gridCols = (cols + 127) / 128;
        Tensor grid = new(new TensorShape(gridRows, gridCols), DType.F32);
        grid.AsSpan<float>().Fill(scale);
        weight.QuantInfo = new QuantWeightInfo { Format = "fp8_block_scale_inv", Fp8BlockScaleInv = grid, Fp8BlockSize = 128 };
        return weight;
    }

    private static Tensor Plain(int rows, int cols)
    {
        Tensor weight = new(new TensorShape(rows, cols), DType.F8E4M3);
        weight.AsSpan<byte>().Fill(One);
        return weight;
    }

    // Hands out a fresh tensor for the one name; the reader owns and disposes what it is given.
    private sealed class OneWeight(Func<Tensor> make) : IErnie5WeightSource
    {
        public IEnumerable<string> Names => ["w"];

        public Tensor Read(string name) => make();
    }
}
