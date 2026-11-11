using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.LLM.Transformer.Quant;
using Xunit;

namespace HartsyInference.LLM.Tests.Quant;

/// <summary>The packed MXFP4 linear reference against a naive decode: E2M1 nibbles times 2^(scale - 127), low nibble first, one scale per
/// 32 columns.</summary>
public sealed class PackedMxfp4LinearTests
{
    // The E2M1 magnitudes for codes 0..7; codes 8..15 are the same with the sign bit set.
    private static readonly float[] E2M1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    [Theory]
    [InlineData(6, 64, 3)]
    [InlineData(4, 96, 1)]
    public void Apply_MatchesNaiveDecode(int outRows, int inCols, int n)
    {
        Random rng = new(outRows * 17 + inCols);
        byte[] nibbles = new byte[outRows * inCols / 2];
        rng.NextBytes(nibbles);
        int blocks = inCols / 32;
        byte[] scales = new byte[outRows * blocks];
        for (int i = 0; i < scales.Length; i++) scales[i] = (byte)rng.Next(118, 130);
        float[] x = new float[n * inCols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        // Naive decode of every weight, row by row.
        float[] expected = new float[n * outRows];
        for (int o = 0; o < outRows; o++)
        {
            for (int r = 0; r < n; r++)
            {
                double acc = 0;
                for (int c = 0; c < inCols; c++)
                {
                    int byteIndex = o * (inCols / 2) + c / 2;
                    int nib = (c & 1) == 0 ? nibbles[byteIndex] & 0xF : nibbles[byteIndex] >> 4;
                    float mag = E2M1[nib & 7];
                    float w = (nib & 8) != 0 ? -mag : mag;
                    double factor = Math.Pow(2, scales[o * blocks + c / 32] - 127);
                    acc += x[r * inCols + c] * w * factor;
                }
                expected[r * outRows + o] = (float)acc;
            }
        }

        using Tensor weight = Bytes(nibbles, outRows, inCols / 2);
        using Tensor scale = Bytes(scales, outRows, blocks);
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0,
            Geometry = new BlockGeometry(1, 32),
            ScaleDType = DType.U8,
            LogicalRows = outRows,
            LogicalCols = inCols,
            Scale = scale,
        };
        using Tensor input = Floats(x, n, inCols);
        using Tensor output = new(new TensorShape(n, outRows), DType.F32);

        PackedMxfp4Linear.Apply(output, input, weight, recipe);

        float[] actual = output.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 3);
    }

    private static Tensor Bytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.U8);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }

    private static Tensor Floats(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
