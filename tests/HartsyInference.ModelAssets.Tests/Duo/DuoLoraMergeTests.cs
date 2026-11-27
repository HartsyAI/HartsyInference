using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>The Duo LoRA merge on synthetic tensors, against a scalar reference: the F32 delta is <c>(B @ A) * scale</c> added at the row offset, and
/// a BF16 weight takes two roundings, the delta and then the sum, both to nearest even, as upstream's <c>load_effective</c> does. No
/// released LoRA bytes are read; the real-bytes parity is not run.</summary>
public sealed class DuoLoraMergeTests
{
    private const int Rows = 4, Cols = 3, Rank = 2, LoraRows = 2, RowStart = 1;
    private const float Scale = 0.5f;

    [Fact]
    public void AnF32MergeAddsTheScaledProductAtTheRowOffset()
    {
        float[] weight = Values(1, Rows * Cols);
        float[] a = Values(2, Rank * Cols);
        float[] b = Values(3, LoraRows * Rank);
        using Tensor w = Make(new TensorShape(Rows, Cols), DType.F32, weight);
        using Tensor lora = Make(new TensorShape(Rank, Cols), DType.F32, a);
        using Tensor up = Make(new TensorShape(LoraRows, Rank), DType.F32, b);

        DuoLoraMerge.MergeRows(w, RowStart, lora, up, Scale, swapHalves: false);

        float[] actual = w.AsSpan<float>().ToArray();
        float[] delta = Delta(a, b);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                float expected = weight[(r * Cols) + c];
                if (r >= RowStart && r < RowStart + LoraRows)
                {
                    expected += delta[((r - RowStart) * Cols) + c];
                }
                Assert.InRange(actual[(r * Cols) + c], expected - 1e-5f, expected + 1e-5f);
            }
        }
    }

    [Fact]
    public void ABf16MergeRoundsTheDeltaThenTheSum()
    {
        float[] weight = Values(4, Rows * Cols);
        float[] a = Values(5, Rank * Cols);
        float[] b = Values(6, LoraRows * Rank);
        using Tensor w = MakeBf16(new TensorShape(Rows, Cols), weight);
        using Tensor lora = Make(new TensorShape(Rank, Cols), DType.F32, a);
        using Tensor up = Make(new TensorShape(LoraRows, Rank), DType.F32, b);

        DuoLoraMerge.MergeRows(w, RowStart, lora, up, Scale, swapHalves: false);

        float[] actual = Bf16ToFloat(w);
        float[] delta = Delta(a, b);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                float start = RoundBf16(weight[(r * Cols) + c]);
                float expected = start;
                if (r >= RowStart && r < RowStart + LoraRows)
                {
                    expected = RoundBf16(start + RoundBf16(delta[((r - RowStart) * Cols) + c]));
                }
                Assert.InRange(actual[(r * Cols) + c], expected - 1e-2f, expected + 1e-2f);
            }
        }
    }

    [Fact]
    public void AMergeThatDoesNotFitTheWeightIsRefused()
    {
        using Tensor w = Make(new TensorShape(Rows, Cols), DType.F32, Values(7, Rows * Cols));
        using Tensor lora = Make(new TensorShape(Rank, Cols), DType.F32, Values(8, Rank * Cols));
        using Tensor up = Make(new TensorShape(LoraRows, Rank), DType.F32, Values(9, LoraRows * Rank));

        Assert.Throws<ArgumentException>(() => DuoLoraMerge.MergeRows(w, Rows - 1, lora, up, Scale, swapHalves: false));
    }

    /// <summary><c>(B @ A)</c> with B <c>[LoraRows, Rank]</c> and A <c>[Rank, Cols]</c>, scaled.</summary>
    private static float[] Delta(float[] a, float[] b)
    {
        float[] delta = new float[LoraRows * Cols];
        for (int r = 0; r < LoraRows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                float sum = 0f;
                for (int k = 0; k < Rank; k++)
                {
                    sum += b[(r * Rank) + k] * a[(k * Cols) + c];
                }
                delta[(r * Cols) + c] = sum * Scale;
            }
        }
        return delta;
    }

    /// <summary>Round to BF16 with round-half-to-even, returned as a float.</summary>
    private static float RoundBf16(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
    }

    private static Tensor Make(TensorShape shape, DType dtype, float[] values)
    {
        Tensor tensor = new(shape, dtype);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor MakeBf16(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.BF16);
        Span<byte> bytes = tensor.AsSpan<byte>();
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]) >> 16;
            bytes[2 * i] = (byte)bits;
            bytes[(2 * i) + 1] = (byte)(bits >> 8);
        }
        return tensor;
    }

    private static float[] Bf16ToFloat(Tensor tensor)
    {
        ReadOnlySpan<byte> bytes = tensor.AsReadOnlySpan<byte>();
        float[] values = new float[(int)tensor.ElementCount];
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = (uint)(bytes[2 * i] | (bytes[(2 * i) + 1] << 8)) << 16;
            values[i] = BitConverter.UInt32BitsToSingle(bits);
        }
        return values;
    }

    private static float[] Values(int seed, int count)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            double h = Math.Sin((seed * 12.9898) + (i * 78.233)) * 43758.5453;
            double frac = h - Math.Floor(h);
            data[i] = (float)((2.0 * frac) - 1.0);
        }
        return data;
    }
}
