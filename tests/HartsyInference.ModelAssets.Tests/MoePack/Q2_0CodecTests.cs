using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.MoePack;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.MoePack;

/// <summary>The pack-local Q2_0 codec: its error on random data, exact reproduction of grid values, and its refusals.</summary>
public sealed class Q2_0CodecTests
{
    // Measured on the seeded data below (8 x 256): relative RMSE 0.321 for uniform [-1, 1] values and 0.381 for Gaussian
    // values, which is what a 4-level grid at 2.25 bits per weight gives. The tolerance sits just above the worst of those.
    internal const double MaxRelativeRmse = 0.40;

    private static float[] UniformValues(int count, int seed)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static float[] GaussianValues(int count, int seed)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            values[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return values;
    }

    private static (double Relative, double MaxAbs) Error(float[] original, float[] decoded)
    {
        double errorSquares = 0, referenceSquares = 0, maxAbs = 0;
        for (int i = 0; i < original.Length; i++)
        {
            double error = decoded[i] - original[i];
            errorSquares += error * error;
            referenceSquares += (double)original[i] * original[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(error));
        }
        return (Math.Sqrt(errorSquares / referenceSquares), maxAbs);
    }

    [Fact]
    public void Uniform_Values_RoundTripWithinTolerance()
    {
        const int rows = 8, cols = 256;
        float[] original = UniformValues(rows * cols, 1);

        float[] decoded = Q2_0Codec.Decode(Q2_0Codec.Encode(original, rows, cols), rows, cols);

        (double relative, double maxAbs) = Error(original, decoded);
        Assert.True(relative < MaxRelativeRmse, $"Uniform relative RMSE {relative:F4} exceeds {MaxRelativeRmse}; max abs {maxAbs:F4}.");
    }

    [Fact]
    public void Gaussian_Values_RoundTripWithinTolerance()
    {
        const int rows = 8, cols = 256;
        float[] original = GaussianValues(rows * cols, 2);

        float[] decoded = Q2_0Codec.Decode(Q2_0Codec.Encode(original, rows, cols), rows, cols);

        (double relative, double maxAbs) = Error(original, decoded);
        Assert.True(relative < MaxRelativeRmse, $"Gaussian relative RMSE {relative:F4} exceeds {MaxRelativeRmse}; max abs {maxAbs:F4}.");
    }

    [Fact]
    public void Encoded_Size_Is_Eighteen_Bytes_Per_Sixty_Four_Values()
    {
        Assert.Equal(18, Q2_0Codec.BlockBytes);
        Assert.Equal(18L * 4 * 2, Q2_0Codec.ByteCount(4, 128));
        Assert.Equal(DType.Q2_0.ComputeByteCount(4 * 128), Q2_0Codec.ByteCount(4, 128));
        Assert.Equal(144, Q2_0Codec.Encode(new float[4 * 128], 4, 128).Length);
    }

    [Fact]
    public void Zero_Block_Encodes_To_Zero_Bytes_And_Decodes_To_Zero()
    {
        float[] zeros = new float[64];

        byte[] packed = Q2_0Codec.Encode(zeros, 1, 64);
        float[] decoded = Q2_0Codec.Decode(packed, 1, 64);

        Assert.All(packed, static b => Assert.Equal(0, b));
        Assert.All(decoded, static v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Block_Of_Exact_Grid_Values_RoundTripsExactly()
    {
        // Scale 0.5 is exact in fp16, and the first value is the grid's top code (2 * 0.5 = 1.0), so the max-abs starting
        // scale is already the true one. Every other value is a grid point at that scale, so the decode must match exactly.
        const float scale = 0.5f;
        float[] grid = [-1f, 0f, 1f, 2f];
        Random rng = new(3);
        float[] original = new float[64];
        original[0] = 2 * scale;
        for (int i = 1; i < 64; i++) original[i] = grid[rng.Next(4)] * scale;

        float[] decoded = Q2_0Codec.Decode(Q2_0Codec.Encode(original, 1, 64), 1, 64);

        for (int i = 0; i < 64; i++) Assert.Equal(original[i], decoded[i]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(100)]
    public void Columns_Not_A_Multiple_Of_64_Are_Refused(int cols)
    {
        float[] values = new float[cols];
        Assert.Throws<ArgumentException>(() => Q2_0Codec.Encode(values, 1, cols));
        Assert.Throws<ArgumentException>(() => Q2_0Codec.Decode(new byte[18], 1, cols));
    }

    [Fact]
    public void Decode_Refuses_A_Byte_Count_That_Does_Not_Match_The_Shape()
    {
        Assert.Throws<ArgumentException>(() => Q2_0Codec.Decode(new byte[17], 1, 64));
    }

    [Fact]
    public void Non_Finite_Values_Are_Refused()
    {
        float[] values = new float[64];
        values[5] = float.NaN;
        Assert.Throws<ArgumentException>(() => Q2_0Codec.Encode(values, 1, 64));
    }
}
