using System.Numerics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed class HadamardAndFp8x128Tests
{
    [Fact]
    public void Hadamard_MatchesDenseSylvesterMatrix()
    {
        const int rows = 3, width = 128;
        float[] x = Random(new Random(11), rows * width, 1f);
        float scale = 1f / MathF.Sqrt(width);
        using Tensor input = Make(x, rows, width);
        using Tensor output = new(new TensorShape(rows, width), DType.F32);

        new CpuBackend().Hadamard(output, input, scale);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < width; j++)
            {
                double sum = 0;
                for (int i = 0; i < width; i++) sum += x[r * width + i] * Sylvester(i, j);
                Assert.Equal(sum * scale, y[r * width + j], 1e-5);
            }
    }

    [Fact]
    public void Hadamard_OrthonormalInPlaceIsItsOwnInverse()
    {
        const int width = 64;
        float[] x = Random(new Random(3), 2 * width, 2f);
        using Tensor t = Make(x, 2, width);
        CpuBackend backend = new();

        backend.Hadamard(t, t, 1f / MathF.Sqrt(width));
        backend.Hadamard(t, t, 1f / MathF.Sqrt(width));

        float[] back = t.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < x.Length; i++) Assert.Equal(x[i], back[i], 1e-5);
    }

    [Fact]
    public void Hadamard_RejectsOutputWithDifferentRowWidth()
    {
        using Tensor input = new(new TensorShape(2, 128), DType.F32);
        using Tensor output = new(new TensorShape(128, 2), DType.F32);
        Assert.Throws<ArgumentException>(() => HadamardReference.Validate(output, input));
    }

    [Fact]
    public void Fp8x128_EqualsFp8x32_WhenEverySubgroupSharesTheMaximum()
    {
        // Each 32-wide subgroup holds the same absolute maximum, so one 128 group and four 32 groups pick the same scale.
        float[] row = Random(new Random(5), 128, 3f);
        for (int g = 0; g < 4; g++) row[g * 32 + 7] = g % 2 == 0 ? 10f : -10f;

        float[] wide = RoundTrip(row, LatentEncoding.Fp8E4M3Ue8m0x128);
        float[] narrow = RoundTrip(row, LatentEncoding.Fp8E4M3Ue8m0x32);

        Assert.Equal(narrow, wide);
    }

    [Fact]
    public void Fp8x128_SharesOneScaleAcross128Values()
    {
        // A single large value in the first 32 sets the scale of the whole 128 group, so small values later in the
        // group lose precision that 32-wide groups keep.
        float[] row = new float[128];
        for (int i = 0; i < row.Length; i++) row[i] = 0.01f * (1 + i % 5);
        row[0] = 1e5f;

        float[] wide = RoundTrip(row, LatentEncoding.Fp8E4M3Ue8m0x128);
        float[] narrow = RoundTrip(row, LatentEncoding.Fp8E4M3Ue8m0x32);

        double wideErr = 0, narrowErr = 0;
        for (int i = 32; i < 128; i++)
        {
            wideErr = Math.Max(wideErr, Math.Abs(wide[i] - row[i]));
            narrowErr = Math.Max(narrowErr, Math.Abs(narrow[i] - row[i]));
            // e4m3 keeps 3 mantissa bits: at most 1/16 relative error
            Assert.True(Math.Abs(narrow[i] - row[i]) <= Math.Abs(row[i]) / 16, $"32-wide group element {i}: {row[i]} -> {narrow[i]}");
        }
        Assert.True(wideErr > 4 * narrowErr, $"The 128 group should share the large scale; errors {wideErr} vs {narrowErr}.");
    }

    [Fact]
    public void Fp8x128_TinyGroupUnderTheAmaxFloorStaysFiniteAndClose()
    {
        float[] row = new float[256];
        Random rng = new(9);
        for (int i = 0; i < 128; i++) row[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 128; i < 256; i++) row[i] = (float)((rng.NextDouble() * 2 - 1) * 5e-5);

        float[] back = RoundTrip(row, LatentEncoding.Fp8E4M3Ue8m0x128);

        for (int i = 0; i < row.Length; i++) Assert.True(float.IsFinite(back[i]));
        // The 1e-4 amax floor still leaves e4m3 precision (1/16 relative) for values this small.
        for (int i = 128; i < 256; i++) Assert.True(Math.Abs(back[i] - row[i]) <= Math.Abs(row[i]) / 16 + 1e-9, $"element {i}: {row[i]} -> {back[i]}");
    }

    [Fact]
    public void Fp8x128_UsesGroupsOf128()
    {
        Assert.Equal(128, LatentEncodings.GroupSize(LatentEncoding.Fp8E4M3Ue8m0x128));
        Assert.True(LatentEncodings.IsFp8(LatentEncoding.Fp8E4M3Ue8m0x128));
        Assert.Throws<NotSupportedException>(() => LatentEncodings.RequireDeviceKernel(LatentEncoding.Fp8E4M3Ue8m0x128, "test"));
    }

    private static float[] RoundTrip(float[] row, LatentEncoding encoding)
    {
        using Tensor t = Make(row, 1, row.Length);
        new CpuBackend().ActQuantDequantInPlace(t, encoding);
        return t.AsReadOnlySpan<float>().ToArray();
    }

    private static int Sylvester(int i, int j) => (BitOperations.PopCount((uint)(i & j)) & 1) == 0 ? 1 : -1;

    private static float[] Random(Random rng, int n, float amplitude)
    {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2 - 1) * amplitude);
        return v;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
