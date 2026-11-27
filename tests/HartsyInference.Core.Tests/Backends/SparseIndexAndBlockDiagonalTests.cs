using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed class SparseIndexAndBlockDiagonalTests
{
    [Fact]
    public void SparseTopK_MatchesBruteForceOrder_WithTiesNaNInfinitiesAndSignedZeros()
    {
        const int rows = 6, cols = 37, topK = 9, offset = 100;
        Random rng = new(5);
        float[] specials = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0f, 0f, 1f, 1f, -2f];
        float[] scores = new float[rows * cols];
        for (int i = 0; i < scores.Length; i++)
            scores[i] = rng.Next(3) == 0 ? specials[rng.Next(specials.Length)] : rng.Next(-3, 4) * 0.5f;
        int[] limits = [0, 5, 20, cols, cols + 4, 9];
        using Tensor s = Make(scores, rows, cols), l = MakeInts(limits, rows);
        using Tensor indices = new(new TensorShape(rows, topK), DType.I32);

        new CpuBackend().SparseTopK(indices, s, l, topK, offset);

        int[] got = indices.AsReadOnlySpan<int>().ToArray();
        for (int r = 0; r < rows; r++)
        {
            int[] order = Enumerable.Range(0, cols).ToArray();
            float[] row = scores.AsSpan(r * cols, cols).ToArray();
            Array.Sort(order, (a, b) =>
            {
                int c = row[b].CompareTo(row[a]);
                return c != 0 ? c : a.CompareTo(b);
            });
            int[] want = order.Take(topK).Order().Select(c => c < limits[r] ? c + offset : -1).ToArray();
            Assert.Equal(want, got.AsSpan(r * topK, topK).ToArray());
        }
    }

    [Fact]
    public void SparseTopK_ClampsKToTheRowWidth()
    {
        using Tensor s = Make([3f, float.NegativeInfinity, 1f], 1, 3), l = MakeInts([2], 1);
        using Tensor indices = new(new TensorShape(1, 3), DType.I32);
        new CpuBackend().SparseTopK(indices, s, l, 10, 0);
        Assert.Equal([0, 1, -1], indices.AsReadOnlySpan<int>().ToArray());

        using Tensor wrong = new(new TensorShape(1, 10), DType.I32);
        Assert.Throws<ArgumentException>(() => new CpuBackend().SparseTopK(wrong, s, l, 10, 0));
    }

    [Fact]
    public void MergeWindowTopK_PlacesTheTopKBeforeTheWindowAndPadsWithMinusOne()
    {
        const int window = 3, cols = 4, width = cols + window + 1;
        int[] positions = [0, 2, 3, 9];
        int[] topk = [-7, -7, -7, -7, -7, -7, -7, -7, 0, -1, -1, -1, 1, 4, 5, -1];
        using Tensor t = MakeInts(topk, 4, cols), p = MakeInts(positions, 4);
        using Tensor rows = new(new TensorShape(4, width), DType.I32);

        new CpuBackend().MergeWindowTopK(rows, t, p, window);

        int[] want =
        [
            0, -1, -1, -1, -1, -1, -1, -1,     // seq 1: no top-k, window [0]
            0, 1, 2, -1, -1, -1, -1, -1,       // seq 3: no top-k, window [0..2]
            0, 1, 2, 3, -1, -1, -1, -1,        // seq 4: one top-k entry, window [1..3]
            1, 4, 5, -1, 7, 8, 9, -1,          // seq 10: n = min(7, 4) = 4 entries copied as given, window [7..9]
        ];
        Assert.Equal(want, rows.AsReadOnlySpan<int>().ToArray());
    }

    [Fact]
    public void BlockDiagonalLinear_F32_ReadsOnlyEachGroupsSlice()
    {
        const int m = 3, groups = 4, r = 5, d = 7;
        Random rng = new(9);
        float[] x = RandomFloats(rng, m * groups * d), w = RandomFloats(rng, groups * r * d);
        using Tensor input = Make(x, m, groups * d), weight = Make(w, groups * r, d);
        using Tensor output = new(new TensorShape(m, groups * r), DType.F32);

        new CpuBackend().BlockDiagonalLinear(output, input, weight, null, groups);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < m; i++)
            for (int n = 0; n < groups * r; n++)
            {
                float sum = 0f;
                for (int c = 0; c < d; c++) sum += x[i * groups * d + n / r * d + c] * w[n * d + c];
                Assert.Equal(BitConverter.SingleToInt32Bits(sum), BitConverter.SingleToInt32Bits(y[i * groups * r + n]));
            }
    }

    [Fact]
    public void BlockDiagonalLinear_Fp8_EqualsFp8BlockLinearPerGroup()
    {
        // 128 rows per group keeps each group on whole scale-grid rows, so a group is an ordinary block-FP8 linear.
        const int m = 3, groups = 2, r = 128, d = 200, block = 128;
        Random rng = new(13);
        byte[] codes = new byte[groups * r * d];
        for (int i = 0; i < codes.Length; i++)
        {
            byte c = LatentCodec.EncodeE4M3((float)(rng.NextDouble() * 2 - 1));
            codes[i] = (c & 0x7F) == 0x7F ? (byte)0 : c;
        }
        float[] scales = RandomFloats(rng, groups * 2).Select(v => 0.5f + MathF.Abs(v)).ToArray();
        float[] x = RandomFloats(rng, m * groups * d);
        CpuBackend cpu = new();
        using Tensor input = Make(x, m, groups * d), weight = MakeBytes(codes, groups * r, d), scale = Make(scales, groups, 2);
        using Tensor output = new(new TensorShape(m, groups * r), DType.F32);

        cpu.BlockDiagonalLinear(output, input, weight, scale, groups, block, block);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        for (int g = 0; g < groups; g++)
        {
            float[] xg = new float[m * d];
            for (int i = 0; i < m; i++) Array.Copy(x, i * groups * d + g * d, xg, i * d, d);
            using Tensor gx = Make(xg, m, d), gw = MakeBytes(codes.AsSpan(g * r * d, r * d).ToArray(), r, d);
            using Tensor gs = Make(scales.AsSpan(g * 2, 2).ToArray(), 1, 2);
            using Tensor gy = new(new TensorShape(m, r), DType.F32);
            cpu.Fp8BlockLinear(gy, gx, gw, gs, null, block, block);
            float[] want = gy.AsReadOnlySpan<float>().ToArray();
            // The CPU block linear is a vectorized GEMM over the decoded weight, while the grouped reference accumulates in double,
            // so the two agree to float rounding rather than bit for bit.
            for (int i = 0; i < m; i++)
                for (int j = 0; j < r; j++)
                {
                    float a = y[i * groups * r + g * r + j], b = want[i * r + j];
                    Assert.True(MathF.Abs(a - b) <= 1e-5f * MathF.Max(1f, MathF.Abs(b)), $"group {g} row {i} col {j}: grouped {a}, linear {b}.");
                }
        }
    }

    [Fact]
    public void BlockDiagonalLinear_RefusesMismatchedOperands()
    {
        CpuBackend cpu = new();
        using Tensor input = new(new TensorShape(2, 12), DType.F32), weight = new(new TensorShape(9, 4), DType.F32);
        using Tensor output = new(new TensorShape(2, 9), DType.F32);
        Assert.Throws<ArgumentException>(() => cpu.BlockDiagonalLinear(output, input, weight, null, 2));
        using Tensor scale = new(new TensorShape(1, 1), DType.F32);
        using Tensor even = new(new TensorShape(6, 4), DType.F32), evenOut = new(new TensorShape(2, 6), DType.F32);
        Assert.Throws<ArgumentException>(() => cpu.BlockDiagonalLinear(evenOut, input, even, scale, 3));
        cpu.BlockDiagonalLinear(evenOut, input, even, null, 3);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor MakeInts(int[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        values.CopyTo(t.AsSpan<int>());
        return t;
    }

    private static Tensor MakeBytes(byte[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F8E4M3);
        values.CopyTo(t.AsSpan<byte>());
        return t;
    }
}
