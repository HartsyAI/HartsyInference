using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Kernels;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary><see cref="Int8GemvKernels"/> must produce exactly what RNNoise's default C build does with the same
/// int8 tables and codes. That rests on three things, each checked here against an independent reference: the sums
/// are exact (no int16 saturation anywhere), the float epilogue rounds in upstream's order, and the AVX2 and scalar
/// paths agree bit for bit. The codes themselves follow the default build's rounding, which differs from the AVX2
/// build's at ties.</summary>
public sealed unsafe class Int8GemvKernelTests
{
    /// <summary>RNNoise's shapes and small remainders, plus two wide ones: at K 8192 and 16384 a call holds only
    /// four or two activation rows' split codes on the stack, so five rows take several passes over the weights.</summary>
    public static TheoryData<int, int, int> Shapes()
    {
        TheoryData<int, int, int> data = new();
        foreach (int m in new[] { 1, 2, 3, 5 })
            foreach ((int n, int k) in new[] { (8, 4), (8, 12), (16, 32), (24, 40), (1152, 384), (384, 384), (8, 8192), (16, 16384) })
                data.Add(m, n, k);
        return data;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Linear_IsTheExactSumWithUpstreamsEpilogue(int m, int n, int k)
    {
        foreach ((bool withBias, bool withDiag) in new[] { (false, false), (true, false), (true, true) })
        {
            if (withDiag && (n % k != 0 || k % 8 != 0)) continue;
            using Case c = Case.Random(m, n, k, withBias, withDiag, seed: m * 7919 + n * 31 + k);
            float[] expected = c.Reference();
            using CpuBackend backend = new();
            foreach (bool inline in new[] { true, false })
            {
                using Tensor output = new(new TensorShape(m, n), DType.F32);
                if (inline)
                {
                    using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
                    c.Run(backend, output);
                }
                else
                {
                    c.Run(backend, output);
                }
                AssertBits(expected, output, $"M={m} N={n} K={k} bias={withBias} diag={withDiag} inline={inline}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SimdAndScalarPaths_AgreeBitForBit(int m, int n, int k)
    {
        bool withDiag = n % k == 0 && k % 8 == 0;
        using Case c = Case.Random(m, n, k, withBias: true, withDiag, seed: m + n + k);
        using Tensor simd = new(new TensorShape(m, n), DType.F32);
        using Tensor scalar = new(new TensorShape(m, n), DType.F32);
        Int8GemvKernels.Linear(simd, c.Codes, c.Tiles, c.Scale, c.Bias, c.Diag, c.DiagInput);
        Int8GemvKernels.LinearScalar(scalar, c.Codes, c.Tiles, c.Scale, c.Bias, c.Diag, c.DiagInput);
        Assert.True(simd.AsSpan<byte>().SequenceEqual(scalar.AsSpan<byte>()),
            $"M={m} N={n} K={k}: the AVX2 path differs from the scalar path (Avx2 {Avx2.IsSupported})");
    }

    [Fact]
    public void EachRowOfABatch_MatchesThatRowAlone()
    {
        const int n = 1152, k = 384, m = 3;
        using Case batch = Case.Random(m, n, k, withBias: true, withDiag: true, seed: 5);
        using CpuBackend backend = new();
        using Tensor all = new(new TensorShape(m, n), DType.F32);
        batch.Run(backend, all);
        for (int r = 0; r < m; r++)
        {
            using Case single = batch.Row(r);
            using Tensor one = new(new TensorShape(1, n), DType.F32);
            single.Run(backend, one);
            Assert.True(all.AsSpan<byte>().Slice(r * n * sizeof(float), n * sizeof(float)).SequenceEqual(one.AsSpan<byte>()),
                $"row {r} of a {m}-row call differs from the same row alone");
        }
    }

    /// <summary>Weights of 127 against codes of 255: each pair of products is 64770, which <c>maddubs</c> would clip
    /// to 32767. The sum here must be the true one.</summary>
    [Fact]
    public void PairsThatWouldSaturateInt16_StayExact()
    {
        const int n = 8, k = 8;
        using Case c = Case.Constant(n, k, weight: 127, code: 255);
        using CpuBackend backend = new();
        using Tensor output = new(new TensorShape(1, n), DType.F32);
        c.Run(backend, output);
        foreach (float value in output.AsSpan<float>()) Assert.Equal(255f * 127f * k, value);

        using Case negative = Case.Constant(n, k, weight: -128, code: 255);
        negative.Run(backend, output);
        foreach (float value in output.AsSpan<float>()) Assert.Equal(-255f * 128f * k, value);
    }

    [Fact]
    public void Linear_AllocatesNothingInline()
    {
        using Case c = Case.Random(2, 1152, 384, withBias: true, withDiag: true, seed: 9);
        using CpuBackend backend = new();
        using Tensor output = new(new TensorShape(2, 1152), DType.F32);
        using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
        for (int i = 0; i < 10; i++) c.Run(backend, output);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            backend.QuantizeActivationsU8(c.Codes, c.Source);
            c.Run(backend, output);
        }
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    /// <summary>The default build adds the half in double after rounding the product to float, then floors: a tie
    /// goes up. Both ties here would go the other way under round-half-to-even, which is what the AVX2 build's
    /// <c>cvtps</c> does.</summary>
    [Fact]
    public void QuantizeActivation_RoundsTiesUp_AsUpstreamsDefaultBuild()
    {
        float up = WithProduct(1.5f);
        Assert.Equal(1.5f, 127f * up);
        Assert.Equal(129, Int8Tiles.QuantizeActivation(up));

        float down = WithProduct(-1.5f);
        Assert.Equal(-1.5f, 127f * down);
        Assert.Equal(126, Int8Tiles.QuantizeActivation(down));
    }

    [Fact]
    public void QuantizeActivation_MapsTheUnitRangeTo0Through254_AndClampsOutside()
    {
        Assert.Equal(0, Int8Tiles.QuantizeActivation(-1f));
        Assert.Equal(127, Int8Tiles.QuantizeActivation(0f));
        Assert.Equal(254, Int8Tiles.QuantizeActivation(1f));
        Assert.Equal(255, Int8Tiles.QuantizeActivation(1.01f));
        Assert.Equal(255, Int8Tiles.QuantizeActivation(1e30f));
        Assert.Equal(0, Int8Tiles.QuantizeActivation(-1.01f));
        Assert.Equal(0, Int8Tiles.QuantizeActivation(float.NegativeInfinity));
        Assert.Equal(0, Int8Tiles.QuantizeActivation(float.NaN));
    }

    [Fact]
    public void Tiles_HoldEightRowsOfFourWeights_RowByRow()
    {
        const int n = 16, k = 12;
        sbyte[] rowMajor = new sbyte[n * k];
        for (int i = 0; i < rowMajor.Length; i++) rowMajor[i] = (sbyte)(i % 251 - 125);
        sbyte[] tiles = new sbyte[n * k];
        Int8Tiles.Pack(rowMajor, n, k, tiles);

        // The second 8-row block's third tile starts at (1·3 + 2)·32; its row 5 holds inputs 8..11 of output 13.
        int start = (1 * 3 + 2) * Int8Tiles.Bytes + 5 * Int8Tiles.Cols;
        for (int c = 0; c < 4; c++) Assert.Equal(rowMajor[13 * k + 8 + c], tiles[start + c]);

        sbyte[] back = new sbyte[n * k];
        Int8Tiles.Unpack(tiles, n, k, back);
        Assert.Equal(rowMajor, back);
    }

    /// <summary>The float nearest <paramref name="target"/>/127 whose product with 127 rounds to exactly
    /// <paramref name="target"/>.</summary>
    private static float WithProduct(float target)
    {
        float x = target / 127f;
        for (int step = 0; step < 64; step++)
        {
            if (127f * x == target) return x;
            x = 127f * x < target ? MathF.BitIncrement(x) : MathF.BitDecrement(x);
        }
        throw new InvalidOperationException($"no float x has fl(127·x) == {target}");
    }

    private static void AssertBits(float[] expected, Tensor actual, string label)
    {
        ReadOnlySpan<float> values = actual.AsSpan<float>();
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(expected[i]) != BitConverter.SingleToUInt32Bits(values[i]))
                Assert.Fail($"{label}: element {i} is {values[i]:R}, expected {expected[i]:R}");
        }
    }

    /// <summary>One product's operands, kept both as tensors for the kernel and as plain arrays for the reference.</summary>
    private sealed class Case : IDisposable
    {
        private readonly sbyte[] _weights;   // row-major [n, k]
        private readonly byte[] _codes;
        private readonly float[] _scale;
        private readonly float[]? _bias;
        private readonly float[]? _diag;
        private readonly float[]? _diagInput;

        public int M { get; }
        public int N { get; }
        public int K { get; }
        public Tensor Codes { get; }
        public Tensor Source { get; }
        public Tensor Tiles { get; }
        public Tensor Scale { get; }
        public Tensor? Bias { get; }
        public Tensor? Diag { get; }
        public Tensor? DiagInput { get; }

        private Case(int m, int n, int k, sbyte[] weights, byte[] codes, float[] source, float[] scale, float[]? bias,
            float[]? diag)
        {
            M = m; N = n; K = k;
            _weights = weights; _codes = codes; _scale = scale; _bias = bias; _diag = diag;
            _diagInput = diag is null ? null : source;
            Codes = Make(new TensorShape(m, k), DType.U8, codes);
            Source = Make(new TensorShape(m, k), DType.F32, source);
            sbyte[] tiles = new sbyte[n * k];
            Int8Tiles.Pack(weights, n, k, tiles);
            Tiles = Make(new TensorShape(n / 8, k / 4, 8, 4), DType.I8, tiles);
            Scale = Make(new TensorShape(n), DType.F32, scale);
            Bias = bias is null ? null : Make(new TensorShape(n), DType.F32, bias);
            Diag = diag is null ? null : Make(new TensorShape(n), DType.F32, diag);
            DiagInput = diag is null ? null : Source;
        }

        public static Case Random(int m, int n, int k, bool withBias, bool withDiag, int seed)
        {
            Random rng = new(seed);
            sbyte[] weights = new sbyte[n * k];
            for (int i = 0; i < weights.Length; i++) weights[i] = (sbyte)rng.Next(-128, 128);
            float[] source = new float[m * k];
            byte[] codes = new byte[m * k];
            for (int i = 0; i < source.Length; i++)
            {
                // Mostly the unit range, with some codes pushed to the ends so the extremes are covered.
                source[i] = (float)(rng.NextDouble() * 2.2 - 1.1);
                codes[i] = Int8Tiles.QuantizeActivation(source[i]);
            }
            float[] scale = Floats(rng, n, 1e-6, 1e-3);
            return new Case(m, n, k, weights, codes, source, scale, withBias ? Floats(rng, n, -1, 1) : null,
                withDiag ? Floats(rng, n, -1, 1) : null);
        }

        public static Case Constant(int n, int k, sbyte weight, byte code)
        {
            sbyte[] weights = Enumerable.Repeat(weight, n * k).ToArray();
            byte[] codes = Enumerable.Repeat(code, k).ToArray();
            return new Case(1, n, k, weights, codes, new float[k], Enumerable.Repeat(1f, n).ToArray(), null, null);
        }

        /// <summary>Row <paramref name="r"/> of this case as a case of its own.</summary>
        public Case Row(int r) => new(1, N, K, _weights, _codes.AsSpan(r * K, K).ToArray(),
            Source.AsSpan<float>().Slice(r * K, K).ToArray(), _scale, _bias, _diag);

        public void Run(CpuBackend backend, Tensor output) =>
            backend.LinearI8U8(output, Codes, Tiles, Scale, Bias, Diag, DiagInput);

        /// <summary>The product as upstream's default build computes it: the exact sum, converted to float, then
        /// three separately rounded steps.</summary>
        public float[] Reference()
        {
            float[] y = new float[M * N];
            for (int r = 0; r < M; r++)
            {
                for (int n = 0; n < N; n++)
                {
                    long sum = 0;
                    for (int c = 0; c < K; c++) sum += (long)_codes[r * K + c] * _weights[n * K + c];
                    float value = (float)sum;
                    value *= _scale[n];
                    if (_bias is not null) value += _bias[n];
                    if (_diag is not null)
                    {
                        float term = _diag[n] * _diagInput![r * K + n % K];
                        value += term;
                    }
                    y[r * N + n] = value;
                }
            }
            return y;
        }

        public void Dispose()
        {
            Codes.Dispose(); Source.Dispose(); Tiles.Dispose(); Scale.Dispose(); Bias?.Dispose(); Diag?.Dispose();
        }

        private static float[] Floats(Random rng, int count, double low, double high)
        {
            float[] values = new float[count];
            for (int i = 0; i < count; i++) values[i] = (float)(low + rng.NextDouble() * (high - low));
            return values;
        }

        private static Tensor Make<T>(TensorShape shape, DType dtype, T[] values) where T : unmanaged
        {
            Tensor tensor = new(shape, dtype);
            values.AsSpan().CopyTo(tensor.AsSpan<T>());
            return tensor;
        }
    }
}
