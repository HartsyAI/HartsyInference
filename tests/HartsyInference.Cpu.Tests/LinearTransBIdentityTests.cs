using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cpu.Tests;

/// <summary><see cref="MatMulKernels.LinearTransB"/> computes four weight rows at a time. That restructuring is
/// meant to be invisible: every CPU model's outputs, digests included, rest on this kernel, so it must produce the
/// same bits as the one-row loop it replaced, not merely close values. The reference below is that loop verbatim
/// (serially — the tile split never changes how one element is summed), compared byte for byte across shapes that
/// leave every remainder: rows not a multiple of four, K tails inside the 8-lane and 32-wide tiles, and bias on and
/// off. On a CPU without FMA the kernel takes the multiply-then-add branch; this box has FMA, so that branch is not
/// exercised here, and .NET 10 ignores <c>DOTNET_EnableFMA=0</c>, so it cannot be forced. Running with
/// <c>DOTNET_EnableAVX2=0</c> does force the scalar branch.</summary>
public sealed unsafe class LinearTransBIdentityTests(ITestOutputHelper log)
{
    private const int TileSize = 32;

    public static TheoryData<int, int, int> Shapes()
    {
        TheoryData<int, int, int> data = new();
        foreach (int m in new[] { 1, 2, 5 })
            foreach (int n in new[] { 1, 3, 4, 5, 31, 32, 33, 100 })
                foreach (int k in new[] { 1, 7, 8, 9, 31, 33, 384 })
                    data.Add(m, n, k);
        return data;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void FourRowPath_IsBitIdenticalToTheSingleRowLoop(int m, int n, int k)
    {
        foreach (bool withBias in new[] { false, true })
        {
            using Tensor input = Random(m, k, seed: m * 1000 + k);
            using Tensor weight = Random(n, k, seed: n * 100 + k);
            using Tensor bias = Random(1, n, seed: n);
            using Tensor actual = new(new TensorShape(m, n), DType.F32);
            using Tensor expected = new(new TensorShape(m, n), DType.F32);

            MatMulKernels.LinearTransB(actual, input, weight, withBias ? bias : null);
            Reference(expected, input, weight, withBias ? bias : null, m, n, k);

            Assert.True(actual.AsSpan<byte>().SequenceEqual(expected.AsSpan<byte>()),
                $"M={m} N={n} K={k} bias={withBias}: output differs from the single-row loop");
        }
    }

    /// <summary>The (outDim, inDim) of each product RNNoise's paired path runs on two rows: conv1 and conv2 as
    /// unrolled windows, a GRU input projection, and the two dense heads. The last shape is not RNNoise's. One row of
    /// it is too little work to fan out and two rows are enough, so it checks that a partition chosen from M changes
    /// no sum.</summary>
    public static TheoryData<int, int> TwoRowShapes() => new()
    {
        { 128, 195 }, { 384, 384 }, { 1152, 384 }, { 32, 1536 }, { 1, 1536 }, { 64, 768 },
    };

    /// <summary>RNNoise's paired path (<c>RnnoiseModel.ProcessPair</c>) promises the bits of two one-frame calls. That
    /// needs each row of a two-row product summed exactly as that row alone is. By construction it is:
    /// <list type="bullet">
    /// <item>a row's sums read only that row of the input;</item>
    /// <item>which column path a weight row takes depends on N alone;</item>
    /// <item>tiles write disjoint outputs, so neither the thread count nor the partition, which can change with M,
    /// reorders a sum.</item>
    /// </list>
    /// This checks it directly, inline and fanned out, with bias on and off. It runs whichever branch this CPU's
    /// instruction set selects, and the log says which.</summary>
    [Theory]
    [MemberData(nameof(TwoRowShapes))]
    public void EachRowOfATwoRowProduct_IsBitIdenticalToThatRowAlone(int n, int k)
    {
        log.WriteLine($"N={n} K={k}: Avx2 {Avx2.IsSupported}, Fma {Fma.IsSupported}; work for one row {(long)n * k}, "
            + $"for two {2L * n * k}, fan-out threshold {CpuParallel.MinWorkForParallel}");
        using Tensor rows = Random(2, k, seed: n + k);
        using Tensor weight = Random(n, k, seed: n * 7 + k);
        using Tensor bias = Random(1, n, seed: n * 3);
        using Tensor row = new(new TensorShape(1, k), DType.F32);
        using Tensor rowsOut = new(new TensorShape(2, n), DType.F32);
        using Tensor rowOut = new(new TensorShape(1, n), DType.F32);
        foreach (bool inline in new[] { true, false })
        {
            foreach (bool withBias in new[] { false, true })
            {
                Linear(rowsOut, rows, weight, withBias ? bias : null, inline);
                for (int r = 0; r < 2; r++)
                {
                    rows.AsSpan<float>().Slice(r * k, k).CopyTo(row.AsSpan<float>());
                    Linear(rowOut, row, weight, withBias ? bias : null, inline);
                    Assert.True(rowsOut.AsSpan<byte>().Slice(r * n * sizeof(float), n * sizeof(float))
                        .SequenceEqual(rowOut.AsSpan<byte>()),
                        $"N={n} K={k} row {r} inline={inline} bias={withBias}: differs from the same row alone");
                }
            }
        }
    }

    /// <summary>The kernel, either inside <see cref="CpuParallel.EnterInline"/> as the voice front end calls it or
    /// free to fan out as the wake path can.</summary>
    private static void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias, bool inline)
    {
        if (!inline)
        {
            MatMulKernels.LinearTransB(output, input, weight, bias);
            return;
        }
        using CpuParallel.InlineScope scope = CpuParallel.EnterInline();
        MatMulKernels.LinearTransB(output, input, weight, bias);
    }

    /// <summary>The kernel's inner loop before the four-row path, unchanged apart from running serially.</summary>
    private static void Reference(Tensor output, Tensor input, Tensor weight, Tensor? bias, int m, int n, int k)
    {
        float* pIn = (float*)input.DataPointer;
        float* pW = (float*)weight.DataPointer;
        float* pOut = (float*)output.DataPointer;
        NativeMemory.Clear(pOut, (nuint)(m * n * sizeof(float)));
        for (int ii = 0; ii < m; ii += TileSize)
        {
            int iEnd = Math.Min(ii + TileSize, m);
            for (int jj = 0; jj < n; jj += TileSize)
            {
                int jEnd = Math.Min(jj + TileSize, n);
                for (int kk = 0; kk < k; kk += TileSize)
                {
                    int kEnd = Math.Min(kk + TileSize, k);
                    for (int i = ii; i < iEnd; i++)
                    {
                        float* rowA = pIn + i * k;
                        float* rowOut = pOut + i * n;
                        for (int j = jj; j < jEnd; j++)
                        {
                            float* rowW = pW + j * k;
                            float sum = 0f;
                            int t = kk;
                            if (Avx2.IsSupported)
                            {
                                Vector256<float> vSum = Vector256<float>.Zero;
                                int vectorEnd = kEnd - Vector256<float>.Count + 1;
                                for (; t < vectorEnd; t += Vector256<float>.Count)
                                {
                                    Vector256<float> vA = Avx.LoadVector256(rowA + t);
                                    Vector256<float> vW = Avx.LoadVector256(rowW + t);
                                    vSum = Fma.IsSupported ? Fma.MultiplyAdd(vA, vW, vSum)
                                        : Avx.Add(vSum, Avx.Multiply(vA, vW));
                                }
                                Vector128<float> hi = Avx.ExtractVector128(vSum, 1);
                                Vector128<float> lo = vSum.GetLower();
                                Vector128<float> v4 = Sse.Add(lo, hi);
                                Vector128<float> v2 = Sse.Add(v4, Sse.MoveHighToLow(v4, v4));
                                Vector128<float> v1 = Sse.AddScalar(v2, Sse.Shuffle(v2, v2, 1));
                                sum = v1.ToScalar();
                            }
                            for (; t < kEnd; t++) sum += rowA[t] * rowW[t];
                            rowOut[j] += sum;
                        }
                    }
                }
            }
        }
        if (bias is null) return;
        float* b = (float*)bias.DataPointer;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                pOut[i * n + j] += b[j];
    }

    private static Tensor Random(int rows, int cols, int seed)
    {
        Tensor tensor = new(new TensorShape(rows, cols), DType.F32);
        Random rng = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return tensor;
    }
}
