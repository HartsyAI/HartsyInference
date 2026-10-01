using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Kernels;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary><see cref="MatMulKernels.LinearTransB"/> computes four weight rows at a time. That restructuring is
/// meant to be invisible: every CPU model's outputs, digests included, rest on this kernel, so it must produce the
/// same bits as the one-row loop it replaced, not merely close values. The reference below is that loop verbatim
/// (serially — the tile split never changes how one element is summed), compared byte for byte across shapes that
/// leave every remainder: rows not a multiple of four, K tails inside the 8-lane and 32-wide tiles, and bias on and
/// off. On a CPU without FMA the kernel takes the multiply-then-add branch; this box has FMA, so that branch is not
/// exercised here.</summary>
public sealed unsafe class LinearTransBIdentityTests
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
