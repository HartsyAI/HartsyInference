using HartsyInference.Core.Configuration;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Kernels;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary>Correctness of the GEMM kernels at sizes that actually fan out across cores.
///
/// <para>The rest of the matmul suite tops out at 32x32 — one tile — so it runs entirely on the serial branch
/// and cannot see a bad work split. These shapes are drawn from what the voice pipeline really issues, and the
/// single-row cases matter most: an LLM decoding one token at a time calls
/// <see cref="MatMulKernels.LinearTransB"/> with M = 1, which is precisely the shape a row-only split would
/// leave stranded on one core.</para>
///
/// <para>Compared against a plain triple loop, within a tolerance: the kernel folds an eight-lane vector sum,
/// which associates the reduction differently from a straight left-to-right sum. The tolerance is far tighter
/// than any real division-of-work bug, which drops or double-counts whole terms rather than shifting a few
/// units in the last place.</para></summary>
public sealed unsafe class MatMulParallelKernelTests
{
    /// <summary>M, K, N. The M = 1 rows are decode; the wide-N row is a vocabulary projection.</summary>
    public static TheoryData<int, int, int> Shapes => new()
    {
        { 1, 576, 4096 },      // decode: one token through a feed-forward
        { 1, 576, 49152 },     // decode: one token through the vocabulary projection
        { 64, 576, 1536 },     // prefill: a short prompt
        { 200, 384, 384 },     // whisper encoder self-attention projection
        { 33, 65, 129 },       // deliberately tile-misaligned on every axis
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void LinearTransB_MatchesSerialReference(int m, int k, int n)
    {
        Tensor input = Filled(m, k, seed: 13);
        Tensor weight = Filled(n, k, seed: 27);      // [N, K], PyTorch convention
        Tensor bias = Filled(1, n, seed: 41);
        Tensor output = new(new TensorShape(m, n), DType.F32);
        try
        {
            MatMulKernels.LinearTransB(output, input, weight, bias);

            float* a = (float*)input.DataPointer;
            float* w = (float*)weight.DataPointer;
            float* b = (float*)bias.DataPointer;
            float[] expected = new float[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int kk = 0; kk < k; kk++)
                    {
                        sum += a[i * k + kk] * w[j * k + kk];
                    }
                    expected[i * n + j] = sum + b[j];
                }
            }
            AssertClose(expected, output);
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            bias.Dispose();
            output.Dispose();
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void MatMul_MatchesSerialReference(int m, int k, int n)
    {
        Tensor a = Filled(m, k, seed: 7);
        Tensor b = Filled(k, n, seed: 19);
        Tensor output = new(new TensorShape(m, n), DType.F32);
        try
        {
            MatMulKernels.MatMul(output, a, b);

            float* pa = (float*)a.DataPointer;
            float* pb = (float*)b.DataPointer;
            float[] expected = new float[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int kk = 0; kk < k; kk++)
                    {
                        sum += pa[i * k + kk] * pb[kk * n + j];
                    }
                    expected[i * n + j] = sum;
                }
            }
            AssertClose(expected, output);
        }
        finally
        {
            a.Dispose();
            b.Dispose();
            output.Dispose();
        }
    }

    /// <summary>Capping workers must change only the schedule, never the numbers.</summary>
    [Fact]
    public void LinearTransB_SingleThreadedKnob_ProducesIdenticalOutput()
    {
        int m = 1, k = 576, n = 8192;
        Tensor input = Filled(m, k, seed: 61);
        Tensor weight = Filled(n, k, seed: 67);
        Tensor parallel = new(new TensorShape(m, n), DType.F32);
        Tensor serial = new(new TensorShape(m, n), DType.F32);
        try
        {
            KnobStore.Clear(EngineKnobs.CpuThreads);
            MatMulKernels.LinearTransB(parallel, input, weight, null);
            KnobStore.Set(EngineKnobs.CpuThreads, 1);
            try
            {
                MatMulKernels.LinearTransB(serial, input, weight, null);
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.CpuThreads);
            }

            float* p = (float*)parallel.DataPointer;
            float* s = (float*)serial.DataPointer;
            for (long i = 0; i < parallel.ElementCount; i++)
            {
                Assert.Equal(s[i], p[i]);
            }
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            parallel.Dispose();
            serial.Dispose();
        }
    }

    /// <summary>A rejected shape must arrive as itself, not wrapped by the parallel dispatch.</summary>
    [Fact]
    public void LinearTransB_UndersizedOutput_ThrowsUnwrapped()
    {
        Tensor input = Filled(64, 128, seed: 2);
        Tensor weight = Filled(256, 128, seed: 3);
        Tensor tooSmall = new(new TensorShape(64, 8), DType.F32);
        try
        {
            Assert.Throws<Core.Exceptions.HartsyInferenceException>(() =>
                MatMulKernels.LinearTransB(tooSmall, input, weight, null));
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            tooSmall.Dispose();
        }
    }

    /// <summary>Deterministic pseudo-random fill; a ramp or a constant would hide an indexing error.</summary>
    private static Tensor Filled(int rows, int cols, int seed)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        uint state = (uint)seed * 2654435761u + 1u;
        for (long i = 0; i < t.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            p[i] = ((state >> 8) & 0xFFFF) / 32768f - 1f;
        }
        return t;
    }

    /// <summary>Tolerance rather than equality: the kernel's vector path sums eight lanes and then folds them,
    /// which is a different association from the reference's straight left-to-right sum. The bound is scaled by
    /// the reduction length because that is what bounds the accumulated rounding.</summary>
    private static void AssertClose(float[] expected, Tensor actual)
    {
        float* p = (float*)actual.DataPointer;
        Assert.Equal(expected.Length, (int)actual.ElementCount);
        for (int i = 0; i < expected.Length; i++)
        {
            float tolerance = 1e-4f * Math.Max(1f, Math.Abs(expected[i]));
            if (Math.Abs(expected[i] - p[i]) > tolerance)
            {
                Assert.Fail($"element {i}: expected {expected[i]:R}, got {p[i]:R}");
            }
        }
    }
}
