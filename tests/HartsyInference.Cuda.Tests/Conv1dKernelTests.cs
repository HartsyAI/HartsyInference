using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Backend-parity tests for the audio Conv1d / ConvTranspose1d PTX kernels. Each shape runs on
/// the CPU backend (the numerical reference) and the CUDA backend, then compares element-wise. Covers the
/// codec/TTS cases that matter: stride, dilation, groups, asymmetric (causal) padding, and bias/no-bias.
/// Skips cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Conv1dKernelTests
{
    private readonly ITestOutputHelper _output;
    public Conv1dKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    private static void RunCuda(Action<CudaBackend> op, params Tensor[] outputs)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        op(cuda);
        cuda.Sync();
        foreach (Tensor t in outputs)
            _ = *(float*)t.DataPointer;
    }

    private void AssertClose(Tensor cpu, Tensor cuda, float tol, string name)
    {
        float* a = (float*)cpu.DataPointer;
        float* b = (float*)cuda.DataPointer;
        long n = cpu.ElementCount;
        double maxErr = 0;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(a[i] - b[i]);
            if (e > maxErr) maxErr = e;
        }
        _output.WriteLine($"{name}: max_err={maxErr:E3} over {n} elems (tol {tol:E0})");
        Assert.True(maxErr < tol, $"{name}: max_err {maxErr:E3} exceeds tol {tol:E0}");
    }

    private static int Conv1dTOut(int tIn, int padLeft, int padRight, int dilation, int kernel, int stride)
        => (tIn + padLeft + padRight - dilation * (kernel - 1) - 1) / stride + 1;

    [Fact]
    public void Conv1d_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        // (batch, cIn, tIn, cOut, kernel, stride, padL, padR, dilation, groups, hasBias)
        (int b, int cIn, int tIn, int cOut, int k, int s, int pl, int pr, int dil, int g, bool bias)[] cases =
        {
            (1, 8, 64, 16, 3, 1, 1, 1, 1, 1, true),    // plain same-length conv
            (2, 16, 50, 16, 7, 1, 3, 3, 1, 1, false),  // wider kernel, no bias
            (1, 32, 128, 64, 4, 2, 1, 1, 1, 1, true),  // strided downsample (codec encoder)
            (1, 16, 80, 16, 3, 1, 2, 2, 2, 1, true),   // dilated
            (1, 32, 40, 32, 3, 1, 1, 1, 1, 32, false), // depthwise (groups == channels)
            (1, 24, 33, 12, 5, 1, 4, 0, 1, 3, true),   // grouped + causal (asymmetric) padding
        };

        foreach ((int b, int cIn, int tIn, int cOut, int k, int s, int pl, int pr, int dil, int g, bool bias) c in cases)
        {
            int tOut = Conv1dTOut(c.tIn, c.pl, c.pr, c.dil, c.k, c.s);
            using Tensor input = Random(new TensorShape(c.b, c.cIn, c.tIn), seed: 11 + c.tIn);
            using Tensor weight = Random(new TensorShape(c.cOut, c.cIn / c.g, c.k), seed: 22 + c.cOut);
            using Tensor bias = Random(new TensorShape(c.cOut), seed: 33 + c.cOut);
            Tensor? biasArg = c.bias ? bias : null;
            using Tensor cpuOut = new Tensor(new TensorShape(c.b, c.cOut, tOut), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(c.b, c.cOut, tOut), DType.F32);

            IBackend cpu = new CpuBackend();
            cpu.Conv1d(cpuOut, input, weight, biasArg, c.s, c.pl, c.pr, c.dil, c.g);
            cpu.Dispose();

            RunCuda(x => x.Conv1d(cudaOut, input, weight, biasArg, c.s, c.pl, c.pr, c.dil, c.g), cudaOut);

            AssertClose(cpuOut, cudaOut, 1e-4f,
                $"Conv1d[b{c.b} cIn{c.cIn} t{c.tIn} cOut{c.cOut} k{c.k} s{c.s} d{c.dil} g{c.g} bias{c.bias}]");
        }
    }

    [Fact]
    public void ConvTranspose1d_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        // ConvTranspose1d weight is [C_in, C_out, K]. tOut = (tIn-1)*stride - padL - padR + dilation*(k-1) + 1.
        // (batch, cIn, tIn, cOut, kernel, stride, padL, padR, dilation, hasBias)
        (int b, int cIn, int tIn, int cOut, int k, int s, int pl, int pr, int dil, bool bias)[] cases =
        {
            (1, 16, 32, 8, 4, 2, 1, 1, 1, true),   // 2x upsample (codec decoder)
            (1, 8, 20, 8, 3, 1, 1, 1, 1, false),   // stride 1, no bias
            (2, 32, 25, 16, 6, 3, 2, 1, 1, true),  // 3x upsample, asymmetric pad
        };

        foreach ((int b, int cIn, int tIn, int cOut, int k, int s, int pl, int pr, int dil, bool bias) c in cases)
        {
            int tOut = (c.tIn - 1) * c.s - c.pl - c.pr + c.dil * (c.k - 1) + 1;
            using Tensor input = Random(new TensorShape(c.b, c.cIn, c.tIn), seed: 44 + c.tIn);
            using Tensor weight = Random(new TensorShape(c.cIn, c.cOut, c.k), seed: 55 + c.cOut);
            using Tensor bias = Random(new TensorShape(c.cOut), seed: 66 + c.cOut);
            Tensor? biasArg = c.bias ? bias : null;
            using Tensor cpuOut = new Tensor(new TensorShape(c.b, c.cOut, tOut), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(c.b, c.cOut, tOut), DType.F32);

            IBackend cpu = new CpuBackend();
            cpu.ConvTranspose1d(cpuOut, input, weight, biasArg, c.s, c.pl, c.pr, c.dil, 1);
            cpu.Dispose();

            RunCuda(x => x.ConvTranspose1d(cudaOut, input, weight, biasArg, c.s, c.pl, c.pr, c.dil, 1), cudaOut);

            AssertClose(cpuOut, cudaOut, 1e-4f,
                $"ConvT1d[b{c.b} cIn{c.cIn} t{c.tIn} cOut{c.cOut} k{c.k} s{c.s} d{c.dil} bias{c.bias}]");
        }
    }
}
