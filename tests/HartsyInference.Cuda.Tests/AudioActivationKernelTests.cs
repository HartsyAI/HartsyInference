using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Backend-parity tests for the audio activation PTX kernels (Sigmoid, Elu, Snake, Snake+beta).
/// Each runs on the CPU reference and the CUDA backend, then compares element-wise. Skips when CUDA
/// is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class AudioActivationKernelTests
{
    private readonly ITestOutputHelper _output;
    public AudioActivationKernelTests(ITestOutputHelper output) => _output = output;

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

    [Fact]
    public void Sigmoid_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = Random(new TensorShape(257), seed: 7, lo: -8f, hi: 8f);
        using Tensor cpuOut = new Tensor(new TensorShape(257), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(257), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.Sigmoid(cpuOut, input);
        cpu.Dispose();

        RunCuda(c => c.Sigmoid(cudaOut, input), cudaOut);
        AssertClose(cpuOut, cudaOut, 1e-5f, "Sigmoid");
    }

    [Fact]
    public void Elu_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = Random(new TensorShape(300), seed: 8, lo: -4f, hi: 4f);
        using Tensor cpuOut = new Tensor(new TensorShape(300), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(300), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.Elu(cpuOut, input, 1.0f);
        cpu.Dispose();

        RunCuda(c => c.Elu(cudaOut, input, 1.0f), cudaOut);
        AssertClose(cpuOut, cudaOut, 1e-5f, "Elu");
    }

    [Fact]
    public void LeakyRelu_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // slope=0.2 is the StyleTTS 2 / Kokoro / HiFi-GAN value.
        using Tensor input = Random(new TensorShape(512), seed: 12, lo: -4f, hi: 4f);
        using Tensor cpuOut = new Tensor(new TensorShape(512), DType.F32);
        using Tensor cudaOut = new Tensor(new TensorShape(512), DType.F32);

        IBackend cpu = new CpuBackend();
        cpu.LeakyRelu(cpuOut, input, 0.2f);
        cpu.Dispose();

        RunCuda(c => c.LeakyRelu(cudaOut, input, 0.2f), cudaOut);
        AssertClose(cpuOut, cudaOut, 1e-6f, "LeakyRelu");
    }

    [Fact]
    public void LeakyRelu_InPlace_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // Kokoro's text encoder calls LeakyRelu in-place (output == input). Verify aliasing is safe.
        using Tensor seed = Random(new TensorShape(257), seed: 13, lo: -4f, hi: 4f);
        using Tensor cpuBuf = new Tensor(new TensorShape(257), DType.F32);
        using Tensor cudaBuf = new Tensor(new TensorShape(257), DType.F32);
        Buffer.MemoryCopy((void*)seed.DataPointer, (void*)cpuBuf.DataPointer, 257 * 4, 257 * 4);
        Buffer.MemoryCopy((void*)seed.DataPointer, (void*)cudaBuf.DataPointer, 257 * 4, 257 * 4);

        IBackend cpu = new CpuBackend();
        cpu.LeakyRelu(cpuBuf, cpuBuf, 0.2f);
        cpu.Dispose();

        RunCuda(c => c.LeakyRelu(cudaBuf, cudaBuf, 0.2f), cudaBuf);
        AssertClose(cpuBuf, cudaBuf, 1e-6f, "LeakyRelu[in-place]");
    }

    [Fact]
    public void AdaInstanceNorm1d_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // [B, C, T] channels-first; gamma/beta either [B,C] (per-batch) or [C] (broadcast).
        foreach (bool perBatch in new[] { true, false })
        {
            const int B = 2, C = 32, T = 97;
            using Tensor input = Random(new TensorShape(B, C, T), seed: 14, lo: -3f, hi: 3f);
            TensorShape affineShape = perBatch ? new TensorShape(B, C) : new TensorShape(C);
            using Tensor gamma = Random(affineShape, seed: 15, lo: -0.8f, hi: 0.8f);
            using Tensor beta = Random(affineShape, seed: 16, lo: -0.8f, hi: 0.8f);
            using Tensor cpuOut = new Tensor(new TensorShape(B, C, T), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(B, C, T), DType.F32);

            IBackend cpu = new CpuBackend();
            cpu.AdaInstanceNorm1d(cpuOut, input, gamma, beta, 1e-5f);
            cpu.Dispose();

            RunCuda(c => c.AdaInstanceNorm1d(cudaOut, input, gamma, beta, 1e-5f), cudaOut);
            AssertClose(cpuOut, cudaOut, 1e-4f, $"AdaInstanceNorm1d[perBatch={perBatch}]");
        }
    }

    [Fact]
    public void Snake_Cpu_Vs_Cuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // [B, C, T], per-channel alpha (strictly positive to avoid the 1/alpha singularity).
        foreach (bool withBeta in new[] { false, true })
        {
            using Tensor input = Random(new TensorShape(1, 16, 48), seed: 9, lo: -2f, hi: 2f);
            using Tensor alpha = Random(new TensorShape(16), seed: 10, lo: 0.3f, hi: 1.7f);
            using Tensor beta = Random(new TensorShape(16), seed: 11, lo: 0.3f, hi: 1.7f);
            Tensor? betaArg = withBeta ? beta : null;
            using Tensor cpuOut = new Tensor(new TensorShape(1, 16, 48), DType.F32);
            using Tensor cudaOut = new Tensor(new TensorShape(1, 16, 48), DType.F32);

            IBackend cpu = new CpuBackend();
            cpu.Snake(cpuOut, input, alpha, betaArg);
            cpu.Dispose();

            RunCuda(c => c.Snake(cudaOut, input, alpha, betaArg), cudaOut);
            AssertClose(cpuOut, cudaOut, 1e-4f, $"Snake[beta={withBeta}]");
        }
    }
}
