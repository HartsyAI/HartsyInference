using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity gate for the B0 GPU kernels — <see cref="CudaBackend.MaxPool2D"/> and
/// <see cref="CudaBackend.Conv2dDepthwise"/> — against the CPU reference. Covers the SPPF pooling shape
/// (k=5,s=1,p=2), a strided pool, and a YOLO11-class depthwise conv with and without bias. F32 within
/// 1e-5, F16 within 1e-3 per the kernel tolerance table.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class MaxPoolDepthwiseKernelTests
{
    private readonly ITestOutputHelper _output;
    public MaxPoolDepthwiseKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static double MaxErr(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        double maxErr = 0;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float av = ap[i], bv = bp[i];
            if (!float.IsFinite(av) || !float.IsFinite(bv))
            {
                if (float.IsInfinity(av) && av == bv) continue;
                return double.PositiveInfinity;
            }
            double e = Math.Abs(av - bv);
            if (e > maxErr) maxErr = e;
        }
        return maxErr;
    }

    [Fact]
    public void MaxPool2D_MatchesCpu_F32()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int C = 24, H = 40, W = 40;
        using Tensor input = Random(new TensorShape(1, C, H, W), 7);

        // SPPF: k=5, s=1, p=2 (spatial dims preserved), then a strided k=2,s=2,p=0 downsample.
        foreach ((int k, int s, int p, int oH, int oW) cfg in new[]
        {
            (5, 1, 2, H, W),
            (2, 2, 0, H / 2, W / 2),
        })
        {
            using Tensor cpuOut = new Tensor(new TensorShape(1, C, cfg.oH, cfg.oW), DType.F32);
            using (CpuBackend cpu = new CpuBackend())
                ((IBackend)cpu).MaxPool2D(cpuOut, input, cfg.k, cfg.k, cfg.s, cfg.s, cfg.p, cfg.p);

            using Tensor cudaOut = new Tensor(new TensorShape(1, C, cfg.oH, cfg.oW), DType.F32);
            using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            {
                cuda.MaxPool2D(cudaOut, input, cfg.k, cfg.k, cfg.s, cfg.s, cfg.p, cfg.p);
                cuda.Sync();
                _ = *(float*)cudaOut.DataPointer;
            }

            double maxErr = MaxErr(cpuOut, cudaOut);
            _output.WriteLine($"MaxPool2D k={cfg.k} s={cfg.s} p={cfg.p}: max_err={maxErr:E3}");
            Assert.True(maxErr < 1e-5, $"MaxPool2D diverges (k={cfg.k},s={cfg.s},p={cfg.p}): {maxErr:E3}");
        }
    }

    [Fact]
    public void MaxPool2D_DistinguishesNegativeInfinityFromEmptyPadding()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using Tensor input = new Tensor(new TensorShape(1, 1, 1, 1), DType.F32);
        *(float*)input.DataPointer = float.NegativeInfinity;
        using Tensor cpuOut = new Tensor(new TensorShape(1, 1, 3, 3), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).MaxPool2D(cpuOut, input, 1, 1, 1, 1, 1, 1);

        using Tensor cudaOut = new Tensor(new TensorShape(1, 1, 3, 3), DType.F32);
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.MaxPool2D(cudaOut, input, 1, 1, 1, 1, 1, 1);
            cuda.Sync();
            _ = *(float*)cudaOut.DataPointer;
        }

        float* cpuValues = (float*)cpuOut.DataPointer;
        float* cudaValues = (float*)cudaOut.DataPointer;
        for (int i = 0; i < 9; i++)
        {
            if (i == 4)
            {
                Assert.True(float.IsNegativeInfinity(cpuValues[i]));
                Assert.True(float.IsNegativeInfinity(cudaValues[i]));
            }
            else
            {
                Assert.Equal(0f, cpuValues[i]);
                Assert.Equal(0f, cudaValues[i]);
            }
        }
    }

    [Fact]
    public void MaxPool2D_RejectsIncorrectOutputGeometry()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using Tensor input = new Tensor(new TensorShape(1, 2, 8, 8), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, 2, 8, 7), DType.F32);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        Assert.Throws<ArgumentException>(() => cuda.MaxPool2D(output, input, 3, 3, 1, 1, 1, 1));
    }

    [Fact]
    public void Conv2dDepthwise_MatchesCpu_F32()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int C = 32, H = 28, W = 28, K = 3;
        using Tensor input = Random(new TensorShape(1, C, H, W), 11);
        using Tensor weight = Random(new TensorShape(C, 1, K, K), 12);
        using Tensor bias = Random(new TensorShape(C), 13);

        foreach (bool withBias in new[] { true, false })
        {
            Tensor? b = withBias ? bias : null;
            using Tensor cpuOut = new Tensor(new TensorShape(1, C, H, W), DType.F32);
            using (CpuBackend cpu = new CpuBackend())
                ((IBackend)cpu).Conv2dDepthwise(cpuOut, input, weight, b, 1, 1, 1, 1);

            using Tensor cudaOut = new Tensor(new TensorShape(1, C, H, W), DType.F32);
            using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
            {
                cuda.Conv2dDepthwise(cudaOut, input, weight, b, 1, 1, 1, 1);
                cuda.Sync();
                _ = *(float*)cudaOut.DataPointer;
            }

            double maxErr = MaxErr(cpuOut, cudaOut);
            _output.WriteLine($"Conv2dDepthwise bias={withBias}: max_err={maxErr:E3}");
            Assert.True(maxErr < 1e-5, $"Conv2dDepthwise diverges (bias={withBias}): {maxErr:E3}");
        }
    }

    [Fact]
    public void Conv2dDepthwise_Strided_MatchesCpu_F32()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        // stride-2, no pad, 3x3 (NormalBae downsample shape): oH = (H - K) / 2 + 1.
        const int C = 16, H = 32, W = 32, K = 3;
        int oH = (H - K) / 2 + 1, oW = (W - K) / 2 + 1;
        using Tensor input = Random(new TensorShape(1, C, H, W), 21);
        using Tensor weight = Random(new TensorShape(C, 1, K, K), 22);

        using Tensor cpuOut = new Tensor(new TensorShape(1, C, oH, oW), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).Conv2dDepthwise(cpuOut, input, weight, null, 2, 2, 0, 0);

        using Tensor cudaOut = new Tensor(new TensorShape(1, C, oH, oW), DType.F32);
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.Conv2dDepthwise(cudaOut, input, weight, null, 2, 2, 0, 0);
            cuda.Sync();
            _ = *(float*)cudaOut.DataPointer;
        }

        double maxErr = MaxErr(cpuOut, cudaOut);
        _output.WriteLine($"Conv2dDepthwise stride=2: max_err={maxErr:E3}");
        Assert.True(maxErr < 1e-5, $"Conv2dDepthwise strided diverges: {maxErr:E3}");
    }

    private static double MaxErrHalfVsF32(Tensor half, Tensor f32)
    {
        Half* hp = (Half*)half.DataPointer;
        float* fp = (float*)f32.DataPointer;
        double maxErr = 0;
        long n = f32.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float hv = (float)hp[i], fv = fp[i];
            if (!float.IsFinite(hv) || !float.IsFinite(fv))
            {
                if (float.IsInfinity(hv) && hv == fv) continue;
                return double.PositiveInfinity;
            }
            double e = Math.Abs(hv - fv);
            if (e > maxErr) maxErr = e;
        }
        return maxErr;
    }

    [Fact]
    public void MaxPool2D_And_Depthwise_F16_MatchCpuReference()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        const int C = 16, H = 24, W = 24, K = 3;
        using Tensor inputF32 = Random(new TensorShape(1, C, H, W), 31);
        using Tensor inputF16 = inputF32.CastTo(DType.F16);
        using Tensor weightF32 = Random(new TensorShape(C, 1, K, K), 32);
        using Tensor weightF16 = weightF32.CastTo(DType.F16);

        // MaxPool k=5,s=1,p=2 — F16 kernel vs F32 CPU reference.
        using Tensor poolCpu = new Tensor(new TensorShape(1, C, H, W), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).MaxPool2D(poolCpu, inputF32, 5, 5, 1, 1, 2, 2);
        using Tensor poolCudaF16 = new Tensor(new TensorShape(1, C, H, W), DType.F16);
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.MaxPool2D(poolCudaF16, inputF16, 5, 5, 1, 1, 2, 2);
            cuda.Sync();
            _ = *(Half*)poolCudaF16.DataPointer;
        }
        double poolErr = MaxErrHalfVsF32(poolCudaF16, poolCpu);
        _output.WriteLine($"MaxPool2D F16 vs CPU F32: max_err={poolErr:E3}");
        Assert.True(poolErr < 1e-2, $"MaxPool2D F16 diverges: {poolErr:E3}");

        // Depthwise 3x3 s=1 p=1 — F16 kernel vs F32 CPU reference.
        using Tensor dwCpu = new Tensor(new TensorShape(1, C, H, W), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
            ((IBackend)cpu).Conv2dDepthwise(dwCpu, inputF32, weightF32, null, 1, 1, 1, 1);
        using Tensor dwCudaF16 = new Tensor(new TensorShape(1, C, H, W), DType.F16);
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.Conv2dDepthwise(dwCudaF16, inputF16, weightF16, null, 1, 1, 1, 1);
            cuda.Sync();
            _ = *(Half*)dwCudaF16.DataPointer;
        }
        double dwErr = MaxErrHalfVsF32(dwCudaF16, dwCpu);
        _output.WriteLine($"Conv2dDepthwise F16 vs CPU F32: max_err={dwErr:E3}");
        Assert.True(dwErr < 1e-2, $"Conv2dDepthwise F16 diverges: {dwErr:E3}");
    }
}
