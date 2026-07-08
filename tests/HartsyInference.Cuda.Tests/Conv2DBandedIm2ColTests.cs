using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity test for Conv2D's banded-im2col path: with HARTSY_IM2COL_BAND_MB=1 a modest conv's col
/// workspace exceeds the cap and runs as many output-row bands; the result must match the CPU reference
/// exactly like the unbanded path does. Runs in its own collection because the cap is a static readonly read
/// once per process — it must be set before the first CudaBackend touch, so run this test standalone
/// (dotnet test --filter Conv2DBandedIm2Col) when validating the banding change.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class Conv2DBandedIm2ColTests
{
    private readonly ITestOutputHelper _output;
    public Conv2DBandedIm2ColTests(ITestOutputHelper output) => _output = output;

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

    [Fact]
    public void Conv2D_BandedIm2Col_MatchesCpu()
    {
        // Pre-set HARTSY_IM2COL_BAND_MB (e.g. 99999) to compare the unbanded path's noise floor.
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("HARTSY_IM2COL_BAND_MB")))
            Environment.SetEnvironmentVariable("HARTSY_IM2COL_BAND_MB", "1");
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        // col = 32ch · 9 · 96·96 · 4B ≈ 10.1 MB per image ⇒ ~11 bands at the 1 MB cap. Odd outH vs bandRows
        // exercises the short last band; stride/pad cover the VAE conv shape (3×3, stride 1, pad 1).
        const int inCh = 32, outCh = 16, H = 96, W = 96;
        using Tensor input = Random(new TensorShape(1, inCh, H, W), 42);
        using Tensor weight = Random(new TensorShape(outCh, inCh, 3, 3), 43);
        using Tensor bias = Random(new TensorShape(outCh), 44);

        Tensor cpuOut = new Tensor(new TensorShape(1, outCh, H, W), DType.F32);
        using (CpuBackend cpu = new CpuBackend())
        {
            cpu.Conv2D(cpuOut, input, weight, bias, 1, 1, 1, 1);
        }

        Tensor cudaOut = new Tensor(new TensorShape(1, outCh, H, W), DType.F32);
        using (CudaBackend cuda = new CudaBackend(0, PtxDir()))
        {
            cuda.Conv2D(cudaOut, input, weight, bias, 1, 1, 1, 1);
            cuda.Sync();
            _ = *(float*)cudaOut.DataPointer;
        }

        float* cp = (float*)cpuOut.DataPointer;
        float* gp = (float*)cudaOut.DataPointer;
        double maxErr = 0;
        long n = cpuOut.ElementCount;
        for (long i = 0; i < n; i++)
        {
            double e = Math.Abs(cp[i] - gp[i]);
            if (e > maxErr) maxErr = e;
        }
        _output.WriteLine($"banded Conv2D vs CPU: max_err={maxErr:E3} over {n} elements");
        cpuOut.Dispose();
        cudaOut.Dispose();
        Environment.SetEnvironmentVariable("HARTSY_IM2COL_BAND_MB", null);
        // TF32 GEMM tolerance vs CPU F32: measured noise floor at this shape is 6.48e-3 for BOTH the banded
        // and unbanded paths (identical to 1e-6 — banding itself is exact); bound leaves 2× headroom.
        Assert.True(maxErr < 1.5e-2, $"banded Conv2D diverges from CPU: max_err {maxErr:E3}");
    }
}
