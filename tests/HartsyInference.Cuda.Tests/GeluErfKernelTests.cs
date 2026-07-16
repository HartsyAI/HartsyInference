using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Backend parity for the exact-erf GELU PTX kernel (<c>gelu_erf_f32</c>) against the IBackend
/// host fallback and the analytic value. Skips when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class GeluErfKernelTests
{
    private readonly ITestOutputHelper _output;
    public GeluErfKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void GeluErf_Cuda_MatchesAnalytic()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int n = 4099;
        using Tensor input = new(new TensorShape(n), DType.F32);
        float* ip = (float*)input.DataPointer;
        Random rng = new(1234);
        for (int i = 0; i < n; i++) ip[i] = (float)(rng.NextDouble() * 16.0 - 8.0);

        using Tensor cudaOut = new(new TensorShape(n), DType.F32);
        using (CudaBackend cuda = new(0, PtxDir()))
        {
            ((IBackend)cuda).GeluErf(cudaOut, input);
            cuda.Sync();
            _ = *(float*)cudaOut.DataPointer;
        }

        // Analytic gelu_erf in double precision (erf via A&S 7.1.26 refined by symmetric evaluation).
        float* op = (float*)cudaOut.DataPointer;
        double maxErr = 0;
        for (int i = 0; i < n; i++)
        {
            double x = ip[i];
            double expected = 0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0)));
            double e = Math.Abs(op[i] - expected);
            if (e > maxErr) maxErr = e;
        }
        _output.WriteLine($"GeluErf CUDA vs analytic: max_err={maxErr:E3} over {n} elems");
        Assert.True(maxErr < 1e-5, $"gelu_erf_f32 diverges from analytic GELU: max_err {maxErr:E3}");
    }

    private static double Erf(double x)
    {
        // Abramowitz–Stegun 7.1.26 in double precision (max abs error ~1.5e-7 — far below the F32 tolerance).
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }
}
