using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validation harness for the hand-written tensor-core HGEMM (<c>hgemm_mma_sm80.ptx</c>). The kernel's PTX fragment layout is authored from documentation and unverified on hardware, so these tests are the gate: they diff the <c>EnableTensorCoreGemm</c> path against the cuBLAS reference through the exact same <see cref="CudaBackend.Linear"/> call. A passing run on an SM 8.0+ GPU is what makes the kernel trustworthy. CPU-only static-logic tests always run.</summary>
[Collection("CudaSerial")]
public sealed class TensorCoreGemmTests
{
    private readonly ITestOutputHelper _output;
    public TensorCoreGemmTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(16, 8, 16, true)]
    [InlineData(32, 64, 32, true)]
    [InlineData(17, 8, 16, false)]   // M not a multiple of 16
    [InlineData(16, 12, 16, false)]  // N not a multiple of 8
    [InlineData(16, 8, 24, false)]   // K not a multiple of 16
    public void IsAligned_GatesOnDimensionMultiples(int m, int n, int k, bool expected)
    {
        Assert.Equal(expected, TensorCoreGemm.IsAligned(m, n, k));
    }

    [Theory]
    [InlineData(16, 8, 16)]
    [InlineData(64, 64, 64)]
    [InlineData(128, 320, 64)]
    public unsafe void TensorCoreLinear_MatchesCublasReference(int m, int n, int k)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (backend.Context.ComputeCapabilityMajor < 8)
        {
            _output.WriteLine("SKIPPED: tensor-core HGEMM requires SM 8.0+");
            return;
        }

        // input [M,K], weight [N,K] (Linear weight); output [M,N], all F16.
        Tensor input = new Tensor(new TensorShape(m, k), DType.F16);
        Tensor weight = new Tensor(new TensorShape(n, k), DType.F16);
        Tensor outTc = new Tensor(new TensorShape(m, n), DType.F16);
        Tensor outRef = new Tensor(new TensorShape(m, n), DType.F16);
        try
        {
            Random rng = new Random(7);
            Half* ip = (Half*)input.DataPointer;
            for (long i = 0; i < (long)m * k; i++) ip[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            Half* wp = (Half*)weight.DataPointer;
            for (long i = 0; i < (long)n * k; i++) wp[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.3);

            backend.EnableTensorCoreGemm = true;
            backend.Linear(outTc, input, weight, bias: null);
            backend.Sync();

            backend.EnableTensorCoreGemm = false;
            backend.Linear(outRef, input, weight, bias: null);
            backend.Sync();

            Half* tcp = (Half*)outTc.DataPointer;
            Half* rp = (Half*)outRef.DataPointer;
            double sumAbs = 0.0;
            float maxAbs = 0f;
            long total = (long)m * n;
            for (long i = 0; i < total; i++)
            {
                float err = MathF.Abs((float)tcp[i] - (float)rp[i]);
                sumAbs += err;
                if (err > maxAbs) maxAbs = err;
            }
            float avgErr = (float)(sumAbs / total);
            _output.WriteLine($"TC HGEMM {m}x{n}x{k}: avg_err={avgErr:E3}, max_err={maxAbs:E3}");
            // F16 GEMM tolerance; both paths accumulate in F32 so they should be close.
            Assert.True(avgErr < 0.05f, $"Tensor-core HGEMM avg_err {avgErr:E3} exceeds tolerance (max_err={maxAbs:E3}). The PTX fragment layout is likely wrong.");
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            outTc.Dispose();
            outRef.Dispose();
        }
    }

    [Fact]
    public unsafe void TensorCore_UnalignedShape_FallsBackToCublas()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);

        // 17x10x24 is unaligned on all three dims, so EnableTensorCoreGemm must fall
        // through to cuBLAS and still produce the correct result.
        int m = 17, n = 10, k = 24;
        Tensor input = new Tensor(new TensorShape(m, k), DType.F16);
        Tensor weight = new Tensor(new TensorShape(n, k), DType.F16);
        Tensor outOn = new Tensor(new TensorShape(m, n), DType.F16);
        Tensor outOff = new Tensor(new TensorShape(m, n), DType.F16);
        try
        {
            Random rng = new Random(11);
            Half* ip = (Half*)input.DataPointer;
            for (long i = 0; i < (long)m * k; i++) ip[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            Half* wp = (Half*)weight.DataPointer;
            for (long i = 0; i < (long)n * k; i++) wp[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.3);

            backend.EnableTensorCoreGemm = true;
            backend.Linear(outOn, input, weight, bias: null);
            backend.Sync();
            backend.EnableTensorCoreGemm = false;
            backend.Linear(outOff, input, weight, bias: null);
            backend.Sync();

            Half* a = (Half*)outOn.DataPointer;
            Half* b = (Half*)outOff.DataPointer;
            for (long i = 0; i < (long)m * n; i++)
                Assert.Equal((float)b[i], (float)a[i], 3);
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            outOn.Dispose();
            outOff.Dispose();
        }
    }
}
