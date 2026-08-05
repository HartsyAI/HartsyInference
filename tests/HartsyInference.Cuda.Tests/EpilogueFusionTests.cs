using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates that the cuBLASLt epilogue bias-fusion path in <see cref="CudaBackend.Linear"/> (<see cref="CudaBackend.EnableEpilogueFusion"/>) produces the same result as the unfused <c>cublasGemmEx</c> + separate <c>BiasAdd</c> path.</summary>
[Collection("CudaSerial")]
public sealed class EpilogueFusionTests
{
    private readonly ITestOutputHelper _output;
    public EpilogueFusionTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(4, 128, 64)]
    [InlineData(16, 320, 128)]
    public unsafe void EpilogueBiasFusion_MatchesUnfusedReference(int m, int outDim, int inDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);

        Tensor input = new Tensor(new TensorShape(m, inDim), DType.F16);
        Tensor weight = new Tensor(new TensorShape(outDim, inDim), DType.F16);
        Tensor bias = new Tensor(new TensorShape(outDim), DType.F16);
        Tensor outFused = new Tensor(new TensorShape(m, outDim), DType.F16);
        Tensor outRef = new Tensor(new TensorShape(m, outDim), DType.F16);
        try
        {
            Random rng = new Random(13);
            Half* ip = (Half*)input.DataPointer;
            for (long i = 0; i < (long)m * inDim; i++) ip[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            Half* wp = (Half*)weight.DataPointer;
            for (long i = 0; i < (long)outDim * inDim; i++) wp[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.3);
            Half* bp = (Half*)bias.DataPointer;
            for (long i = 0; i < outDim; i++) bp[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.2);

            backend.EnableEpilogueFusion = true;
            backend.Linear(outFused, input, weight, bias);
            backend.Sync();

            backend.EnableEpilogueFusion = false;
            backend.Linear(outRef, input, weight, bias);
            backend.Sync();

            Half* fp = (Half*)outFused.DataPointer;
            Half* rp = (Half*)outRef.DataPointer;
            double sumAbs = 0.0;
            float maxAbs = 0f;
            long total = (long)m * outDim;
            for (long i = 0; i < total; i++)
            {
                float err = MathF.Abs((float)fp[i] - (float)rp[i]);
                sumAbs += err;
                if (err > maxAbs) maxAbs = err;
            }
            float avgErr = (float)(sumAbs / total);
            _output.WriteLine($"Epilogue fusion {m}x{outDim}x{inDim}: avg_err={avgErr:E3}, max_err={maxAbs:E3}");
            // Bias fusion is the same arithmetic; allow only tiny F16-rounding differences.
            Assert.True(avgErr < 1e-2f, $"Epilogue-fused Linear avg_err {avgErr:E3} exceeds tolerance (max_err={maxAbs:E3}).");
        }
        finally
        {
            input.Dispose();
            weight.Dispose();
            bias.Dispose();
            outFused.Dispose();
            outRef.Dispose();
        }
    }

    /// <summary>F32 operands writing a 16-bit output: cuBLASLt cannot produce a D type unrelated to its operand
    /// type, and the bias epilogue returned ~2.6e38 instead of failing, so the fused path must decline this shape
    /// the way the plain cuBLAS branch already does (GEMM into an operand-dtype temp, then cast). Checks against a
    /// host reference, not just fused-vs-unfused — after the fix both take one path, so agreement alone would not
    /// prove the values are real.</summary>
    [Theory]
    [InlineData("BF16")]
    [InlineData("F16")]
    public unsafe void EpilogueBiasFusion_F32Operands_SixteenBitOutput_IsNotGarbage(string outDtype)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int M = 8, OutDim = 96, InDim = 64;
        DType outType = outDtype == "F16" ? DType.F16 : DType.BF16;
        using CudaBackend backend = new CudaBackend(0, ptxDir);
        using Tensor input = new Tensor(new TensorShape(M, InDim), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(OutDim, InDim), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(OutDim), DType.F32);
        using Tensor output = new Tensor(new TensorShape(M, OutDim), outType);

        Random rng = new Random(7);
        float* ip = (float*)input.DataPointer;
        for (long i = 0; i < (long)M * InDim; i++) ip[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f;
        float* wp = (float*)weight.DataPointer;
        for (long i = 0; i < (long)OutDim * InDim; i++) wp[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.3f;
        float* bp = (float*)bias.DataPointer;
        for (long i = 0; i < OutDim; i++) bp[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.2f;

        backend.EnableEpilogueFusion = true;
        backend.Linear(output, input, weight, bias);
        backend.Sync();

        double maxErr = 0;
        long maxIdx = 0;
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < OutDim; n++)
            {
                double expected = bp[n];
                for (int k = 0; k < InDim; k++) expected += (double)ip[m * InDim + k] * wp[n * InDim + k];
                long idx = (long)m * OutDim + n;
                float actual = outType == DType.F16
                    ? (float)((Half*)output.DataPointer)[idx]
                    : BitConverter.Int32BitsToSingle(((ushort*)output.DataPointer)[idx] << 16);
                double err = Math.Abs(expected - actual);
                if (err > maxErr) { maxErr = err; maxIdx = idx; }
            }
        }
        _output.WriteLine($"Linear F32→{outDtype} with fused bias: max_err={maxErr:E3} @ {maxIdx}");
        Assert.True(maxErr < 5e-2, $"Linear F32 operands → {outDtype} output max_err {maxErr:E3} (index {maxIdx}).");
    }
}
