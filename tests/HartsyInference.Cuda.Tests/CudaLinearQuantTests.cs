using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.Gguf.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>End-to-end <see cref="CudaBackend.Linear"/> tests with a quantized weight. Verifies the wiring is complete: <see cref="GpuTransferHelper.ByteSize"/> reports the right bytes for Q*_K, GpuTransferHelper uploads quantized bytes correctly, <see cref="CudaBackend.Linear"/> resolves the GEMM dtype to F16, <see cref="CudaBackend.CastIfNeeded"/> dispatches the GPU dequant kernel, and the cuBLAS GEMM produces an answer comparable to the same operation with the dequantized F16 weight.</summary>
[Collection("CudaSerial")]
public sealed class CudaLinearQuantTests
{
    private readonly ITestOutputHelper _output;
    public CudaLinearQuantTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Linear_Q8_0_Weight_MatchesF16Reference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunLinearQuantTest(DType.Q8_0, tolerance: 0.05f, batch: 4, inDim: 64, outDim: 128);
    }

    [Fact]
    public unsafe void Linear_Q4_K_Weight_MatchesF16Reference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunLinearQuantTest(DType.Q4_K, tolerance: 0.5f, batch: 4, inDim: 256, outDim: 256);
    }

    [Fact]
    public unsafe void Linear_Q6_K_Weight_MatchesF16Reference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        RunLinearQuantTest(DType.Q6_K, tolerance: 0.1f, batch: 4, inDim: 256, outDim: 256);
    }

    private unsafe void RunLinearQuantTest(DType quantDtype, float tolerance, int batch, int inDim, int outDim)
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);

        long inputCount = (long)batch * inDim;
        long weightCount = (long)outDim * inDim;
        Tensor input = new Tensor(new TensorShape(batch, inDim), DType.F16);
        Tensor weightF32 = new Tensor(new TensorShape(outDim, inDim), DType.F32);
        Tensor weightQuant = new Tensor(new TensorShape(outDim, inDim), quantDtype);
        Tensor weightF16Reference = null!;
        Tensor outputQuant = new Tensor(new TensorShape(batch, outDim), DType.F16);
        Tensor outputReference = new Tensor(new TensorShape(batch, outDim), DType.F16);

        try
        {
            Random rng = new Random(42);
            Half* ip = (Half*)input.DataPointer;
            for (long i = 0; i < inputCount; i++) ip[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.5);

            float* wf32p = (float*)weightF32.DataPointer;
            for (long i = 0; i < weightCount; i++) wf32p[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.3);

            IGgufCodec codec = GgufCodecRegistry.Get(quantDtype);
            codec.QuantizeFromF32(wf32p, (byte*)weightQuant.DataPointer, weightCount);

            weightF16Reference = GgufDequantizer.Dequantize(weightQuant, DType.F16);

            backend.Linear(outputQuant, input, weightQuant, bias: null);
            backend.Sync();

            backend.Linear(outputReference, input, weightF16Reference, bias: null);
            backend.Sync();

            Half* qp = (Half*)outputQuant.DataPointer;
            Half* rp = (Half*)outputReference.DataPointer;
            double sumAbs = 0.0;
            float maxAbs = 0f;
            long total = (long)batch * outDim;
            for (long i = 0; i < total; i++)
            {
                float qv = (float)qp[i];
                float rv = (float)rp[i];
                float err = MathF.Abs(qv - rv);
                sumAbs += err;
                if (err > maxAbs) maxAbs = err;
            }
            float avgErr = (float)(sumAbs / total);
            _output.WriteLine($"{quantDtype} Linear: avg_err={avgErr:E3}, max_err={maxAbs:E3}, tolerance={tolerance:E3}");
            Assert.True(avgErr < tolerance, $"{quantDtype} Linear avg_err {avgErr:E3} exceeds tolerance {tolerance:E3}; max_err = {maxAbs:E3}");
        }
        finally
        {
            input.Dispose();
            weightF32.Dispose();
            weightQuant.Dispose();
            weightF16Reference?.Dispose();
            outputQuant.Dispose();
            outputReference.Dispose();
        }
    }
}
