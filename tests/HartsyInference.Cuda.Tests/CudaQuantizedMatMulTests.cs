using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Parity for the low-VRAM <see cref="CudaBackend.QuantizedMatMul"/> path: it must match the
/// dequant-to-F16 <see cref="CudaBackend.Linear"/> on the same quantized weight (they share the GEMM; the only
/// difference is QuantizedMatMul keeps the weight compressed and frees the dequant transiently rather than
/// caching an F16 copy). Output should be bit-for-bit equal.</summary>
[Collection("CudaSerial")]
public sealed class CudaQuantizedMatMulTests
{
    private readonly ITestOutputHelper _output;
    public CudaQuantizedMatMulTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("Q8_0", 64, 128)]
    [InlineData("Q4_K", 256, 256)]
    [InlineData("Q6_K", 256, 256)]
    public unsafe void QuantizedMatMul_MatchesLinear(string dtypeName, int inDim, int outDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        DType quantDtype = dtypeName switch
        {
            "Q8_0" => DType.Q8_0,
            "Q4_K" => DType.Q4_K,
            "Q6_K" => DType.Q6_K,
            _ => throw new ArgumentOutOfRangeException(nameof(dtypeName)),
        };

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int batch = 4;
        using CudaBackend backend = new(0, ptxDir);
        Tensor input = new(new TensorShape(batch, inDim), DType.F32);
        Tensor weightF32 = new(new TensorShape(outDim, inDim), DType.F32);
        Tensor weightQuant = new(new TensorShape(outDim, inDim), quantDtype);
        Tensor outLinear = new(new TensorShape(batch, outDim), DType.F32);
        Tensor outQmm = new(new TensorShape(batch, outDim), DType.F32);
        try
        {
            Random rng = new(7);
            float* ip = (float*)input.DataPointer;
            for (long i = 0; i < (long)batch * inDim; i++) ip[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            float* wp = (float*)weightF32.DataPointer;
            long wc = (long)outDim * inDim;
            for (long i = 0; i < wc; i++) wp[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.3);
            GgufCodecRegistry.Get(quantDtype).QuantizeFromF32(wp, (byte*)weightQuant.DataPointer, wc);

            backend.Linear(outLinear, input, weightQuant, bias: null);
            backend.QuantizedMatMul(outQmm, input, weightQuant, bias: null);
            backend.Sync();

            float* a = (float*)outLinear.DataPointer;
            float* b = (float*)outQmm.DataPointer;
            float maxDiff = 0f;
            for (long i = 0; i < (long)batch * outDim; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - b[i]));
            _output.WriteLine($"{dtypeName}: max |Linear - QuantizedMatMul| = {maxDiff:E3}");
            Assert.True(maxDiff <= 1e-4f, $"{dtypeName} QuantizedMatMul diverges from Linear by {maxDiff:E3}");
        }
        finally
        {
            input.Dispose(); weightF32.Dispose(); weightQuant.Dispose(); outLinear.Dispose(); outQmm.Dispose();
        }
    }
}
