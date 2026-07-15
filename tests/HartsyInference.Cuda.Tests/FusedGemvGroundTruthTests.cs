using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.Gguf.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Ground-truth correctness for the new Q4_0/Q5_K fused GEMV kernels (decode path,
/// M small): quantizes a random F32 weight, dequantizes it back on the CPU (canonical codec,
/// independent of the GPU kernel under test), computes the reference matmul in plain C#, and
/// compares against <see cref="CudaBackend.Linear"/> — which for these dtypes at M&lt;=8 now
/// dispatches to the fused GEMV kernel. Unlike <see cref="CudaQuantizedMatMulTests"/> (which only
/// checks that two entry points sharing the same fused-path code agree with each other), this
/// compares against an independently-computed reference.</summary>
[Collection("CudaSerial")]
public sealed class FusedGemvGroundTruthTests
{
    private readonly ITestOutputHelper _output;
    public FusedGemvGroundTruthTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("Q4_0", 128, 96)]
    [InlineData("Q5_K", 512, 32)]
    public unsafe void FusedGemv_MatchesCpuDequantReference(string dtypeName, int inDim, int outDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        DType quantDtype = dtypeName switch
        {
            "Q4_0" => DType.Q4_0,
            "Q5_K" => DType.Q5_K,
            _ => throw new ArgumentOutOfRangeException(nameof(dtypeName)),
        };

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int batch = 4; // M <= 8 so CudaBackend.Linear dispatches to the new fused GEMV branch.
        using CudaBackend backend = new(0, ptxDir);
        Tensor input = new(new TensorShape(batch, inDim), DType.F32);
        Tensor weightF32 = new(new TensorShape(outDim, inDim), DType.F32);
        Tensor weightQuant = new(new TensorShape(outDim, inDim), quantDtype);
        Tensor outGpu = new(new TensorShape(batch, outDim), DType.F32);
        try
        {
            Random rng = new(11);
            float* ip = (float*)input.DataPointer;
            for (long i = 0; i < (long)batch * inDim; i++) ip[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            if (quantDtype == DType.Q4_0)
            {
                // Q4_0's codec is read-only (no F32->Q4_0 quantize path — see GgufCodecBase.QuantizeFromF32).
                // Synthesize raw quantized bytes directly instead: random per-block scale + nibbles.
                FillQ4_0Random(weightQuant, blocks: (int)((long)outDim * inDim / 32), rng);
            }
            else
            {
                float* wp = (float*)weightF32.DataPointer;
                long wc = (long)outDim * inDim;
                for (long i = 0; i < wc; i++) wp[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.3);
                GgufCodecRegistry.Get(quantDtype).QuantizeFromF32(wp, (byte*)weightQuant.DataPointer, wc);
            }

            backend.Linear(outGpu, input, weightQuant, bias: null);
            backend.Sync();

            using Tensor weightDequant = GgufDequantizer.Dequantize(weightQuant, DType.F32);
            float* dw = (float*)weightDequant.DataPointer;
            float* cpuRef = stackalloc float[batch * outDim];
            for (int m = 0; m < batch; m++)
            {
                for (int n = 0; n < outDim; n++)
                {
                    float acc = 0f;
                    float* xrow = ip + (long)m * inDim;
                    float* wrow = dw + (long)n * inDim;
                    for (int k = 0; k < inDim; k++) acc += xrow[k] * wrow[k];
                    cpuRef[m * outDim + n] = acc;
                }
            }

            float* gpuOut = (float*)outGpu.DataPointer;
            double sumAbs = 0.0;
            float maxAbs = 0f;
            for (int i = 0; i < batch * outDim; i++)
            {
                float err = MathF.Abs(cpuRef[i] - gpuOut[i]);
                sumAbs += err;
                if (err > maxAbs) maxAbs = err;
            }
            float avgErr = (float)(sumAbs / (batch * outDim));
            _output.WriteLine($"{dtypeName}: avg_err={avgErr:E3} max_err={maxAbs:E3}");
            // Tolerance matches the existing GPU-dequant parity tests (F16-scale quant error, F32 accumulate).
            Assert.True(avgErr < 5e-3f, $"{dtypeName} fused GEMV avg_err {avgErr:E3} exceeds tolerance; max_err={maxAbs:E3}");
        }
        finally
        {
            input.Dispose(); weightF32.Dispose(); weightQuant.Dispose(); outGpu.Dispose();
        }
    }

    /// <summary>Dense BF16/F16 fused GEMV (the mul_mat_vec_bf16/f16_f32 kernels): the CPU reference uses the
    /// 16-bit-ROUNDED weight (what the kernel reads after bf16/f16→f32) against the F32 activation, so the only
    /// residual is F32 accumulation order → a tight tolerance. Covers a non-multiple-of-32 K to exercise the
    /// strided-lane tail. M≤8 so <see cref="CudaBackend.Linear"/> takes the fused GEMV branch.</summary>
    [Theory]
    [InlineData("BF16", 3072, 128)]
    [InlineData("BF16", 100, 48)]
    [InlineData("F16", 512, 64)]
    public unsafe void FusedGemv_Float16_MatchesRoundedReference(string dtypeName, int inDim, int outDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        DType dt = dtypeName == "BF16" ? DType.BF16 : DType.F16;
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int batch = 4;
        using CudaBackend backend = new(0, ptxDir);
        Tensor input = new(new TensorShape(batch, inDim), DType.F32);
        Tensor weightF32 = new(new TensorShape(outDim, inDim), DType.F32);
        Tensor outGpu = new(new TensorShape(batch, outDim), DType.F32);
        try
        {
            Random rng = new(7);
            float* ip = (float*)input.DataPointer;
            for (long i = 0; i < (long)batch * inDim; i++) ip[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.5);
            float* wp = (float*)weightF32.DataPointer;
            long wc = (long)outDim * inDim;
            for (long i = 0; i < wc; i++) wp[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.3);

            using Tensor weight16 = weightF32.CastTo(dt);          // what the kernel actually multiplies
            using Tensor weightRounded = weight16.CastTo(DType.F32);  // same values, back in F32 for the reference

            backend.Linear(outGpu, input, weight16, bias: null);
            backend.Sync();

            float* rw = (float*)weightRounded.DataPointer;
            float* gpuOut = (float*)outGpu.DataPointer;
            double sumAbs = 0.0; float maxAbs = 0f;
            for (int m = 0; m < batch; m++)
                for (int n = 0; n < outDim; n++)
                {
                    float acc = 0f;
                    float* xrow = ip + (long)m * inDim;
                    float* wrow = rw + (long)n * inDim;
                    for (int k = 0; k < inDim; k++) acc += xrow[k] * wrow[k];
                    float err = MathF.Abs(acc - gpuOut[m * outDim + n]);
                    sumAbs += err; if (err > maxAbs) maxAbs = err;
                }
            float avgErr = (float)(sumAbs / (batch * outDim));
            _output.WriteLine($"{dtypeName} K={inDim} N={outDim}: avg_err={avgErr:E3} max_err={maxAbs:E3}");
            Assert.True(avgErr < 1e-4f, $"{dtypeName} fused GEMV avg_err {avgErr:E3} exceeds tolerance; max_err={maxAbs:E3}");
        }
        finally
        {
            input.Dispose(); weightF32.Dispose(); outGpu.Dispose();
        }
    }

    private static unsafe void FillQ4_0Random(Tensor t, int blocks, Random rng)
    {
        byte* p = (byte*)t.DataPointer;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = p + b * 18;
            *(Half*)block = (Half)((rng.NextDouble() * 0.4) + 0.05);
            byte* qs = block + 2;
            for (int i = 0; i < 16; i++) qs[i] = (byte)rng.Next(256);
        }
    }
}
