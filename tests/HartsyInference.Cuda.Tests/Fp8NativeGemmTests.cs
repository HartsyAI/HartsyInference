using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Stage-4 gate for the native FP8 GEMM path (<see cref="CudaBackend.EnableNativeFp8Gemm"/>,
/// <see cref="Fp8GemmExecutor"/> via <c>cublasLtMatmul</c>). Validates that the native FP8 tensor-core
/// path produces the same result as the default cast-FP8→F16-then-F16-GEMM fallback, on identical FP8
/// operands. Both paths consume the SAME FP8 bytes, so the only variable is the GEMM implementation;
/// E4M3 values are exactly representable in F16, so the expected diff is at tensor-core accumulation
/// noise, well under the ~1e-2 relative tolerance the plan allows for the FP8 path.
///
/// <para>Runs only on Ada+ (SM 8.9+, where <see cref="Fp8GemmExecutor.IsSupported"/>); SKIPS on Ampere
/// (e.g. the RTX 3060) — that hardware has no native FP8 GEMM and the fallback is the only correct path.</para></summary>
[Collection("CudaSerial")]
public sealed class Fp8NativeGemmTests
{
    private readonly ITestOutputHelper _output;
    public Fp8NativeGemmTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(64, 256, 128)]
    [InlineData(256, 512, 512)]
    public unsafe void NativeFp8Gemm_MatchesCastF16Reference(int m, int outDim, int inDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (!backend.Fp8Executor.IsSupported)
        {
            _output.WriteLine($"SKIPPED: native FP8 GEMM needs SM 8.9+ (this GPU is SM " +
                $"{backend.Context.ComputeCapabilityMajor}.{backend.Context.ComputeCapabilityMinor}).");
            return;
        }

        // out[M,N] = input[M,K] · weight[N,K]^T, all FP8 in, F16 out. Keep values small so the K-length
        // dot products stay well inside the F16 output range (no overflow → no inf/NaN to confound the diff).
        Tensor input = new Tensor(new TensorShape(m, inDim), DType.F8E4M3);
        Tensor weight = new Tensor(new TensorShape(outDim, inDim), DType.F8E4M3);
        Tensor outNative = new Tensor(new TensorShape(m, outDim), DType.F16);
        Tensor outRef = new Tensor(new TensorShape(m, outDim), DType.F16);
        try
        {
            Random rng = new Random(7);
            byte* ip = (byte*)input.DataPointer;
            for (long i = 0; i < (long)m * inDim; i++) ip[i] = EncodeE4M3((float)(rng.NextDouble() * 0.5 - 0.25));
            byte* wp = (byte*)weight.DataPointer;
            for (long i = 0; i < (long)outDim * inDim; i++) wp[i] = EncodeE4M3((float)(rng.NextDouble() * 0.5 - 0.25));

            // Native FP8 tensor-core GEMM (cublasLtMatmul).
            backend.EnableNativeFp8Gemm = true;
            backend.Linear(outNative, input, weight, bias: null);
            backend.Sync();

            // Reference: cast FP8→F16, then cublasGemmEx F16 (the default fallback).
            backend.EnableNativeFp8Gemm = false;
            backend.Linear(outRef, input, weight, bias: null);
            backend.Sync();

            Half* np = (Half*)outNative.DataPointer;
            Half* rp = (Half*)outRef.DataPointer;
            double sumAbs = 0.0, sumRefAbs = 0.0;
            float maxAbs = 0f;
            long total = (long)m * outDim;
            for (long i = 0; i < total; i++)
            {
                float n = (float)np[i], r = (float)rp[i];
                Assert.False(float.IsNaN(n) || float.IsInfinity(n), $"native output non-finite at {i}: {n}");
                float err = MathF.Abs(n - r);
                sumAbs += err;
                sumRefAbs += MathF.Abs(r);
                if (err > maxAbs) maxAbs = err;
            }
            float avgErr = (float)(sumAbs / total);
            float relErr = sumRefAbs > 0 ? (float)(sumAbs / sumRefAbs) : avgErr;
            _output.WriteLine($"Native FP8 {m}x{outDim}x{inDim}: avg_err={avgErr:E3}, max_err={maxAbs:E3}, rel_err={relErr:E3}");
            Assert.True(relErr < 1e-2f, $"Native FP8 GEMM rel_err {relErr:E3} exceeds tolerance (avg={avgErr:E3}, max={maxAbs:E3}).");
        }
        finally
        {
            input.Dispose(); weight.Dispose(); outNative.Dispose(); outRef.Dispose();
        }
    }

    /// <summary>Validates the activation-quantization kernels feeding the native path: per-tensor absmax →
    /// dequant scale (amax/448) + e4m3 quantized bytes. Pure compute kernels, so this runs on ANY CUDA GPU
    /// (Ampere included) — no fp8 GEMM involved.</summary>
    [Fact]
    public unsafe void Fp8QuantKernels_DynamicQuant_MatchesHostReference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new CudaBackend(0, ptxDir);

        const int count = 300_007;   // odd size exercises the grid tail
        Tensor input = new Tensor(new TensorShape(count), DType.F32);
        Tensor fp8Out = new Tensor(new TensorShape(count), DType.F8E4M3);
        Tensor scaleOut = new Tensor(new TensorShape(1), DType.F32);
        try
        {
            Random rng = new Random(11);
            float* ip = (float*)input.DataPointer;
            for (int i = 0; i < count; i++) ip[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
            ip[12345] = -173.25f;   // known amax, negative on purpose
            float amax = 173.25f;

            backend.Fp8QuantizeActivationForTest(fp8Out, scaleOut, input);
            backend.Sync();

            float scale = *(float*)scaleOut.DataPointer;
            float expectedScale = amax / 448.0f;
            Assert.True(MathF.Abs(scale - expectedScale) < 1e-6f, $"scale {scale} != amax/448 {expectedScale}");

            // Each byte must decode back to ≈ x/scale within half an e4m3 ulp (2^-4 relative for
            // normals, 2^-9·448-absolute near zero).
            byte* qp = (byte*)fp8Out.DataPointer;
            double sumRel = 0; int relCount = 0; float maxRel = 0;
            for (int i = 0; i < count; i++)
            {
                float target = ip[i] / scale;
                float dec = DecodeE4M3(qp[i]);
                Assert.False(float.IsNaN(dec), $"NaN code at {i} (byte 0x{qp[i]:X2}, target {target})");
                float err = MathF.Abs(dec - target);
                if (MathF.Abs(target) > 1.0f)
                {
                    float rel = err / MathF.Abs(target);
                    sumRel += rel; relCount++;
                    if (rel > maxRel) maxRel = rel;
                    Assert.True(rel <= 0.0625f + 1e-4f, $"elem {i}: decoded {dec} vs target {target} (rel {rel})");
                }
                else
                    Assert.True(err <= 0.125f, $"elem {i}: decoded {dec} vs target {target} (abs {err})");
            }
            _output.WriteLine($"quant {count} elems: scale={scale:E6}, avg_rel={(float)(sumRel / Math.Max(1, relCount)):E3}, max_rel={maxRel:E3}");
        }
        finally
        {
            input.Dispose(); fp8Out.Dispose(); scaleOut.Dispose();
        }
    }

    /// <summary>The full new wiring on Ada: F32 activation → dynamic e4m3 quantization (device-side scale via
    /// B_SCALE_POINTER) → native fp8 GEMM → F32/F16 output, diffed against the default cast-to-F16 fallback on
    /// the same F32 input. Tolerance is dominated by activation quantization (e4m3 ≈ 2 decimal digits).</summary>
    [Theory]
    [InlineData(64, 256, 128, false)]
    [InlineData(256, 512, 512, false)]
    [InlineData(200, 320, 256, true)]    // F32 out + M not multiple of 16 (only leading dims need alignment)
    public unsafe void NativeFp8Gemm_F32ActivationDynamicQuant_MatchesF16Fallback(int m, int outDim, int inDim, bool outF32)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (!backend.Fp8Executor.IsSupported)
        {
            _output.WriteLine("SKIPPED: native FP8 GEMM needs SM 8.9+.");
            return;
        }

        DType outDtype = outF32 ? DType.F32 : DType.F16;
        Tensor input = new Tensor(new TensorShape(m, inDim), DType.F32);
        Tensor weight = new Tensor(new TensorShape(outDim, inDim), DType.F8E4M3);
        Tensor bias = new Tensor(new TensorShape(outDim), DType.F32);
        Tensor outNative = new Tensor(new TensorShape(m, outDim), outDtype);
        Tensor outRef = new Tensor(new TensorShape(m, outDim), outDtype);
        try
        {
            Random rng = new Random(23);
            float* ip = (float*)input.DataPointer;
            for (long i = 0; i < (long)m * inDim; i++) ip[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
            byte* wp = (byte*)weight.DataPointer;
            for (long i = 0; i < (long)outDim * inDim; i++) wp[i] = EncodeE4M3((float)(rng.NextDouble() * 0.5 - 0.25));
            weight.Fp8ScaleFactor = 0.5f;   // must be applied exactly once by BOTH paths
            float* bp = (float*)bias.DataPointer;
            for (int i = 0; i < outDim; i++) bp[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            backend.EnableNativeFp8Gemm = true;
            backend.Linear(outNative, input, weight, bias);
            backend.Sync();

            backend.EnableNativeFp8Gemm = false;
            backend.Linear(outRef, input, weight, bias);
            backend.Sync();

            double sumAbs = 0.0, sumRefAbs = 0.0;
            float maxAbs = 0f;
            long total = (long)m * outDim;
            for (long i = 0; i < total; i++)
            {
                float n = outF32 ? ((float*)outNative.DataPointer)[i] : (float)((Half*)outNative.DataPointer)[i];
                float r = outF32 ? ((float*)outRef.DataPointer)[i] : (float)((Half*)outRef.DataPointer)[i];
                Assert.False(float.IsNaN(n) || float.IsInfinity(n), $"native output non-finite at {i}: {n}");
                float err = MathF.Abs(n - r);
                sumAbs += err;
                sumRefAbs += MathF.Abs(r);
                if (err > maxAbs) maxAbs = err;
            }
            float relErr = sumRefAbs > 0 ? (float)(sumAbs / sumRefAbs) : (float)(sumAbs / total);
            _output.WriteLine($"F32-act native FP8 {m}x{outDim}x{inDim} out={outDtype}: rel_err={relErr:E3}, max_err={maxAbs:E3}");
            Assert.True(relErr < 5e-2f, $"F32-activation native FP8 rel_err {relErr:E3} exceeds tolerance (max={maxAbs:E3}).");
        }
        finally
        {
            input.Dispose(); weight.Dispose(); bias.Dispose(); outNative.Dispose(); outRef.Dispose();
        }
    }

    /// <summary>e4m3fn decode (inverse of <see cref="EncodeE4M3"/>): 0x7F/0xFF → NaN.</summary>
    private static float DecodeE4M3(byte b)
    {
        int sign = (b & 0x80) != 0 ? -1 : 1;
        int expField = (b >> 3) & 0xF;
        int mant = b & 0x7;
        if (expField == 0xF && mant == 0x7) return float.NaN;
        float value = expField == 0
            ? mant * MathF.Pow(2f, -9)                                  // subnormal
            : (1f + mant / 8f) * MathF.Pow(2f, expField - 7);           // normal
        return sign * value;
    }

    /// <summary>Minimal F32 → FP8 E4M3 (OCP: 1 sign, 4 exp bias 7, 3 mantissa, no inf, max normal 448,
    /// 0x7F/0xFF reserved for NaN). Round-to-nearest on the mantissa; flushes tiny values to zero. Exact
    /// rounding is immaterial to the gate — both GEMM paths consume the identical encoded bytes — it only
    /// needs to emit valid, in-range, non-NaN codes.</summary>
    private static byte EncodeE4M3(float x)
    {
        if (x == 0f || float.IsNaN(x)) return 0;
        int sign = x < 0f ? 0x80 : 0;
        float a = MathF.Abs(x);
        if (a > 448f) a = 448f;
        int e = (int)MathF.Floor(MathF.Log2(a)); // unbiased exponent
        if (e < -6)
        {
            // Subnormal: value = mantissa/8 · 2^-6.
            int ms = (int)MathF.Round(a / MathF.Pow(2f, -6) * 8f);
            if (ms <= 0) return (byte)sign;
            if (ms > 7) ms = 7;
            return (byte)(sign | ms);
        }
        if (e > 8) e = 8;
        float scaled = a / MathF.Pow(2f, e); // [1, 2)
        int mant = (int)MathF.Round((scaled - 1f) * 8f);
        if (mant >= 8) { mant = 0; e += 1; if (e > 8) { e = 8; mant = 6; } }
        int biasedE = e + 7;
        if (biasedE >= 15 && mant > 6) mant = 6; // avoid S.1111.111 = NaN
        if (biasedE > 15) { biasedE = 15; mant = 6; }
        return (byte)(sign | (biasedE << 3) | mant);
    }
}
