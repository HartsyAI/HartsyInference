using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates the cuDNN conv-forward fast path (HARTSY_CONV_CUDNN) against the im2col→GEMM path
/// on the same backend inputs — SDXL-UNet-style F16 3×3/1×1 convs and a stride-2 downsample, plus a BF16
/// (VAE-style) case. Both routes accumulate in F32 from 16-bit operands, so tolerance is fp16-scale.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnConvTests
{
    private readonly ITestOutputHelper _output;
    public CudnnConvTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x8badf00du;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 65535f - 0.5f; }

    private static Tensor RndF32(TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    private static Tensor To16(Tensor f32, DType dt)
    {
        Tensor t = new(f32.Shape, dt);
        float* src = (float*)f32.DataPointer;
        ushort* dst = (ushort*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            dst[i] = dt == DType.F16
                ? BitConverter.HalfToUInt16Bits((Half)src[i])
                : (ushort)(BitConverter.SingleToUInt32Bits(src[i]) >> 16);
        }
        return t;
    }

    private static float ToF32(ushort v, DType dt)
        => dt == DType.F16 ? (float)BitConverter.UInt16BitsToHalf(v) : BitConverter.UInt32BitsToSingle((uint)v << 16);

    [Theory]
    [InlineData(2, 32, 24, 24, 48, 3, 1, 1, "F16")]   // UNet-style 3x3
    [InlineData(2, 32, 24, 24, 48, 3, 2, 1, "F16")]   // downsample stride 2
    [InlineData(2, 32, 24, 24, 48, 1, 1, 0, "F16")]   // 1x1 shortcut
    [InlineData(1, 32, 24, 24, 48, 3, 1, 1, "BF16")]  // VAE-style BF16
    public void CudnnConv_MatchesIm2ColPath(int batch, int inCh, int h, int w, int outCh, int k, int stride, int pad, string dtypeName)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        DType dt = dtypeName == "F16" ? DType.F16 : DType.BF16;
        int outH = (h + 2 * pad - k) / stride + 1;
        int outW = (w + 2 * pad - k) / stride + 1;

        using Tensor inF32 = RndF32(new TensorShape(batch, inCh, h, w));
        using Tensor wF32 = RndF32(new TensorShape(outCh, inCh, k, k));
        using Tensor bF32 = RndF32(new TensorShape(outCh));
        using Tensor input = To16(inF32, dt);
        using Tensor weight = To16(wF32, dt);
        using Tensor bias = To16(bF32, dt);

        string? prev = Environment.GetEnvironmentVariable("HARTSY_CONV_CUDNN");
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_CONV_CUDNN", "1");
            using Tensor outCudnn = new(new TensorShape(batch, outCh, outH, outW), dt);
            bool engaged;
            using (CudaBackend cudnnBackend = new(0, ptxDir))
            {
                ((IBackend)cudnnBackend).Conv2D(outCudnn, input, weight, bias, stride, stride, pad, pad);
                cudnnBackend.Sync();
                unsafe { _ = (nint)outCudnn.DataPointer; }
                engaged = cudnnBackend.CudnnConvEngaged;
            }

            Environment.SetEnvironmentVariable("HARTSY_CONV_CUDNN", "0");
            using Tensor outIm2Col = new(new TensorShape(batch, outCh, outH, outW), dt);
            using (CudaBackend im2colBackend = new(0, ptxDir))
            {
                ((IBackend)im2colBackend).Conv2D(outIm2Col, input, weight, bias, stride, stride, pad, pad);
                im2colBackend.Sync();
                unsafe { _ = (nint)outIm2Col.DataPointer; }
            }

            ushort* a = (ushort*)outCudnn.DataPointer;
            ushort* b = (ushort*)outIm2Col.DataPointer;
            float maxDiff = 0f, maxMag = 0f;
            for (long i = 0; i < outCudnn.ElementCount; i++)
            {
                float fa = ToF32(a[i], dt), fb = ToF32(b[i], dt);
                maxDiff = MathF.Max(maxDiff, MathF.Abs(fa - fb));
                maxMag = MathF.Max(maxMag, MathF.Abs(fb));
            }
            _output.WriteLine($"{dtypeName} {batch}x{inCh}x{h}x{w} k{k}s{stride}: max|Δ|={maxDiff:E3} (maxMag={maxMag:F3}) engaged={engaged}");
            Assert.True(engaged, "cuDNN conv fast path did NOT engage — it fell back to im2col");
            float tol = dt == DType.F16 ? 3e-2f : 2e-1f;
            Assert.True(maxDiff <= tol, $"cuDNN conv diverges from im2col path by {maxDiff:E3} (tol {tol:E1})");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_CONV_CUDNN", prev);
        }
    }
}
