using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Nvfp4;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Bring-up gate for the native block-scaled GEMM: on Blackwell, a resident nvfp4 weight multiplied through <see cref="BlockScaledGemmExecutor"/> (W4A4, the activation block-quantized on the stream) against the same weight through the unpack path (W4A16). The difference is the activation's own e2m1 error. The largest shape also reports both paths' time per Linear, which is the number that says whether native FP4 beats the unpack on the card. Skips below Blackwell — this hardware has no block-scaled tensor cores and the unpack is the only correct path.</summary>
[Collection("CudaSerial")]
public sealed class BlockScaledGemmTests
{
    private readonly ITestOutputHelper _output;
    public BlockScaledGemmTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(64, 256, 128)]
    [InlineData(200, 320, 256)]
    [InlineData(4096, 3072, 3072)]   // DiT-shaped: the timing row
    public unsafe void NativeNvfp4Gemm_MatchesTheUnpackPath(int m, int n, int k)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (!backend.BlockScaledExecutor.IsSupported)
        {
            _output.WriteLine($"SKIPPED: block-scaled GEMM needs Blackwell (this GPU is SM " +
                $"{backend.Context.ComputeCapabilityMajor}.{backend.Context.ComputeCapabilityMinor}).");
            return;
        }

        int paddedRows = (n + 127) / 128 * 128, paddedCols = (k / 16 + 3) / 4 * 4;
        using Tensor packed = new Tensor(new TensorShape(n, k / 2), DType.U8);
        using Tensor scales = new Tensor(new TensorShape(paddedRows, paddedCols), DType.F8E4M3);
        using Tensor global = new Tensor(new TensorShape(1), DType.F32);
        using Tensor input = new Tensor(new TensorShape(m, k), DType.F32);
        using Tensor bias = new Tensor(new TensorShape(n), DType.F32);
        using Tensor outNative = new Tensor(new TensorShape(m, n), DType.F16);
        using Tensor outRef = new Tensor(new TensorShape(m, n), DType.F16);
        Random rng = new Random(37);
        byte* wp = (byte*)packed.DataPointer;
        for (long i = 0; i < (long)n * (k / 2); i++) wp[i] = (byte)rng.Next(256);
        byte* sp = (byte*)scales.DataPointer;
        for (long i = 0; i < (long)paddedRows * paddedCols; i++) sp[i] = (byte)(0x20 + rng.Next(0x21));   // 2^-3 .. 2^1
        ((float*)global.DataPointer)[0] = 0.01f;
        for (long i = 0; i < (long)m * k; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        for (int i = 0; i < n; i++) ((float*)bias.DataPointer)[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        Assert.True(Nvfp4Codec.TryAttachResident(packed, scales, global, hasPreQuantScale: false, out Tensor weight));

        backend.EnableNativeFp4Gemm = true;
        backend.Linear(outNative, input, weight, bias);
        backend.Sync();
        backend.EnableNativeFp4Gemm = false;
        backend.Linear(outRef, input, weight, bias);
        backend.Sync();

        double sumAbs = 0, sumRefAbs = 0;
        long total = (long)m * n;
        Half* np = (Half*)outNative.DataPointer;
        Half* rp = (Half*)outRef.DataPointer;
        for (long i = 0; i < total; i++)
        {
            float a = (float)np[i], r = (float)rp[i];
            Assert.False(float.IsNaN(a) || float.IsInfinity(a), $"native output non-finite at {i}: {a}");
            sumAbs += MathF.Abs(a - r);
            sumRefAbs += MathF.Abs(r);
        }
        float relErr = (float)(sumAbs / Math.Max(sumRefAbs, 1e-9));
        _output.WriteLine($"native nvfp4 {m}x{n}x{k}: rel_err={relErr:E3} vs the unpack path");
        Assert.True(relErr < 8e-2f, $"native block-scaled GEMM rel_err {relErr:E3} exceeds the activation-quantization budget");

        if (m < 1024) return;
        const int reps = 20;
        double TimePerLinear(bool native)
        {
            backend.EnableNativeFp4Gemm = native;
            backend.Linear(outNative, input, weight, bias);
            backend.Sync();
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < reps; i++) backend.Linear(outNative, input, weight, bias);
            backend.Sync();
            return Stopwatch.GetElapsedTime(t0).TotalMilliseconds / reps;
        }
        double nativeMs = TimePerLinear(true), unpackMs = TimePerLinear(false);
        _output.WriteLine($"TIMING {m}x{n}x{k}: native {nativeMs:F3} ms/Linear, unpack {unpackMs:F3} ms/Linear ({unpackMs / nativeMs:F2}x)");
    }
}
