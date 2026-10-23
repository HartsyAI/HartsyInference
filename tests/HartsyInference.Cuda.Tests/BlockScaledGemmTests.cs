using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Mxfp8;
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

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(64, 256, 128)]
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

    /// <summary>The weight's own <see cref="Tensor.Fp8ScaleFactor"/> is part of alpha on every GEMM path — it is what Z-Image's
    /// and the Flux blocks' F16 damps scale. A native path that dropped it ran those projections 64× hot and decoded NaN.
    /// The unpack arm is fed the activations the native path quantizes (the same kernel, decoded on the host), so the two
    /// differ only by the unpack GEMM's F16 rounding, not by the ~10 % that FP4 activations cost.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public unsafe void NativeNvfp4Gemm_AppliesTheWeightScaleFactor()
    {
        const int m = 64, n = 256, k = 128;
        const float factor = 1.0f / 64.0f;
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
        using Tensor plain = new Tensor(new TensorShape(m, n), DType.F32);
        using Tensor damped = new Tensor(new TensorShape(m, n), DType.F32);
        using Tensor unpacked = new Tensor(new TensorShape(m, n), DType.F32);
        Random rng = new Random(41);
        byte* wp = (byte*)packed.DataPointer;
        for (long i = 0; i < (long)n * (k / 2); i++) wp[i] = (byte)rng.Next(256);
        byte* sp = (byte*)scales.DataPointer;
        for (long i = 0; i < (long)paddedRows * paddedCols; i++) sp[i] = (byte)(0x20 + rng.Next(0x21));
        ((float*)global.DataPointer)[0] = 0.01f;
        for (long i = 0; i < (long)m * k; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        Assert.True(Nvfp4Codec.TryAttachResident(packed, scales, global, hasPreQuantScale: false, out Tensor weight));

        backend.EnableNativeFp4Gemm = true;
        backend.Linear(plain, input, weight, null);
        backend.Sync();
        weight.Fp8ScaleFactor = factor;
        backend.Linear(damped, input, weight, null);
        backend.Sync();
        backend.EnableNativeFp4Gemm = false;
        using Tensor quantizedInput = QuantizeLikeTheNativePath(backend, input, m, k);
        backend.Linear(unpacked, quantizedInput, weight, null);
        backend.Sync();

        double scaledDiff = 0, unpackDiff = 0, sumDamped = 0;
        float* pp = (float*)plain.DataPointer, dp = (float*)damped.DataPointer, up = (float*)unpacked.DataPointer;
        for (long i = 0; i < (long)m * n; i++)
        {
            scaledDiff += Math.Abs(dp[i] - pp[i] * factor);
            unpackDiff += Math.Abs(dp[i] - up[i]);
            sumDamped += Math.Abs(dp[i]);
        }
        double scaledErr = scaledDiff / Math.Max(sumDamped, 1e-12), unpackErr = unpackDiff / Math.Max(sumDamped, 1e-12);
        _output.WriteLine($"native with factor vs native·factor: {scaledErr:E3}; vs the unpack path on the same FP4 activations: {unpackErr:E3}");
        Assert.True(scaledErr < 1e-5, $"the native path did not apply the weight's scale factor ({scaledErr:E3})");
        Assert.True(unpackErr < 5e-3, $"the native and unpack paths disagree once the weight carries a factor ({unpackErr:E3})");
    }

    /// <summary>The activations exactly as the native path sees them: block_quant run on the device, then the packed values
    /// decoded with its own per-tensor scale (high nibble = even element, scales in the blocked layout).</summary>
    private static unsafe Tensor QuantizeLikeTheNativePath(CudaBackend backend, Tensor input, int m, int k)
    {
        int paddedRows = (m + 127) / 128 * 128, paddedCols = (k / 16 + 3) / 4 * 4;
        int count = m * k;
        ulong x = CudaMemory.Allocate((nuint)((long)count * sizeof(float)));
        ulong packed = CudaMemory.Allocate((nuint)(count / 2));
        ulong scale = CudaMemory.Allocate((nuint)((long)paddedRows * paddedCols));
        ulong scratch = CudaMemory.Allocate((nuint)(CudaKernels.BlockQuantScratchFloats(count) * sizeof(float)));
        try
        {
            CudaMemory.CopyHostToDevice(x, (void*)input.DataPointer, (nuint)((long)count * sizeof(float)));
            backend.Kernels!.LaunchBlockQuant(BlockScaleFormat.Nvfp4, packed, scale, scratch, x, DType.F32, m, k,
                paddedRows, paddedCols, 1f, backend.Stream.Handle);
            backend.Sync();
            byte[] packedHost = new byte[count / 2], scaleHost = new byte[paddedRows * paddedCols];
            float sf;
            fixed (byte* p = packedHost) CudaMemory.CopyDeviceToHost(p, packed, (nuint)packedHost.Length);
            fixed (byte* p = scaleHost) CudaMemory.CopyDeviceToHost(p, scale, (nuint)scaleHost.Length);
            CudaMemory.CopyDeviceToHost(&sf, scratch, sizeof(float));

            using Tensor scaleBytes = new Tensor(new TensorShape(256), DType.F8E4M3);
            for (int i = 0; i < 256; i++) ((byte*)scaleBytes.DataPointer)[i] = (byte)i;
            using Tensor e4m3 = scaleBytes.CastTo(DType.F32);
            float* e4 = (float*)e4m3.DataPointer;
            float[] e2m1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];
            Tensor result = new Tensor(new TensorShape(m, k), DType.F32);
            float* dst = (float*)result.DataPointer;
            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < k; c++)
                {
                    byte b = packedHost[(r * k + c) / 2];
                    int nibble = c % 2 == 0 ? b >> 4 : b & 0xF;
                    float magnitude = e2m1[nibble & 7] * e4[scaleHost[Nvfp4ResidentCodec.SwizzledScaleIndex(r, c / 16, paddedCols)]] * sf;
                    dst[r * k + c] = (nibble & 8) != 0 ? -magnitude : magnitude;
                }
            }
            return result;
        }
        finally
        {
            foreach (ulong buffer in new[] { x, packed, scale, scratch }) CudaMemory.Free(buffer);
        }
    }

    /// <summary>A shape cuBLASLt declines must not take the generation down: the call is served by the unpack path, and
    /// the weight stays there instead of being retried every step. The decline is forced through the test hook because
    /// cuBLASLt 13.8 accepts every shape the gate lets through; the hook overrides the status after the GEMM ran, so this
    /// covers the fallback and its bookkeeping, not a real refusal.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public unsafe void DeclinedNativeGemm_FallsBackToTheUnpackPath()
    {
        const int m = 64, n = 256, k = 128;
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
        using Tensor declinedOut = new Tensor(new TensorShape(m, n), DType.F32);
        using Tensor retriedOut = new Tensor(new TensorShape(m, n), DType.F32);
        using Tensor unpackOut = new Tensor(new TensorShape(m, n), DType.F32);
        Random rng = new Random(43);
        byte* wp = (byte*)packed.DataPointer;
        for (long i = 0; i < (long)n * (k / 2); i++) wp[i] = (byte)rng.Next(256);
        byte* sp = (byte*)scales.DataPointer;
        for (long i = 0; i < (long)paddedRows * paddedCols; i++) sp[i] = (byte)(0x20 + rng.Next(0x21));
        ((float*)global.DataPointer)[0] = 0.01f;
        for (long i = 0; i < (long)m * k; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        Assert.True(Nvfp4Codec.TryAttachResident(packed, scales, global, hasPreQuantScale: false, out Tensor weight));

        backend.EnableNativeFp4Gemm = true;
        CudaBackend.ForceNativeBlockScaledDeclineForTest = 15;   // CUBLAS_STATUS_NOT_SUPPORTED
        try
        {
            backend.Linear(declinedOut, input, weight, null);
            backend.Sync();
        }
        finally
        {
            CudaBackend.ForceNativeBlockScaledDeclineForTest = null;
        }
        backend.Linear(retriedOut, input, weight, null);   // native allowed again, but this weight was declined
        backend.Sync();
        backend.EnableNativeFp4Gemm = false;
        backend.Linear(unpackOut, input, weight, null);
        backend.Sync();

        float* d = (float*)declinedOut.DataPointer, r = (float*)retriedOut.DataPointer, u = (float*)unpackOut.DataPointer;
        for (long i = 0; i < (long)m * n; i++)
        {
            Assert.Equal(u[i], d[i]);
            Assert.Equal(u[i], r[i]);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(64, 256, 128)]
    [InlineData(200, 320, 256)]
    public unsafe void NativeMxfp8Gemm_MatchesTheUnpackPath(int m, int n, int k)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new CudaBackend(0, ptxDir);
        if (!backend.BlockScaledExecutor.IsSupported)
        {
            _output.WriteLine("SKIPPED: block-scaled GEMM needs Blackwell.");
            return;
        }
        int paddedRows = (n + 127) / 128 * 128, paddedCols = (k / 32 + 3) / 4 * 4;
        using Tensor weight = new Tensor(new TensorShape(n, k), DType.F8E4M3);
        using Tensor scales = new Tensor(new TensorShape(paddedRows, paddedCols), DType.U8);
        using Tensor input = new Tensor(new TensorShape(m, k), DType.F32);
        using Tensor outNative = new Tensor(new TensorShape(m, n), DType.F16);
        using Tensor outRef = new Tensor(new TensorShape(m, n), DType.F16);
        Random rng = new Random(43);
        byte* wp = (byte*)weight.DataPointer;
        for (long i = 0; i < (long)n * k; i++) wp[i] = (byte)(rng.Next(0x30, 0x40) | (rng.Next(2) << 7));   // |w| in [0.5, 2)
        byte* sp = (byte*)scales.DataPointer;
        for (long i = 0; i < (long)paddedRows * paddedCols; i++) sp[i] = (byte)(124 + rng.Next(7));   // 2^-3 .. 2^3
        for (long i = 0; i < (long)m * k; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        Assert.True(Mxfp8Codec.TryAttachResident(weight, scales));

        backend.EnableNativeFp4Gemm = true;
        backend.Linear(outNative, input, weight, bias: null);
        backend.Sync();
        backend.EnableNativeFp4Gemm = false;
        backend.Linear(outRef, input, weight, bias: null);
        backend.Sync();

        double sumAbs = 0, sumRefAbs = 0;
        Half* np = (Half*)outNative.DataPointer;
        Half* rp = (Half*)outRef.DataPointer;
        for (long i = 0; i < (long)m * n; i++)
        {
            float a = (float)np[i], r = (float)rp[i];
            Assert.False(float.IsNaN(a) || float.IsInfinity(a), $"native output non-finite at {i}: {a}");
            sumAbs += MathF.Abs(a - r);
            sumRefAbs += MathF.Abs(r);
        }
        float relErr = (float)(sumAbs / Math.Max(sumRefAbs, 1e-9));
        _output.WriteLine($"native mxfp8 {m}x{n}x{k}: rel_err={relErr:E3} vs the unpack path");
        Assert.True(relErr < 5e-2f, $"native MXFP8 GEMM rel_err {relErr:E3} exceeds the activation-quantization budget");
    }
}
