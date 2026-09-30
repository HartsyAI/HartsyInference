using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The channels-last cuDNN convolution route (NHWC transposes around the call) against the NCHW route on the
/// same inputs, at a size above <see cref="CudaBackend.ChannelsLastMinElements"/>, plus the tiled transpose it rests on.
/// Skips cleanly without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class ConvChannelsLastTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(64, 96, 1, 1)]
    [InlineData(64, 48, 2, 1)]   // stride 2
    public void ChannelsLastConv2D_MatchesNchw(int cIn, int cOut, int stride, int pad)
    {
        string? ptx = PtxDir();
        if (ptx is null) return;
        const int h = 720, w = 736;   // 64·720·736 ≈ 33.9M ≥ 2^25; both dims non-square and not tile multiples
        Assert.True((long)cIn * h * w >= CudaBackend.ChannelsLastMinElements);
        using Tensor x = RandomBf16(new TensorShape(1, cIn, h, w), 1, 1f);
        using Tensor wt = RandomBf16(new TensorShape(cOut, cIn, 3, 3), 2, 0.05f);
        using Tensor bias = RandomBf16(new TensorShape(cOut), 3, 0.5f);
        int oh = (h + 2 * pad - 3) / stride + 1, ow = (w + 2 * pad - 3) / stride + 1;

        float[] cl = Conv(ptx, channelsLast: true, x, wt, bias, oh, ow, stride, pad, out long clCount);
        float[] nchw = Conv(ptx, channelsLast: false, x, wt, bias, oh, ow, stride, pad, out long nchwCount);
        Assert.Equal(1, clCount);
        Assert.Equal(0, nchwCount);
        double maxDiff = 0, maxAbs = 0;
        for (int i = 0; i < cl.Length; i++) { maxDiff = Math.Max(maxDiff, Math.Abs(cl[i] - nchw[i])); maxAbs = Math.Max(maxAbs, Math.Abs(nchw[i])); }
        output.WriteLine($"cIn={cIn} cOut={cOut} stride={stride}: max|CL - NCHW| = {maxDiff:E3} (max |out| {maxAbs:F2})");
        // Both accumulate in F32 and round the output to BF16 once; engines differ in reduction order only.
        Assert.True(maxDiff <= 1e-2 * Math.Max(1, maxAbs), $"channels-last diverges from NCHW by {maxDiff:E3}");
    }

    [Theory]
    [InlineData(2, 3, 77, 45)]
    [InlineData(4, 1, 33, 1000)]
    public void TransposeTiled_MatchesHostTranspose(int elementBytes, int batch, int d1, int d2)
    {
        string? ptx = PtxDir();
        if (ptx is null) return;
        using CudaBackend backend = new(0, ptx);
        using (Tensor warm = new(new TensorShape(4), DType.F32)) ((IBackend)backend).Add(warm, warm, warm);   // loads the kernels
        CudaKernels kernels = backend.Kernels ?? throw new InvalidOperationException("kernels not loaded");
        if (!kernels.HasTransposeTiled) { output.WriteLine("SKIPPED: channels_last.ptx absent"); return; }
        long n = (long)batch * d1 * d2;
        byte[] src = new byte[n * elementBytes];
        new Random(5).NextBytes(src);
        ulong dIn = CudaMemory.Allocate((nuint)src.Length), dOut = CudaMemory.Allocate((nuint)src.Length);
        try
        {
            fixed (byte* p = src) CudaMemory.CopyHostToDevice(dIn, p, (nuint)src.Length);
            backend.Sync();
            kernels.LaunchTransposeTiled(dOut, dIn, batch, d1, d2, elementBytes, backend.Stream.Handle);
            byte[] got = new byte[src.Length];
            backend.Sync();
            fixed (byte* p = got) CudaMemory.CopyDeviceToHost(p, dOut, (nuint)got.Length);
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < d1; i++)
                    for (int j = 0; j < d2; j++)
                        for (int e = 0; e < elementBytes; e++)
                            Assert.Equal(src[(((long)b * d1 + i) * d2 + j) * elementBytes + e], got[(((long)b * d2 + j) * d1 + i) * elementBytes + e]);
        }
        finally
        {
            CudaMemory.Free(dIn);
            CudaMemory.Free(dOut);
        }
    }

    private static float[] Conv(string ptx, bool channelsLast, Tensor x, Tensor wt, Tensor bias, int oh, int ow,
        int stride, int pad, out long clCount)
    {
        KnobStore.Set(EngineKnobs.ConvChannelsLast, channelsLast);
        try
        {
            using CudaBackend backend = new(0, ptx);
            using Tensor y = new(new TensorShape(1, wt.Shape[0], oh, ow), DType.BF16);
            ((IBackend)backend).Conv2D(y, x, wt, bias, stride, stride, pad, pad);
            backend.Sync();
            clCount = backend.ChannelsLastConvCount;
            using Tensor f = y.CastTo(DType.F32);
            return new ReadOnlySpan<float>(f.DataPointer, (int)f.ElementCount).ToArray();
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.ConvChannelsLast);
        }
    }

    private string? PtxDir()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return null; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.Available) { output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return null; }
        string ptx = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(ptx) ? ptx : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor RandomBf16(TensorShape shape, int seed, float scale)
    {
        using Tensor f = new(shape, DType.F32);
        Random rng = new(seed);
        for (long i = 0; i < f.ElementCount; i++) ((float*)f.DataPointer)[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return f.CastTo(DType.BF16);
    }
}
