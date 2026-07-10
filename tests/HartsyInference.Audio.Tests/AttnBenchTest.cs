using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Micro-bench F5's attention shape [1,16,454,64]: monolithic FlashAttention vs
/// ScaledDotProductAttention (materialized TF32 GEMM path). F5_CUDA=1 + F5_PTX required.</summary>
public sealed class AttnBenchTest
{
    private readonly ITestOutputHelper _out;
    public AttnBenchTest(ITestOutputHelper o) => _out = o;

    [Fact]
    public unsafe void BenchF5Attn()
    {
        if (Environment.GetEnvironmentVariable("F5_CUDA") != "1") { _out.WriteLine("skip (no CUDA)"); return; }
        using IBackend b = new HartsyInference.Cuda.CudaBackend(0, Environment.GetEnvironmentVariable("F5_PTX")!);
        const int B = 1, H = 16, T = 454, D = 64;
        float scale = 1f / MathF.Sqrt(D);
        Tensor q = Rand(B, H, T, D), k = Rand(B, H, T, D), v = Rand(B, H, T, D);
        Tensor oFlash = new(new TensorShape(B, H, T, D), DType.F32);
        Tensor oSdpa = new(new TensorShape(B, H, T, D), DType.F32);

        // Warm both.
        b.FlashAttention(oFlash, q, k, v, kvLen: T, kvGroup: 1, causal: false, qOffset: 0, scale);
        b.ScaledDotProductAttention(oSdpa, q, k, v, null, scale); b.Sync();

        int N = 50;
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < N; i++) b.FlashAttention(oFlash, q, k, v, kvLen: T, kvGroup: 1, causal: false, qOffset: 0, scale);
        b.Sync(); double flashMs = sw.Elapsed.TotalMilliseconds / N;

        sw.Restart();
        for (int i = 0; i < N; i++) b.ScaledDotProductAttention(oSdpa, q, k, v, null, scale);
        b.Sync(); double sdpaMs = sw.Elapsed.TotalMilliseconds / N;

        // Correctness: max abs diff.
        float* pf = (float*)oFlash.DataPointer; float* ps = (float*)oSdpa.DataPointer;
        float maxDiff = 0; for (long i = 0; i < oFlash.ElementCount; i++) maxDiff = MathF.Max(maxDiff, MathF.Abs(pf[i] - ps[i]));
        _out.WriteLine($"[ATTN BENCH] FlashAttention {flashMs:F2} ms/call | SDPA {sdpaMs:F2} ms/call | speedup {flashMs / sdpaMs:F1}x | maxDiff {maxDiff:E3}");
    }

    private static unsafe Tensor Rand(int b, int h, int t, int d)
    {
        Tensor x = new(new TensorShape(b, h, t, d), DType.F32);
        float* p = (float*)x.DataPointer; uint s = 12345;
        for (long i = 0; i < x.ElementCount; i++) { s = s * 1664525 + 1013904223; p[i] = ((s >> 8) / 16777216f - 0.5f) * 2f; }
        return x;
    }
}
