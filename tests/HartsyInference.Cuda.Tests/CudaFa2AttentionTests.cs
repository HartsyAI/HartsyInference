using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The tensor-core FlashAttention-2 prefill kernel against the CPU reference: grouped-query heads, causal masking with an
/// absolute query offset, a key buffer whose stride exceeds the valid length, a sliding window and partial query tiles.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaFa2AttentionTests
{
    private readonly ITestOutputHelper _output;
    public CudaFa2AttentionTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Random(Random rng, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)((rng.NextDouble() - 0.5) * 2.0);
        return t;
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(128, 8, 2, 100, 0, 0, 100)]      // partial query tile, no cache slack
    [InlineData(128, 8, 2, 128, 0, 0, 300)]      // key buffer stride larger than the valid length
    [InlineData(128, 8, 2, 70, 197, 0, 500)]     // chunked prefill: queries start mid-sequence
    [InlineData(128, 8, 2, 90, 60, 48, 400)]     // sliding window
    [InlineData(64, 6, 2, 150, 0, 0, 150)]
    [InlineData(64, 4, 4, 33, 31, 20, 128)]      // MHA, window
    [InlineData(128, 4, 1, 16, 5, 0, 64)]        // the smallest block the kernel takes
    public void Fa2_MatchesCpuReference(int d, int hq, int hkv, int tq, int qOffset, int window, int keyStride)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random rng = new(d * 31 + tq);
        int kvLen = qOffset + tq;
        int group = hq / hkv;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Random(rng, 1, hq, tq, d);
        using Tensor k = Random(rng, 1, hkv, keyStride, d);
        using Tensor v = Random(rng, 1, hkv, keyStride, d);
        using Tensor expected = new(new TensorShape(1, hq, tq, d), DType.F32);
        AttentionReference.FlashAttention(expected, q, k, v, kvLen, group, true, qOffset, scale, 0f, null, window, null);

        using CudaBackend cuda = new(0, PtxDir());
        using Tensor actual = new(new TensorShape(1, hq, tq, d), DType.F32);
        cuda.FlashAttention(actual, q, k, v, kvLen, group, true, qOffset, scale, 0f, null, window, null);
        cuda.Sync();

        float* pe = (float*)expected.DataPointer, pa = (float*)actual.DataPointer;
        double diff = 0, norm = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double e = pe[i], a = pa[i];
            diff += (a - e) * (a - e);
            norm += e * e;
        }
        double rel = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        _output.WriteLine($"d={d} hq={hq} hkv={hkv} tq={tq} off={qOffset} win={window} stride={keyStride}: relative L2 error {rel:E3}");
        Assert.True(rel <= 3e-3, $"FA2 diverges from the reference by {rel:E3}");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Fa2_PrefillThroughput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int d = 128, hq = 16, hkv = 16, tq = 4096;
        Random rng = new(5);
        using Tensor q = Random(rng, 1, hq, tq, d);
        using Tensor k = Random(rng, 1, hkv, tq, d);
        using Tensor v = Random(rng, 1, hkv, tq, d);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor o = new(new TensorShape(1, hq, tq, d), DType.F32);
        float scale = 1f / MathF.Sqrt(d);
        cuda.FlashAttention(o, q, k, v, tq, 1, true, 0, scale);   // warm-up and upload
        cuda.Sync();
        Stopwatch sw = Stopwatch.StartNew();
        const int reps = 5;
        for (int i = 0; i < reps; i++) cuda.FlashAttention(o, q, k, v, tq, 1, true, 0, scale);
        cuda.Sync();
        double ms = sw.Elapsed.TotalMilliseconds / reps;
        double tflops = 4.0 * tq * (double)tq * d * hq / 2 / (ms * 1e9);
        _output.WriteLine($"FlashAttention tq={tq} d={d} hq={hq}: {ms:F2} ms per call, {tflops:F1} TFLOP/s causal");
    }
}
