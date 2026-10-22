using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Output digests of the native-F16 SageAttention row and the F16 flash row across a rebuild of their PTX; public API
/// only, so this file also compiles on the base tree. <c>HARTSY_SDPA_BASELINE_MODE=record</c> on base writes them to
/// <c>HARTSY_SDPA_BASELINE_DIR</c>; on head they are compared (unset dir skips).</summary>
[Collection("CudaSerial")]
public sealed unsafe class AttentionKernelDigestTests
{
    private readonly ITestOutputHelper _output;
    public AttentionKernelDigestTests(ITestOutputHelper output) => _output = output;

    /// <summary>Head dims 64 and 128, power-of-two key counts at and above 2048 (the padded V stride), a key tail inside a
    /// 32-key step, a query tail inside a 64-row tile, and two batches of three heads.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("sage-d128-s8192", 1, 2, 8192, 8192, 128)]
    [InlineData("sage-d64-s2048", 1, 4, 2048, 2048, 64)]
    [InlineData("sage-d128-q250-k1025", 1, 2, 250, 1025, 128)]
    [InlineData("sage-d64-b2-h3-q300-k250", 2, 3, 300, 250, 64)]
    public void Sage_OutputDigest_IsUnchanged(string name, int b, int h, int sq, int skv, int d)
    {
        if (BaselineDir() is not { } dir) return;
        Random rng = Seeded(name);
        using Tensor q = Make(Gauss(rng, (long)b * h * sq * d), new TensorShape(b, h, sq, d));
        using Tensor k = Make(Gauss(rng, (long)b * h * skv * d), new TensorShape(b, h, skv, d));
        using Tensor v = Make(Gauss(rng, (long)b * h * skv * d), new TensorShape(b, h, skv, d));
        using Tensor o = new(q.Shape, DType.F16);
        KnobStore.Set(EngineKnobs.SageAttn, true);
        KnobStore.Set(EngineKnobs.SageF16MinSkv, 1);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            ((IBackend)backend).ScaledDotProductAttention(o, q, k, v, null, 1f / MathF.Sqrt(d), allowF16: true);
            backend.Sync();
            Assert.True(backend.SageAttentionEngaged, $"{name} did not run on SageAttention");
            Check(dir, name, o);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SageAttn);
            KnobStore.Clear(EngineKnobs.SageF16MinSkv);
        }
    }

    /// <summary>Head dim 256 head-major (one and two batches) and token-major, with query and key tails inside a tile.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("flashf16-d256-hm-q300", false, 1, 2, 300, 300)]
    [InlineData("flashf16-d256-hm-b2-q200-k333", false, 2, 3, 200, 333)]
    [InlineData("flashf16-d256-tm-q300-k260", true, 1, 2, 300, 260)]
    public void FlashF16_OutputDigest_IsUnchanged(string name, bool tokenMajor, int b, int h, int sq, int skv)
    {
        if (BaselineDir() is not { } dir) return;
        const int D = 256;
        Random rng = Seeded(name);
        TensorShape qs = tokenMajor ? new(sq, h * D) : new(b, h, sq, D);
        TensorShape ks = tokenMajor ? new(skv, h * D) : new(b, h, skv, D);
        using Tensor q = Make(Gauss(rng, qs.ElementCount), qs);
        using Tensor k = Make(Gauss(rng, ks.ElementCount), ks);
        using Tensor v = Make(Gauss(rng, ks.ElementCount), ks);
        using Tensor o = new(qs, DType.F16);
        KnobStore.Set(EngineKnobs.FlashF16, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            if (tokenMajor) ((IBackend)backend).ScaledDotProductAttentionTokenMajor(o, q, k, v, null, h, D, 1f / 16f, allowF16: true);
            else ((IBackend)backend).ScaledDotProductAttention(o, q, k, v, null, 1f / 16f, allowF16: true);
            backend.Sync();
            Assert.True(backend.FlashF16ExecutionCount > 0, $"{name} did not run on the F16 flash kernel");
            Check(dir, name, o);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.FlashF16);
        }
    }

    private string? BaselineDir()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return null; }
        string? dir = Environment.GetEnvironmentVariable("HARTSY_SDPA_BASELINE_DIR");
        if (string.IsNullOrWhiteSpace(dir)) { _output.WriteLine("SKIPPED: HARTSY_SDPA_BASELINE_DIR is unset"); return null; }
        return dir;
    }

    /// <summary>Hashes <paramref name="o"/> while its backend is alive: the output is device-resident until read.</summary>
    private void Check(string dir, string name, Tensor o)
    {
        ReadOnlySpan<ushort> bits = new(o.DataPointer, checked((int)o.ElementCount));
        Assert.True(bits.ContainsAnyExcept((ushort)0), $"{name} produced an all-zero output");
        string digest = Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(bits)));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, name + ".sha256");
        if (string.Equals(Environment.GetEnvironmentVariable("HARTSY_SDPA_BASELINE_MODE"), "record", StringComparison.OrdinalIgnoreCase))
        {
            File.WriteAllText(path, digest);
            _output.WriteLine($"{name}: recorded {digest}");
            return;
        }
        Assert.True(File.Exists(path), $"no baseline for {name} in {dir}: run this test with HARTSY_SDPA_BASELINE_MODE=record on the base tree first");
        Assert.Equal(File.ReadAllText(path).Trim(), digest);
    }

    // string.GetHashCode is randomized per process.
    private static Random Seeded(string name) => new(name.Aggregate(17, (a, c) => a * 31 + c));

    private static float[] Gauss(Random rng, long n)
    {
        float[] x = new float[n];
        for (long i = 0; i < n; i++)
            x[i] = (float)(Math.Sqrt(-2.0 * Math.Log(1.0 - rng.NextDouble())) * Math.Cos(2.0 * Math.PI * rng.NextDouble()));
        return x;
    }

    private static Tensor Make(float[] values, TensorShape shape)
    {
        using Tensor f32 = new(shape, DType.F32);
        values.AsSpan().CopyTo(new Span<float>(f32.DataPointer, values.Length));
        return f32.CastTo(DType.F16);
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }
}
