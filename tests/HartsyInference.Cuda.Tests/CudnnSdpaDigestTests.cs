using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Plain fp16 MHA output digests before and after the by-value scale and the strided plan key (head-major and token-major B=1); public API only, so this file also compiles on the base tree.
/// <c>HARTSY_SDPA_BASELINE_MODE=record</c> on base writes them to <c>HARTSY_SDPA_BASELINE_DIR</c>; on head they are compared (unset dir skips).</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaDigestTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaDigestTests(ITestOutputHelper output) => _output = output;

    private static float[] Gauss(Random rng, long n)
    {
        float[] x = new float[n];
        for (long i = 0; i < n; i++)
            x[i] = (float)(Math.Sqrt(-2.0 * Math.Log(1.0 - rng.NextDouble())) * Math.Cos(2.0 * Math.PI * rng.NextDouble()));
        return x;
    }

    private static Tensor Make(float[] values, TensorShape shape, DType dtype)
    {
        Tensor f32 = new(shape, DType.F32);
        values.AsSpan().CopyTo(new Span<float>(f32.DataPointer, values.Length));
        if (dtype == DType.F32) return f32;
        try { return f32.CastTo(dtype); }
        finally { f32.Dispose(); }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("mha-h24-s1024-d128", 1, 24, 1024, 128, false, 0.08838835f, false)]
    [InlineData("mha-h24-s1024-d128-scale03", 1, 24, 1024, 128, false, 0.3f, false)]
    [InlineData("mha-h8-s512-d64-bias-b2", 2, 8, 512, 64, true, 0.125f, false)]
    [InlineData("mha-h4-s256-d96", 1, 4, 256, 96, false, 0.102062073f, false)]
    [InlineData("tm-h24-s1024-d128", 1, 24, 1024, 128, false, 0.08838835f, true)]
    [InlineData("tm-h8-s512-d64-bias", 1, 8, 512, 64, true, 0.125f, true)]
    public void Fp16Mha_OutputDigest_IsUnchangedByTheByValueScaleAndTheStridedKey(string name, int b, int h, int s, int d, bool bias, float scale, bool tokenMajor)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return; }
        string? dir = Environment.GetEnvironmentVariable("HARTSY_SDPA_BASELINE_DIR");
        if (string.IsNullOrWhiteSpace(dir)) { _output.WriteLine("SKIPPED: HARTSY_SDPA_BASELINE_DIR is unset"); return; }
        bool record = string.Equals(Environment.GetEnvironmentVariable("HARTSY_SDPA_BASELINE_MODE"), "record", StringComparison.OrdinalIgnoreCase);

        Random rng = new(name.Aggregate(17, (a, c) => a * 31 + c));   // string.GetHashCode is randomized per process
        TensorShape shape = tokenMajor ? new(s, h * d) : new(b, h, s, d);   // token-major B=1: [S, H·D]
        using Tensor q = Make(Gauss(rng, shape.ElementCount), shape, DType.F16);
        using Tensor k = Make(Gauss(rng, shape.ElementCount), shape, DType.F16);
        using Tensor v = Make(Gauss(rng, shape.ElementCount), shape, DType.F16);
        using Tensor? mask = bias ? Make(Gauss(rng, (long)s * s), new TensorShape(1, 1, s, s), DType.F32) : null;
        using Tensor o = new(shape, DType.F16);

        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new(0, ptxDir);
        if (tokenMajor) ((IBackend)backend).ScaledDotProductAttentionTokenMajor(o, q, k, v, mask, h, d, scale, allowF16: true);
        else ((IBackend)backend).ScaledDotProductAttention(o, q, k, v, mask, scale, allowF16: true);
        backend.Sync();
        Assert.True(backend.CudnnSdpaEngaged, $"{name} did not run on cuDNN, so it says nothing about the plan");
        string digest = Convert.ToHexString(SHA256.HashData(new ReadOnlySpan<byte>(o.DataPointer, checked((int)(o.ElementCount * sizeof(ushort))))));

        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, name + ".sha256");
        if (record)
        {
            File.WriteAllText(path, digest);
            _output.WriteLine($"{name}: recorded {digest}");
            return;
        }
        Assert.True(File.Exists(path), $"no baseline for {name} in {dir}: run this test with HARTSY_SDPA_BASELINE_MODE=record on the base tree first");
        Assert.Equal(File.ReadAllText(path).Trim(), digest);
    }
}
