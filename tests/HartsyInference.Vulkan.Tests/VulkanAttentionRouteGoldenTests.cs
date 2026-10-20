using System.Reflection;
using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The dispatch route and output digest of every Vulkan attention shape class. The routes pin the planner rows;
/// the digests, printed per case, compare a build against the build before it (a build without the route recorder only
/// prints them), so a refactor that moves no kernel shows identical bytes case by case.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanAttentionRouteGoldenTests(ITestOutputHelper output)
{
    /// <param name="Entry">sdpa (head-major), tm (token-major), flash (host positions), dev (device positions), dev0 (no buffer).</param>
    /// <param name="Mask">none, bias (<c>[Sq,Skv]</c>), perhead (<c>[1,H,Sq,Skv]</c>) or key (<c>[1,1,1,Skv]</c>).</param>
    /// <param name="Expected">The route with a cooperative-matrix-2 device; <paramref name="ExpectedScalar"/> without one.</param>
    private sealed record Case(string Name, string Entry, DType Dtype, int Hq, int Hkv, int Sq, int Skv, int D, string Mask,
        bool AllowF16, string Expected, string? ExpectedScalar = null, bool Causal = true, float Softcap = 0f, int Window = 0);

    private static readonly Case[] Table =
    [
        new("f16-d128", "sdpa", DType.F16, 4, 4, 256, 256, 128, "none", true, "A.1", "A.2"),
        new("f32-allowf16-d128", "sdpa", DType.F32, 4, 4, 256, 256, 128, "none", true, "A.1", "A.2"),
        new("f32-d128", "sdpa", DType.F32, 4, 4, 256, 256, 128, "none", false, "A.2"),
        new("f16-d40", "sdpa", DType.F16, 4, 4, 256, 256, 40, "none", true, "A.2"),
        new("f32-d256", "sdpa", DType.F32, 2, 2, 128, 128, 256, "none", false, "A.3"),
        new("f32-gqa-d64-bias", "sdpa", DType.F32, 6, 3, 65, 129, 64, "bias", true, "A.1", "A.2"),
        new("f32-d64-perhead", "sdpa", DType.F32, 4, 4, 64, 64, 64, "perhead", true, "A.1", "A.2"),
        new("f32-d64-key", "sdpa", DType.F32, 4, 4, 64, 64, 64, "key", true, "A.1", "A.2"),
        new("tm-f16-d128", "tm", DType.F16, 4, 4, 256, 256, 128, "none", true, "B.1", "B.2/A.2"),
        new("tm-gqa-f32-d128", "tm", DType.F32, 48, 12, 300, 300, 128, "none", true, "B.1", "B.2/A.2"),
        new("tm-key-f32-d64", "tm", DType.F32, 4, 4, 97, 97, 64, "key", true, "B.1", "B.2/A.2"),
        new("tm-f32-d40", "tm", DType.F32, 4, 4, 64, 64, 40, "none", true, "B.2/A.2"),
        new("tm-f32-d256", "tm", DType.F32, 2, 2, 64, 64, 256, "none", false, "B.2/A.3"),
        new("flash-decode", "flash", DType.F32, 8, 2, 1, 256, 128, "none", false, "C.3"),
        new("flash-decode-window", "flash", DType.F32, 8, 2, 1, 256, 64, "none", false, "C.3", Window: 32),
        new("flash-prefill", "flash", DType.F32, 16, 8, 96, 96, 64, "none", false, "C.2", "C.4"),
        new("flash-prefill-short", "flash", DType.F32, 16, 8, 16, 16, 64, "none", false, "C.4"),
        new("flash-noncausal", "flash", DType.F32, 16, 8, 96, 96, 64, "none", false, "C.4", Causal: false),
        new("flash-softcap", "flash", DType.F32, 8, 2, 4, 32, 64, "none", false, "C.1", Softcap: 30f),
        new("flash-d256", "flash", DType.F32, 2, 1, 4, 32, 256, "none", false, "C.1"),
        new("dev-decode", "dev", DType.F32, 8, 2, 1, 256, 128, "none", false, "D.1"),
        new("dev-prefill", "dev", DType.F32, 8, 2, 4, 64, 64, "none", false, "D.2"),
        new("dev-softcap", "dev", DType.F32, 8, 2, 1, 64, 64, "none", false, "throws", Softcap: 30f),
        new("dev-pos0", "dev0", DType.F32, 8, 2, 1, 256, 128, "none", false, "D.4/C.3"),
    ];

    public static IEnumerable<object[]> Names => Table.Select(c => new object[] { c.Name });

    /// <summary>The route recorder, absent from builds before the attention rows.</summary>
    private static readonly PropertyInfo? RouteProperty =
        typeof(VulkanBackend).GetProperty("LastAttentionRoute", BindingFlags.NonPublic | BindingFlags.Static);

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private static Tensor Random(TensorShape shape, DType dt, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor f32 = new(shape, DType.F32);
        Span<float> p = f32.AsSpan<float>();
        Random rng = new(seed);
        for (int i = 0; i < p.Length; i++) p[i] = lo + (float)rng.NextDouble() * (hi - lo);
        if (dt == DType.F32) return f32;
        try { return f32.CastTo(dt); } finally { f32.Dispose(); }
    }

    [Theory]
    [MemberData(nameof(Names))]
    public void TakesTheGoldenRoute(string name)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        Case c = Table.Single(x => x.Name == name);
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (c.Expected == "throws")
        {
            Assert.Throws<NotSupportedException>(() => Run(backend, c));
            return;
        }
        using Tensor result = Run(backend, c);
        string digest = Convert.ToHexString(SHA256.HashData(result.AsReadOnlySpan<byte>()))[..16];
        string? route = RouteProperty?.GetValue(null) as string;
        output.WriteLine($"GOLDEN {c.Name} route={route ?? "n/a"} sha={digest}");
        if (RouteProperty is null) return;
        string expected = backend.SupportsTokenMajorGqaAttention ? c.Expected : c.ExpectedScalar ?? c.Expected;
        Assert.Equal(expected, route);
    }

    private static Tensor Run(VulkanBackend backend, Case c)
    {
        IBackend b = backend;
        float scale = 1f / MathF.Sqrt(c.D);
        if (c.Entry is "flash" or "dev" or "dev0")
        {
            using Tensor q = Random(new TensorShape(1, c.Hq, c.Sq, c.D), DType.F32, 1);
            using Tensor k = Random(new TensorShape(1, c.Hkv, c.Skv, c.D), DType.F32, 2);
            using Tensor v = Random(new TensorShape(1, c.Hkv, c.Skv, c.D), DType.F32, 3);
            Tensor o = new(new TensorShape(1, c.Hq, c.Sq, c.D), DType.F32);
            int qOffset = c.Causal ? c.Skv - c.Sq : 0;
            if (c.Entry == "flash")
            {
                b.FlashAttention(o, q, k, v, c.Skv, c.Hq / c.Hkv, c.Causal, qOffset, scale, c.Softcap, sink: null, c.Window);
                return o;
            }
            ulong pos = c.Entry == "dev0" ? 0 : b.AllocDevicePos();
            try
            {
                if (pos != 0) b.WriteDevicePos(pos, c.Skv, qOffset);
                b.FlashAttentionDev(o, q, k, v, c.Skv, c.Hq / c.Hkv, c.Causal, qOffset, scale, pos, c.Softcap, c.Window);
            }
            catch
            {
                o.Dispose();
                throw;
            }
            finally
            {
                if (pos != 0) b.FreeDevicePos(pos);
            }
            return o;
        }
        bool tm = c.Entry == "tm";
        TensorShape qs = tm ? new(c.Sq, c.Hq * c.D) : new(1, c.Hq, c.Sq, c.D);
        TensorShape ks = tm ? new(c.Skv, c.Hkv * c.D) : new(1, c.Hkv, c.Skv, c.D);
        using Tensor qt = Random(qs, c.Dtype, 1);
        using Tensor kt = Random(ks, c.Dtype, 2);
        using Tensor vt = Random(ks, c.Dtype, 3);
        Tensor ot = new(qs, c.Dtype);
        using Tensor? mask = c.Mask switch
        {
            "bias" => Random(new TensorShape(c.Sq, c.Skv), DType.F32, 5, -2f, 2f),
            "perhead" => Random(new TensorShape(1, c.Hq, c.Sq, c.Skv), DType.F32, 5, -2f, 2f),
            "key" => Random(new TensorShape(1, 1, 1, c.Skv), DType.F32, 5, -2f, 2f),
            _ => null,
        };
        if (tm && c.Hkv != c.Hq) b.ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, mask, c.Hq, c.Hkv, c.D, scale, c.AllowF16);
        else if (tm) b.ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, mask, c.Hq, c.D, scale, c.AllowF16);
        else b.ScaledDotProductAttention(ot, qt, kt, vt, mask, scale, c.AllowF16);
        return ot;
    }
}
