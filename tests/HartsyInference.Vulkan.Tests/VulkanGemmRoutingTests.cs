using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The GEMM sites that used to be tiled-only — the batched product and the convolution's im2col product —
/// reach the cooperative-matrix kernel through <c>DispatchGemm</c> when their operands are F16 and 16-aligned, and
/// agree with the same op computed in F32 on the tiled kernel (the reference the cross-backend parity tests hold
/// to the CPU).</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanGemmRoutingTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private static Tensor Rand(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return t;
    }

    private float MaxRelError(Tensor f16Result, Tensor f32Result)
    {
        using Tensor wide = f16Result.CastTo(DType.F32);
        ReadOnlySpan<float> a = wide.AsReadOnlySpan<float>();
        ReadOnlySpan<float> b = f32Result.AsReadOnlySpan<float>();
        float maxErr = 0f, maxAbs = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            maxErr = MathF.Max(maxErr, MathF.Abs(a[i] - b[i]));
            maxAbs = MathF.Max(maxAbs, MathF.Abs(b[i]));
        }
        float rel = maxAbs > 0 ? maxErr / maxAbs : maxErr;
        _out.WriteLine($"maxErr={maxErr:E3} maxAbs={maxAbs:E3} rel={rel:E3}");
        return rel;
    }

    [Fact]
    public void BatchedMatMul_OnF16_RunsOnCoopmat_AndMatchesTheF32Product()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Vk.HasCooperativeMatrix) { _out.WriteLine("SKIPPED: no cooperative matrix on this device"); return; }
        const int B = 3, M = 64, K = 64, N = 128;
        using Tensor a32 = Rand(new TensorShape(B, M, K), 1);
        using Tensor b32 = Rand(new TensorShape(B, K, N), 2);
        using Tensor out32 = new(new TensorShape(B, M, N), DType.F32);
        backend.BatchedMatMul(out32, a32, b32);
        backend.Sync();

        using Tensor a16 = a32.CastTo(DType.F16);
        using Tensor b16 = b32.CastTo(DType.F16);
        using Tensor out16 = new(new TensorShape(B, M, N), DType.F16);
        (_, long coopBefore, _) = backend.GemmEngagementCounts;
        backend.BatchedMatMul(out16, a16, b16);
        backend.Sync();
        (_, long coopAfter, _) = backend.GemmEngagementCounts;

        Assert.Equal(B, coopAfter - coopBefore);
        Assert.True(MaxRelError(out16, out32) < 2e-2f, "the F16 cooperative-matrix product diverges from the F32 tiled one");
    }

    [Fact]
    public void Conv2D_OnF16_RunsOnCoopmat_AndMatchesTheF32Convolution()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = new();
        if (!backend.Vk.HasCooperativeMatrix) { _out.WriteLine("SKIPPED: no cooperative matrix on this device"); return; }
        const int batch = 2, cin = 32, cout = 64, h = 16, w = 16;   // K = 32·9 = 288, N = 256, M = 64: all 16-aligned
        using Tensor x32 = Rand(new TensorShape(batch, cin, h, w), 3);
        using Tensor w32 = Rand(new TensorShape(cout, cin, 3, 3), 4);
        using Tensor bias32 = Rand(new TensorShape(cout), 5);
        using Tensor out32 = new(new TensorShape(batch, cout, h, w), DType.F32);
        backend.Conv2D(out32, x32, w32, bias32, 1, 1, 1, 1);
        backend.Sync();

        using Tensor x16 = x32.CastTo(DType.F16);
        using Tensor w16 = w32.CastTo(DType.F16);
        using Tensor bias16 = bias32.CastTo(DType.F16);
        using Tensor out16 = new(new TensorShape(batch, cout, h, w), DType.F16);
        (_, long coopBefore, _) = backend.GemmEngagementCounts;
        backend.Conv2D(out16, x16, w16, bias16, 1, 1, 1, 1);
        backend.Sync();
        (_, long coopAfter, _) = backend.GemmEngagementCounts;

        Assert.Equal(batch, coopAfter - coopBefore);
        Assert.True(MaxRelError(out16, out32) < 2e-2f, "the F16 cooperative-matrix convolution diverges from the F32 tiled one");
    }
}
