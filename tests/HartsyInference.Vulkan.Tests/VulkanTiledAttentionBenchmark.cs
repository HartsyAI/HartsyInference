using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>GPU-timed per-call cost of dense F32 attention at Anima's shape (4096 tokens, 16 heads, head dim 128): the scalar
/// sdpa_flash route against the query-tiled kernel, and their ratio. Timing only; correctness lives in
/// <see cref="VulkanTiledAttentionTests"/>. Opt-in: <c>dotnet test --filter Category=GpuBenchmark</c>.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class VulkanTiledAttentionBenchmark(ITestOutputHelper output)
{
    private const int Seq = 4096, Heads = 16, HeadDim = 128;
    private const int ScalarIterations = 3, TiledIterations = 20;

    [Fact]
    public void AnimaShape_ScalarVersusTiled()
    {
        try
        {
            using VulkanInstance instance = new();
            if (instance.EnumeratePhysicalDevices().Length == 0) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        }
        catch (Exception e)
        {
            output.WriteLine($"SKIPPED: {e.Message}");
            return;
        }

        using VulkanBackend backend = VulkanTestDevice.Create();
        float scale = 1f / MathF.Sqrt(HeadDim);
        using Tensor q = new(new TensorShape(1, Heads, Seq, HeadDim), DType.F32);
        using Tensor k = new(new TensorShape(1, Heads, Seq, HeadDim), DType.F32);
        using Tensor v = new(new TensorShape(1, Heads, Seq, HeadDim), DType.F32);
        using Tensor o = new(new TensorShape(1, Heads, Seq, HeadDim), DType.F32);
        System.Random rng = new(9);
        foreach (Tensor t in new[] { q, k, v })
        {
            Span<float> p = t.AsSpan<float>();
            for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        backend.EnableTiledAttention = false;
        backend.ScaledDotProductAttention(o, q, k, v, null, scale);
        string scalarRoute = VulkanBackend.LastAttentionRoute ?? "?";
        double scalarMs = backend.MeasureGpuTimeMs(ScalarIterations,
            () => backend.ScaledDotProductAttention(o, q, k, v, null, scale)) / ScalarIterations;

        backend.EnableTiledAttention = true;
        backend.ScaledDotProductAttention(o, q, k, v, null, scale);
        string tiledRoute = VulkanBackend.LastAttentionRoute ?? "?";
        double tiledMs = backend.MeasureGpuTimeMs(TiledIterations,
            () => backend.ScaledDotProductAttention(o, q, k, v, null, scale)) / TiledIterations;

        // Two matmuls per head: QK^T and PV, each 2*Seq*Seq*HeadDim flops.
        double gflops = 4.0 * Seq * Seq * HeadDim * Heads / (tiledMs * 1e6);
        output.WriteLine($"{backend.Vk.DeviceName}: F32 sdpa {Seq} tokens x {Heads} heads x {HeadDim}: "
            + $"scalar ({scalarRoute}) {scalarMs:F3} ms/call, tiled ({tiledRoute}) {tiledMs:F3} ms/call, "
            + $"speedup {scalarMs / tiledMs:F1}x, tiled {gflops:F0} GFLOP/s");
        Assert.Equal("A.2t", tiledRoute);
    }
}
