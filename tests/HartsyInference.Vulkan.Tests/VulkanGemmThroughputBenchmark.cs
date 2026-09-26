using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>GPU-timed F16 Linear throughput at Krea2's 1024² shapes (4352 tokens, hidden 6144, FFN 16384), so a GEMM
/// kernel change is measured without the rest of a generation around it.</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class VulkanGemmThroughputBenchmark(ITestOutputHelper output)
{
    private const int Iterations = 20;

    [Theory]
    [InlineData(4352, 6144, 6144)]
    [InlineData(4352, 6144, 16384)]
    [InlineData(4352, 16384, 6144)]
    [InlineData(4352, 6144, 1536)]
    [InlineData(4337, 6144, 6144)]
    public void F16Linear(int m, int k, int n)
    {
        try { using VulkanInstance i = new(); if (i.EnumeratePhysicalDevices().Length == 0) return; }
        catch { return; }
        using VulkanBackend backend = new();
        using Tensor a = Filled(new TensorShape(m, k), 1);
        using Tensor w = Filled(new TensorShape(n, k), 2);
        using Tensor o = new(new TensorShape(m, n), DType.F16);
        backend.PreloadWeights([a, w]);
        backend.Linear(o, a, w, null);
        double ms = backend.MeasureGpuTimeMs(Iterations, () => backend.Linear(o, a, w, null)) / Iterations;
        double tflops = 2.0 * m * n * k / (ms * 1e9);
        output.WriteLine($"M={m} K={k} N={n}: {ms:F3} ms, {tflops:F1} TFLOPS");
    }

    private static Tensor Filled(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F16);
        Span<Half> s = t.AsSpan<Half>();
        Random rng = new(seed);
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = (Half)((rng.NextDouble() * 2 - 1) * 0.1);
        }
        return t;
    }
}
