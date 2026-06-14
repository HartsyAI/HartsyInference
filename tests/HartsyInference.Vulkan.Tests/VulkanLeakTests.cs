using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>
/// Phase 3.5 acceptance gate #7: 100-iteration loop returns to baseline VRAM. Validates that the
/// activation cache, transient upload buffers, and deferred-free list don't leak across step
/// boundaries — the exact pattern that bit us as deviation #5 (transient buffer leak).
/// </summary>
public sealed class VulkanLeakTests
{
    private readonly ITestOutputHelper _output;
    public VulkanLeakTests(ITestOutputHelper output) => _output = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance instance = new(); return instance.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    /// <summary>Runs 100 iterations of MatMul + activation reads with explicit Tensor disposal,
    /// then verifies device-memory usage has returned to within a small epsilon of the baseline.</summary>
    [Fact]
    public void Vulkan_100Iter_DeviceMemory_Returns_To_Baseline()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();

        const int M = 64, K = 256, N = 64;
        const int Iterations = 100;
        const long EpsilonBytes = 16L * 1024 * 1024;   // 16 MB — covers slab grow-once-then-stable

        // Warm-up: get past first-iteration slab/pipeline allocations. The baseline is taken
        // *after* warm-up so we measure steady-state, not the cost of bringing the backend up.
        RunOneIter(backend, M, K, N, iter: -1);
        backend.Sync();

        (long baselineUsed, long baselineReserved, int baselineBlocks, _) = backend.MemoryStats;
        _output.WriteLine($"Baseline: used={baselineUsed / 1024 / 1024} MB, reserved={baselineReserved / 1024 / 1024} MB, slabs={baselineBlocks}");

        for (int i = 0; i < Iterations; i++)
            RunOneIter(backend, M, K, N, i);

        backend.Sync();

        (long endUsed, long endReserved, int endBlocks, long endCached) = backend.MemoryStats;
        _output.WriteLine($"After {Iterations} iters: used={endUsed / 1024 / 1024} MB, reserved={endReserved / 1024 / 1024} MB, slabs={endBlocks}, cachedTensorBytes={endCached / 1024 / 1024} MB");

        long delta = endUsed - baselineUsed;
        _output.WriteLine($"Delta: used grew by {delta / 1024} KB ({delta} bytes)");
        Assert.True(delta <= EpsilonBytes,
            $"VRAM grew by {delta / 1024 / 1024} MB after {Iterations} iters (epsilon: {EpsilonBytes / 1024 / 1024} MB) — possible activation/transient/deferred-free leak");
    }

    private static void RunOneIter(VulkanBackend backend, int M, int K, int N, int iter)
    {
        Tensor a = new(new TensorShape(M, K), DType.F32);
        Tensor b = new(new TensorShape(K, N), DType.F32);
        Tensor c = new(new TensorShape(M, N), DType.F32);
        try
        {
            Span<float> aS = a.AsSpan<float>();
            Span<float> bS = b.AsSpan<float>();
            for (int i = 0; i < M * K; i++) aS[i] = 0.001f * (i + iter);
            for (int i = 0; i < K * N; i++) bS[i] = 0.001f * (i - iter);

            backend.MatMul(c, a, b);

            // Force a CPU-side read so the activation cache fires its lazy-sync callback,
            // which is the path that historically left transient buffers leaked across steps.
            ReadOnlySpan<float> cS = c.AsReadOnlySpan<float>();
            // Touch one element to make sure the read isn't elided.
            float _ = cS[0];
        }
        finally
        {
            a.Dispose(); b.Dispose(); c.Dispose();
        }
    }
}
