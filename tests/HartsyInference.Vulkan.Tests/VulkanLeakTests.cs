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
[Trait("Category", "GpuIntegration")]
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

    /// <summary>Regression guard for the dedicated-block pooling fix in <c>VulkanMemoryAllocator</c>:
    /// large (&gt;= 16 MB, <see cref="VulkanMemoryAllocator.DedicatedThreshold"/>) transient buffers used
    /// to be destroyed the instant they emptied (a real <c>vkAllocateMemory</c>/<c>vkFreeMemory</c> pair
    /// on every call — measured at ~30x the cost of reuse, see <c>benchmarks/scoreboards/VULKAN.md</c>).
    /// They're now pooled like slab blocks. This asserts BOTH halves of that fix: (a) VRAM usage still
    /// returns to baseline (no leak from keeping blocks alive) and (b) the live block count stays flat
    /// after the first large allocation (proving reuse actually happens, not just "didn't leak").</summary>
    [Fact]
    public void Vulkan_100Iter_LargeTransient_PoolsInsteadOfReallocating()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();

        // >= DedicatedThreshold (16 MB): 1,310,720 elements * 4 bytes = ~5.24 MB is too small; use a
        // shape whose F32 buffer is unambiguously in the dedicated tier.
        const int Rows = 1280, Cols = 4096;   // 1280*4096*4 bytes = 20 MB
        const int WarmupIterations = 20;   // each Silu call needs TWO ~20 MB blocks (x upload + y output),
                                            // and the transient-upload list only drains every FlushThreshold
                                            // dispatches — this pattern's working set needs ~20 iters to reach
                                            // steady state (empirically: it plateaus exactly at iter 20 and
                                            // stays bit-for-bit flat through iter 100). Baseline must be taken
                                            // AFTER that, or "still warming up" reads as "still leaking."
        const int Iterations = 100;
        const long EpsilonBytes = 4L * 1024 * 1024;   // post-warmup steady state should be exactly flat

        for (int i = 0; i < WarmupIterations; i++)
            RunOneSiluIter(backend, Rows, Cols, iter: -1);
        backend.Sync();

        (long baselineUsed, long baselineReserved, int baselineBlocks, _) = backend.MemoryStats;
        _output.WriteLine($"Baseline (post-warmup): used={baselineUsed / 1024 / 1024} MB, reserved={baselineReserved / 1024 / 1024} MB, blocks={baselineBlocks}");

        for (int i = 0; i < Iterations; i++)
            RunOneSiluIter(backend, Rows, Cols, i);
        backend.Sync();

        (long endUsed, long endReserved, int endBlocks, _) = backend.MemoryStats;
        _output.WriteLine($"After {Iterations} iters: used={endUsed / 1024 / 1024} MB, reserved={endReserved / 1024 / 1024} MB, blocks={endBlocks}");

        long usedDelta = endUsed - baselineUsed;
        Assert.True(usedDelta <= EpsilonBytes,
            $"VRAM grew by {usedDelta / 1024 / 1024} MB after {Iterations} large-buffer iters — possible leak in the pooling fix");
        Assert.True(endBlocks <= baselineBlocks + 1,
            $"Block count grew from {baselineBlocks} to {endBlocks} over {Iterations} iters of the SAME buffer size — " +
            "large blocks are being re-allocated instead of pooled/reused (the exact regression this test guards against).");
    }

    private static void RunOneSiluIter(VulkanBackend backend, int rows, int cols, int iter)
    {
        Tensor x = new(new TensorShape(rows, cols), DType.F32);
        Tensor y = new(new TensorShape(rows, cols), DType.F32);
        try
        {
            Span<float> xS = x.AsSpan<float>();
            xS[0] = 0.001f * iter;   // touch one element; content doesn't matter for this test

            backend.Silu(y, x);

            ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
            float _ = yS[0];
        }
        finally
        {
            x.Dispose(); y.Dispose();
        }
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
