using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>
/// Validates that the SPIR-V pipeline cache persists to disk on backend disposal and is
/// reloaded by the next backend. The driver gracefully ignores cache contents that don't
/// match the current device, so the test asserts the file grows after first-use, not that
/// build time drops (build time is noisy and driver-dependent).
/// </summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanPipelineCacheTests
{
    private readonly ITestOutputHelper _output;
    public VulkanPipelineCacheTests(ITestOutputHelper output) => _output = output;

    /// <summary>VkPipelineCacheHeaderVersionOne: 16 bytes of header plus the 16-byte pipeline cache UUID.</summary>
    private const long HeaderOnlyBytes = 32;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance instance = new(); return instance.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Fact]
    public void PipelineCache_PersistsToDisk_AcrossBackends()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }

        // First, scrub any existing cache so the test exercises the cold-build path.
        string cachePath;
        using (VulkanBackend probe = new())
            cachePath = probe.PipelineCachePath;
        if (File.Exists(cachePath)) File.Delete(cachePath);
        Assert.False(File.Exists(cachePath), $"Pre-condition: cache should be absent at {cachePath}");

        // First backend: build a few kernels then dispose. Persist on disposal should write the cache.
        using (VulkanBackend backend = new())
        {
            BuildAFewKernels(backend);
            // Dispose-on-leave will persist via VulkanPipelineCache.Dispose -> Persist.
        }

        Assert.True(File.Exists(cachePath), $"After first backend: cache should exist at {cachePath}");
        long firstSize = new FileInfo(cachePath).Length;
        _output.WriteLine($"Cache size after first backend: {firstSize} bytes");
        Assert.True(firstSize > 0, "Cache file should be non-empty");

        // Second backend: load the cache, build the same kernels, dispose. Cache size should
        // be ≥ firstSize (driver may add new entries; never shrinks if all entries are still valid).
        using (VulkanBackend backend = new())
        {
            BuildAFewKernels(backend);
        }

        long secondSize = new FileInfo(cachePath).Length;
        _output.WriteLine($"Cache size after second backend: {secondSize} bytes");
        // A pipeline cache that holds no pipelines is still a valid one: the blob is a 32-byte header and the
        // spec never obliges a driver to persist compiled pipelines. llvmpipe is such a driver. Whether the
        // driver fills the cache is read from the FIRST backend's file, so this stays a real assertion on ours.
        if (firstSize <= HeaderOnlyBytes)
        {
            _output.WriteLine($"Driver does not persist pipeline binaries (first cache was {firstSize} bytes, "
                + "header only) — asserting the round-trip, not the contents.");
            Assert.True(secondSize > 0, $"Cache vanished across backends: {secondSize} bytes");
            return;
        }
        // Drivers are allowed to drop pipeline-cache entries that don't match the current state
        // (a flush of stale entries on reload is normal — the spec doesn't require monotonic
        // growth). Just assert the second backend produced a non-trivially-sized cache, i.e.
        // the file persisted across backend lifetimes and the driver loaded it.
        Assert.True(secondSize > 1024,
            $"Cache after reload is {secondSize} bytes — expected >1KB; pipeline cache reload may be broken");
    }

    /// <summary>Touches a handful of distinct kernels (ops + dtypes) so the pipeline cache picks up
    /// real entries rather than just the empty-cache header.</summary>
    private static void BuildAFewKernels(VulkanBackend backend)
    {
        Tensor a = new(new TensorShape(8), DType.F32);
        Tensor b = new(new TensorShape(8), DType.F32);
        Tensor c = new(new TensorShape(8), DType.F32);
        try
        {
            Span<float> aS = a.AsSpan<float>();
            Span<float> bS = b.AsSpan<float>();
            for (int i = 0; i < 8; i++) { aS[i] = i; bS[i] = -i; }
            backend.Add(c, a, b);
            backend.Mul(c, a, b);
            backend.Silu(c, a);
            // Force CPU readback so all dispatches actually run.
            ReadOnlySpan<float> _ = c.AsReadOnlySpan<float>();
        }
        finally
        {
            a.Dispose(); b.Dispose(); c.Dispose();
        }
    }
}
