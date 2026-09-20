using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>What this backend tells a planner about device memory.
///
/// <para><c>FreeMemoryBytes</c> returned zero here until this change, and zero is not "unknown" to the code that
/// reads it — it is "nothing fits". The LLM preload kept no weights at all and the VAE decoder always tiled, on a
/// card with twenty-two gigabytes free.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanVramReportTests
{
    private readonly ITestOutputHelper _output;

    public VulkanVramReportTests(ITestOutputHelper output) => _output = output;

    /// <summary>~1 GB of F32, big enough that a report which ignores it is obvious.</summary>
    private const int Rows = 8192, Cols = 32768;

    [Fact]
    public void FreeMemoryBytes_IsTheSameNumberAsGetVramInfo()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;

        (long free, long total) = backend.GetVramInfo();
        long viaOldName = backend.FreeMemoryBytes();

        _output.WriteLine($"free={free >> 20} MB of total={total >> 20} MB, budget extension={backend.Vk.HasMemoryBudget}");
        Assert.True(total > 0, "the device reported no device-local memory at all");
        Assert.InRange(free, 1, total);
        // Two probes of a live figure can differ if something else on the card moved between them, but not by
        // anything like a gigabyte on an otherwise idle test host.
        Assert.True(Math.Abs(free - viaOldName) < (1L << 30),
            $"the two spellings disagree: {free >> 20} MB vs {viaOldName >> 20} MB");
    }

    /// <summary>The number has to move when memory is taken and given back, or a planner is reading a constant.</summary>
    [Fact]
    public void GetVramInfo_FallsWhenThisBackendAllocatesAndRecoversWhenItFrees()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;

        long before = backend.GetVramInfo().FreeBytes;
        long duringFree;
        long bytes;
        Tensor weight = new(new TensorShape(Rows, Cols), DType.F32);
        try
        {
            bytes = Tensor.ComputeByteSize(weight.Shape, weight.DType);
            backend.PreloadWeights([weight]);
            backend.Sync();
            duringFree = backend.GetVramInfo().FreeBytes;
        }
        finally
        {
            backend.FreeWeights([weight]);
            weight.Dispose();
        }
        backend.TrimMemoryPool();
        long after = backend.GetVramInfo().FreeBytes;

        _output.WriteLine($"free {before >> 20} MB -> {duringFree >> 20} MB with {bytes >> 20} MB resident "
            + $"-> {after >> 20} MB after release");
        // Allow slack for the allocator rounding up to a slab and for anything else on the card; the point is that
        // most of a gigabyte showed up in the number, not that it accounted for every byte.
        Assert.True(before - duringFree >= bytes / 2,
            $"a {bytes >> 20} MB residency moved the free figure by only {(before - duringFree) >> 20} MB");
        Assert.True(after >= duringFree + (bytes / 2),
            $"releasing {bytes >> 20} MB recovered only {(after - duringFree) >> 20} MB");
    }

    /// <summary>Where the driver answers, its number must be no larger than the allocator's own arithmetic — the
    /// budget accounts for everything else on the card, which this backend's bookkeeping cannot see.</summary>
    /// <remarks>Equality is the correct result on a card with nothing else running, so this is an upper bound
    /// rather than a strict inequality. It is the invariant that fails if the budget query is wired to the wrong
    /// heaps or reads usage and budget the wrong way round, which is the mistake available here.</remarks>
    [Fact]
    public void GetVramInfo_NeverReportsMoreFreeThanTheAllocatorsOwnArithmetic()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;
        if (!backend.Vk.HasMemoryBudget)
        {
            _output.WriteLine("SKIPPED: VK_EXT_memory_budget unavailable, so the fallback IS the arithmetic");
            return;
        }

        using Tensor weight = new(new TensorShape(Rows, Cols), DType.F32);
        backend.PreloadWeights([weight]);
        backend.Sync();

        (long free, long total) = backend.GetVramInfo();
        (_, long reserved, _, _) = backend.MemoryStats;
        long arithmetic = total - reserved;

        _output.WriteLine($"driver says {free >> 20} MB free; total {total >> 20} MB minus {reserved >> 20} MB "
            + $"reserved = {arithmetic >> 20} MB");
        Assert.True(free <= arithmetic,
            $"the driver reported MORE free ({free >> 20} MB) than this backend's own arithmetic allows "
            + $"({arithmetic >> 20} MB) — budget and usage are likely swapped");

        backend.FreeWeights([weight]);
    }
}
