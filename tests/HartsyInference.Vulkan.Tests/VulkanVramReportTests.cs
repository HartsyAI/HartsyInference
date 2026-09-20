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
    /// <summary>~1 GB of F32, big enough that a report which ignores it is obvious.</summary>
    private const int Rows = 8192, Cols = 32768;

    /// <summary>How much of a residency has to show up in the reported figure.</summary>
    /// <remarks>A quarter, not all of it. This backend shares the card with whatever else is running, and the
    /// driver's figure moves with those processes too — the assertion has to survive a co-tenant taking or
    /// releasing a few hundred megabytes between two probes, while still failing a figure that does not move.</remarks>
    private const int ObservedFraction = 4;

    private readonly ITestOutputHelper _output;

    public VulkanVramReportTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void FreeMemoryBytes_IsTheSameNumberAsGetVramInfo()
    {
        if (!BackendGate.TryOpen("vulkan", _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using VulkanBackend backend = (VulkanBackend)opened!;

        (long free, long total) = backend.GetVramInfo();
        // Through the interface deliberately: the fix is the interface's own default, so every backend that
        // answers GetVramInfo answers this too, including ones that never inherit the shared GPU base.
        long viaOldName = ((IBackend)backend).FreeMemoryBytes();

        _output.WriteLine($"free={free >> 20} MB of total={total >> 20} MB, budget extension={backend.Vk.HasMemoryBudget}");
        Assert.True(total > 0, "the device reported no device-local memory at all");
        Assert.InRange(free, 1, total);
        // Two probes of a live figure differ when something else on the card moves between them, so this is a
        // sanity bound rather than an equality — what it catches is the two spellings reading different sources,
        // which is what they did before (one of them returned a constant zero).
        Assert.True(Math.Abs(free - viaOldName) < (2L << 30),
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
        // Slack for the allocator rounding up to a slab, and for anything else on the card moving between the two
        // probes: the driver's figure is live, so it answers for every process, not just this one.
        Assert.True(before - duringFree >= bytes / ObservedFraction,
            $"a {bytes >> 20} MB residency moved the free figure by only {(before - duringFree) >> 20} MB");
        Assert.True(after >= duringFree + (bytes / ObservedFraction),
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
        try
        {
            // Which path answered, asserted rather than assumed: the two produce different numbers, and a query
            // that quietly stopped answering would otherwise read as a card that happens to be busy.
            Assert.True(backend.TryQueryDriverVram(out long driverFree),
                "the extension is present but the driver query did not answer");
            (long free, long total) = backend.GetVramInfo();
            (_, long reserved, _, _) = backend.MemoryStats;
            long arithmetic = total - reserved;

            _output.WriteLine($"driver says {free >> 20} MB free (direct query {driverFree >> 20} MB); "
                + $"total {total >> 20} MB minus {reserved >> 20} MB reserved = {arithmetic >> 20} MB");
            Assert.True(free <= arithmetic,
                $"the driver reported MORE free ({free >> 20} MB) than this backend's own arithmetic allows "
                + $"({arithmetic >> 20} MB) — budget and usage are likely swapped");
        }
        finally
        {
            // Before the assertions could throw: a leaked gigabyte of device memory would fail the NEXT test in
            // this class rather than this one, turning one real failure into a cascade.
            backend.FreeWeights([weight]);
        }
    }
}
