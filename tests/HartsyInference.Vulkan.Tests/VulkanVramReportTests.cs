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
        // Zero is a legal answer — a card another process has filled — so the range starts there. It is the total
        // that says whether the number is real.
        Assert.InRange(free, 0, total);
        // The regression this exists for is one spelling reading a different source from the other; the old name
        // used to be a constant zero. So both sides are bounded, not just their difference: a 2 GB window around
        // a card with 1.5 GB free would otherwise swallow a return to that constant.
        Assert.InRange(viaOldName, 0, total);
        Assert.True(free == 0 || viaOldName > 0,
            $"GetVramInfo reports {free >> 20} MB free while the older spelling reports {viaOldName} bytes");
        // Two probes of a live figure differ when something else on the card moves between them, so the window is
        // a sanity bound rather than an equality.
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
        try
        {
            // Inside the try: a gigabyte upload is the likeliest thing here to throw, and the release below has to
            // cover it.
            backend.PreloadWeights([weight]);
            backend.Sync();

            // Which path answered, asserted rather than assumed — and that GetVramInfo actually RETURNED it. With
            // only the first of these, deleting the driver branch from GetVramInfo leaves the test green: the
            // fallback equals the arithmetic, and an upper bound holds at equality.
            Assert.True(backend.TryQueryDriverVram(out long driverFree, out long driverTotal),
                "the extension is present but the driver query did not answer");
            (long free, long total) = backend.GetVramInfo();
            (_, long reserved, _, _) = backend.MemoryStats;
            long arithmetic = total - reserved;

            _output.WriteLine($"driver says {free >> 20} MB free of {driverTotal >> 20} MB; GetVramInfo reports "
                + $"{free >> 20} MB of {total >> 20} MB; arithmetic would say {arithmetic >> 20} MB");
            Assert.Equal(driverTotal, total);
            Assert.True(Math.Abs(free - driverFree) < (1L << 30),
                $"GetVramInfo reported {free >> 20} MB where the driver query says {driverFree >> 20} MB — "
                + "the driver branch is not the one that answered");
            Assert.True(free <= arithmetic,
                $"the driver reported MORE free ({free >> 20} MB) than this backend's own arithmetic allows "
                + $"({arithmetic >> 20} MB) — budget and usage are likely swapped");
        }
        finally
        {
            // Not for the next test's sake — each opens its own backend and disposes it — but so a failure here is
            // reported against a card in the state the next assertion expects.
            backend.FreeWeights([weight]);
        }
    }
}
