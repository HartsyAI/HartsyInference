using HartsyInference.Core.Backends;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Gpu.Tests;

/// <summary>What every GPU backend must answer about device memory, asked of each of them.
///
/// <para>The Vulkan suite checks this too, but it cannot catch the failure that matters here: there,
/// <c>FreeMemoryBytes</c> literally IS <c>GetVramInfo().FreeBytes</c> through the interface default, so the two
/// can never disagree. CUDA keeps its own override reading a different route to the driver, which is exactly the
/// shape that drifts — and until this ran on both, nothing compared them.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VramReportContractTests
{
    /// <summary>How far two probes of a live figure may drift between calls without meaning anything.</summary>
    private const long ProbeSlackBytes = 1L << 30;

    private readonly ITestOutputHelper _output;

    public VramReportContractTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void TheTwoSpellingsOfFreeVramAgree(string kind)
    {
        if (!BackendGate.TryOpen(kind, _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using IBackend backend = opened!;

        (long free, long total) = backend.GetVramInfo();
        long viaOldName = backend.FreeMemoryBytes();

        _output.WriteLine($"[{kind}] GetVramInfo {free >> 20} MB free of {total >> 20} MB; "
            + $"FreeMemoryBytes {viaOldName >> 20} MB");
        Assert.True(total > 0, $"[{kind}] reported no device memory at all");
        // Zero free is a legal answer from a card another process has filled; the total is what says the number
        // is real.
        Assert.InRange(free, 0, total);
        Assert.InRange(viaOldName, 0, total);
        Assert.True(Math.Abs(free - viaOldName) < ProbeSlackBytes,
            $"[{kind}] the two spellings disagree: {free >> 20} MB vs {viaOldName >> 20} MB — they are reading "
            + "different sources, which on a multi-GPU placement can mean different devices");
    }
}
