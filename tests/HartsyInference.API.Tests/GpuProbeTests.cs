using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.API.Tests;

/// <summary>Asking whether a GPU works, rather than whether one exists.
///
/// <para><c>CudaContext.IsAvailable</c> answers the second question, and everything between the two is
/// untested by it: kernels compiled for another architecture, a PTX directory that did not ship with the
/// build, a card with no free memory, a driver and toolkit that disagree. All of those say yes and then throw
/// on the first real operation — in the middle of somebody's request rather than at startup, which is where a
/// host would rather find out.</para></summary>
public sealed class GpuProbeTests
{
    private readonly ITestOutputHelper _out;
    public GpuProbeTests(ITestOutputHelper o) => _out = o;

    /// <summary>Whatever the probe decides, <c>auto</c> agrees with it and a failure has a reason.
    ///
    /// <para>Deliberately not "the probe passes here". It does not pass on the machine this was written on: the
    /// shipped kernels target sm_80 and the card is sm_75, so the driver's JIT refuses them — while
    /// <c>IsAvailable</c> happily returns true, which is the entire problem. Asserting the outcome would have
    /// meant deleting the test on the one machine that proved it was needed.</para></summary>
    [Fact]
    public void AutoFollowsTheProbe_AndAFailureSaysWhy()
    {
        bool available = CudaContext.IsAvailable();
        bool ok = BackendFactory.ProbeCuda();
        _out.WriteLine($"IsAvailable: {available} ({CudaContext.LastUnavailableReason ?? "no reason"})");
        _out.WriteLine($"probe: {(ok ? "passed" : "failed")} {BackendFactory.CudaProbeFailureReason}");

        // A failed CUDA probe means Vulkan gets asked, not that CPU wins: this machine's card answers on one API
        // after refusing the other, and that is the case the fallthrough exists for.
        Assert.Equal(ok ? "cuda" : BackendFactory.ProbeVulkan() ? "vulkan" : "cpu",
            BackendFactory.ResolveProbed("auto"));
        if (ok)
        {
            Assert.Null(BackendFactory.CudaProbeFailureReason);
        }
        else
        {
            Assert.False(string.IsNullOrWhiteSpace(BackendFactory.CudaProbeFailureReason),
                "the probe failed without saying why, which is the situation it exists to end");
        }
    }

    /// <summary>A software rasterizer is not a GPU. Mesa's lavapipe enumerates as a Vulkan device on a machine with
    /// no graphics hardware at all, and it is a CPU implementation of Vulkan, so letting it win <c>auto</c> would
    /// route the engine onto something slower than the CPU backend it was chosen over. Linux CI images ship it, so
    /// this is the default case on a runner rather than an exotic one.</summary>
    [Fact]
    public void ASoftwareRasterizerDoesNotCountAsAVulkanGpu()
    {
        if (VulkanContext.IsAvailable())
        {
            _out.WriteLine($"Real Vulkan GPU present here ({VulkanContext.GetDeviceCount()}); nothing to check.");
            return;
        }
        // Unavailable must mean unavailable all the way down: no probe success, and never the auto answer.
        Assert.False(BackendFactory.ProbeVulkan());
        Assert.NotEqual("vulkan", BackendFactory.ResolveProbed("auto"));
        Assert.NotEqual("vulkan", BackendFactory.Resolve("auto"));
        Assert.False(string.IsNullOrWhiteSpace(VulkanContext.LastUnavailableReason),
            "a host that cannot offer Vulkan has to be able to tell the user why");
    }

    [Fact]
    public void AnExplicitSelectorIsNeverProbed()
    {
        // Someone who wrote "cpu" gets cpu whatever the hardware is, and someone who wrote "cuda" gets a real
        // error from the backend rather than a silent downgrade. Only "auto" is a question.
        Assert.Equal("cpu", BackendFactory.ResolveProbed("cpu"));
        Assert.Equal("cuda", BackendFactory.ResolveProbed("cuda"));
        Assert.Equal("vulkan", BackendFactory.ResolveProbed("vulkan:1"));
    }
}
