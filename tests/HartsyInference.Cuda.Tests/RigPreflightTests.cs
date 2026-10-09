using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// Rig preflight: fails unless CUDA is usable on this machine, so a GPU suite cannot pass by skipping every test. Run it
/// first (docs/Checklists/MOE_RIG_READINESS.md, step 1). Logs the device so the run record names the hardware.
/// </summary>
public sealed class RigPreflightTests
{
    private readonly ITestOutputHelper _output;

    public RigPreflightTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void CudaIsUsable_OnThisMachine()
    {
        Assert.True(CudaContext.IsAvailable(), $"CUDA is not usable here: {CudaContext.LastUnavailableReason}");
        using CudaContext context = new();
        _output.WriteLine($"device {context.DeviceName}, sm_{context.Sm}");
        Assert.True(context.Sm > 0, "The device reported no compute capability.");
    }
}
